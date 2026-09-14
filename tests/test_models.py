"""Tests for `runa._models`: the two-backend router that replaces `openai-agents`/`openai`."""

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import httpx2 as httpx
import pytest

from runa import content
from runa._models import (
    AnthropicModel,
    ModelProvider,
    OpenAICompatibleModel,
    _anthropic_deltas,
    _check_plain_text_output,
    _to_anthropic_content,
    _to_anthropic_image,
    _to_anthropic_messages,
    _to_anthropic_tool,
    _to_anthropic_tool_choice,
    _to_chat_message,
    _to_usage,
)
from runa._types import ModelSettings
from runa.exceptions import UserError


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every provider key starts unset; individual tests set only what they need."""
    for env in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "LLAMA_API_KEY",
        "DEEPSEEK_API_KEY",
        "DASHSCOPE_API_KEY",
    ):
        monkeypatch.delenv(env, raising=False)


def test_routes_gpt_to_openai_compatible_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `gpt-*` model gets `OpenAICompatibleModel` pointed at OpenAI's own endpoint."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model = ModelProvider().get_model("gpt-5.4-nano")

    assert isinstance(model, OpenAICompatibleModel)
    assert str(model._client.base_url) == "https://api.openai.com/v1/"


def test_unrecognized_name_defaults_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model name with none of the five prefixes still resolves through OpenAI, like before."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model = ModelProvider().get_model("o3-mini")

    assert isinstance(model, OpenAICompatibleModel)


@pytest.mark.parametrize(
    ("model_name", "env", "base_url"),
    [
        (
            "gemini-2.5-pro",
            "GEMINI_API_KEY",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        ("llama-3.1-70b", "LLAMA_API_KEY", "https://api.llama.com/compat/v1/"),
        ("deepseek-chat", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1/"),
        (
            "qwen-max",
            "DASHSCOPE_API_KEY",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/",
        ),
    ],
)
def test_routes_each_openai_compatible_provider(
    monkeypatch: pytest.MonkeyPatch, model_name: str, env: str, base_url: str
) -> None:
    """Gemini/Llama/DeepSeek/Qwen each get an `OpenAICompatibleModel` at their own base_url."""
    monkeypatch.setenv(env, "key")
    model = ModelProvider().get_model(model_name)

    assert isinstance(model, OpenAICompatibleModel)
    assert str(model._client.base_url) == base_url


def test_routes_claude_to_the_anthropic_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `claude-*` model gets `AnthropicModel`, not the OpenAI-compatible path."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    model = ModelProvider().get_model("claude-sonnet-4-6")

    assert isinstance(model, AnthropicModel)


def test_missing_provider_key_raises_a_clear_error() -> None:
    """An unset provider-specific key names itself, not OPENAI_API_KEY, in the error."""
    with pytest.raises(UserError, match="DEEPSEEK_API_KEY"):
        ModelProvider().get_model("deepseek-chat")


def test_reuses_one_client_per_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two calls for the same provider share a client instead of opening a new one each time."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    provider = ModelProvider()

    first = provider.get_model("deepseek-chat")
    second = provider.get_model("deepseek-coder")

    assert isinstance(first, OpenAICompatibleModel)
    assert isinstance(second, OpenAICompatibleModel)
    assert first._client is second._client


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = "Get the weather."
        self.params_json_schema = {"type": "object", "properties": {"city": {"type": "string"}}}


def _mock_client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="https://example.test/v1/", transport=httpx.MockTransport(handler)
    )


def test_openai_compatible_get_response_parses_text_and_tool_calls() -> None:
    """A chat-completions response's message becomes one chat-completions-shaped output item."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "gpt-5.4-nano"
        assert body["tools"][0]["function"]["name"] == "weather"
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Checking.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "weather", "arguments": '{"city": "NYC"}'},
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    model = OpenAICompatibleModel("gpt-5.4-nano", _mock_client(handler))

    async def call() -> Any:
        return await model.get_response(
            None, [{"role": "user", "content": "hi"}], ModelSettings(), [_Tool("weather")], None, []
        )

    response = asyncio.run(call())

    assert response.output == [
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city": "NYC"}'},
                }
            ],
        }
    ]
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5
    assert response.response_id == "resp_1"


def test_openai_compatible_get_response_raises_on_error_status() -> None:
    """A non-2xx response is surfaced as a `ModelBehaviorError` instead of a raw HTTP error."""
    from runa.exceptions import ModelBehaviorError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad request"})

    model = OpenAICompatibleModel("gpt-5.4-nano", _mock_client(handler))

    async def call() -> Any:
        return await model.get_response(
            None, [{"role": "user", "content": "hi"}], ModelSettings(), [], None, []
        )

    with pytest.raises(ModelBehaviorError, match="400"):
        asyncio.run(call())


def test_openai_compatible_stream_response_yields_text_and_tool_call_deltas() -> None:
    """SSE chunks become `StreamDelta`s carrying text, tool-call fragments, and final usage."""
    sse_body = (
        b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n'
        b'data: {"choices": [{"delta": {"tool_calls": '
        b'[{"index": 0, "id": "call_1", "function": {"name": "weather", "arguments": ""}}]}}]}\n\n'
        b'data: {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, '
        b'"total_tokens": 5}}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse_body)

    model = OpenAICompatibleModel("gpt-5.4-nano", _mock_client(handler))

    async def collect() -> list[Any]:
        return [
            d
            async for d in model.stream_response(
                None, [{"role": "user", "content": "hi"}], ModelSettings(), [], None, []
            )
        ]

    deltas = asyncio.run(collect())

    assert deltas[0].text == "Hi"
    assert deltas[1].tool_call_id == "call_1"
    assert deltas[1].tool_call_name == "weather"
    assert deltas[2].usage is not None
    assert deltas[2].usage.input_tokens == 3


def test_split_system_and_merges_consecutive_tool_results() -> None:
    """Two tool replies to one multi-tool-call turn merge into a single Anthropic user message."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "weather in two cities?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "function": {"name": "weather", "arguments": '{"city": "NYC"}'}},
                {"id": "call_2", "function": {"name": "weather", "arguments": '{"city": "SF"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        {"role": "tool", "tool_call_id": "call_2", "content": "foggy"},
    ]

    system, turns = _to_anthropic_messages(messages)

    assert system == "Be terse."
    assert [t["role"] for t in turns] == ["user", "assistant", "user"]
    tool_results = turns[2]["content"]
    assert [b["tool_use_id"] for b in tool_results] == ["call_1", "call_2"]
    assert [b["content"] for b in tool_results] == ["sunny", "foggy"]


def test_assistant_turn_carries_text_and_tool_use_blocks() -> None:
    """An assistant message with both text and a tool call becomes two Anthropic content blocks."""
    messages = [
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "weather", "arguments": '{"city": "NYC"}'}}
            ],
        }
    ]

    _, turns = _to_anthropic_messages(messages)

    assert turns[0]["content"] == [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "call_1", "name": "weather", "input": {"city": "NYC"}},
    ]


def test_to_anthropic_content_flattens_a_plain_string() -> None:
    """A bare string content becomes one text block; empty/`None` content becomes no blocks."""
    assert _to_anthropic_content("hi") == [{"type": "text", "text": "hi"}]
    assert _to_anthropic_content("") == []
    assert _to_anthropic_content(None) == []


def test_to_anthropic_content_translates_text_and_image_parts() -> None:
    """A `runa.content` parts list keeps its text blocks and translates `image_url` parts."""
    parts = [content.text("what is this?"), content.image("https://example.test/cat.png")]

    assert _to_anthropic_content(parts) == [
        {"type": "text", "text": "what is this?"},
        {"type": "image", "source": {"type": "url", "url": "https://example.test/cat.png"}},
    ]


def test_to_anthropic_image_decodes_a_data_uri_to_a_base64_source() -> None:
    """A `data:` URI image part becomes Anthropic's base64 source, media type and data split out."""
    block = _to_anthropic_image({"url": "data:image/png;base64,aGVsbG8="})

    assert block == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="},
    }


def test_to_anthropic_image_keeps_a_plain_url_as_a_url_source() -> None:
    """An `http(s)` image part becomes Anthropic's own url source, not re-encoded."""
    block = _to_anthropic_image({"url": "https://example.test/cat.png"})

    assert block == {
        "type": "image",
        "source": {"type": "url", "url": "https://example.test/cat.png"},
    }


def test_user_turn_with_an_image_survives_the_full_message_split() -> None:
    """A user message with mixed text/image content keeps both blocks through the full pipeline."""
    messages = [
        {
            "role": "user",
            "content": [content.text("describe this"), content.image("https://example.test/x.png")],
        }
    ]

    _, turns = _to_anthropic_messages(messages)

    assert turns == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image", "source": {"type": "url", "url": "https://example.test/x.png"}},
            ],
        }
    ]


def test_to_anthropic_tool_translates_the_function_schema() -> None:
    """A chat-completions-shaped function-tool dict maps to Anthropic's own shape."""
    tool = {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get the weather.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }

    assert _to_anthropic_tool(tool) == {
        "name": "weather",
        "description": "Get the weather.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        ("auto", {"type": "auto"}),
        ("required", {"type": "any"}),
        ("none", {"type": "none"}),
        ("weather", {"type": "tool", "name": "weather"}),
        (None, None),
    ],
)
def test_to_anthropic_tool_choice(tool_choice: Any, expected: dict[str, Any] | None) -> None:
    """Each `ToolChoice` value maps to Anthropic's own `tool_choice` shape."""
    assert _to_anthropic_tool_choice(tool_choice) == expected


def test_to_chat_message_collects_text_and_tool_use() -> None:
    """A Claude response's text and tool_use blocks become message content and tool_calls."""
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Checking the weather."),
            SimpleNamespace(type="tool_use", id="call_1", name="weather", input={"city": "NYC"}),
        ],
    )

    result = _to_chat_message(message)

    assert result["content"] == "Checking the weather."
    assert result["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "weather", "arguments": '{"city": "NYC"}'},
        }
    ]


def test_to_usage_converts_anthropic_token_counts() -> None:
    """Anthropic's usage fields map onto Runa's own `Usage`, cache fields included."""
    usage = SimpleNamespace(
        input_tokens=10, output_tokens=5, cache_read_input_tokens=2, cache_creation_input_tokens=1
    )

    result = _to_usage(usage)

    assert result.requests == 1
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.total_tokens == 15
    assert result.input_tokens_details.cached_tokens == 2


def test_check_plain_text_output_allows_none_and_str_and_rejects_a_schema() -> None:
    """A structured output type on a Claude model raises, since translation isn't implemented."""
    _check_plain_text_output(None)
    _check_plain_text_output(str)

    with pytest.raises(UserError):
        _check_plain_text_output(dict)


async def _stream(events: list[SimpleNamespace]) -> AsyncIterator[SimpleNamespace]:
    for event in events:
        yield event


def test_anthropic_deltas_carries_text_and_tool_call_fragments() -> None:
    """Anthropic's SSE events become `StreamDelta`s with text, tool-call fragments, and usage."""
    events = [
        SimpleNamespace(
            type="message_start", message=SimpleNamespace(usage=SimpleNamespace(input_tokens=7))
        ),
        SimpleNamespace(
            type="content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text="Hi")
        ),
        SimpleNamespace(
            type="content_block_start",
            index=1,
            content_block=SimpleNamespace(type="tool_use", id="call_1", name="weather"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"city":'),
        ),
        SimpleNamespace(type="message_delta", usage=SimpleNamespace(output_tokens=4)),
    ]

    async def collect() -> list[Any]:
        return [d async for d in _anthropic_deltas(_stream(events))]

    deltas = asyncio.run(collect())

    assert deltas[0].text == "Hi"
    assert deltas[1].tool_call_id == "call_1"
    assert deltas[1].tool_call_name == "weather"
    assert deltas[2].tool_call_index == 1
    assert deltas[2].tool_call_arguments == '{"city":'
    assert deltas[3].usage is not None
    assert deltas[3].usage.input_tokens == 7
    assert deltas[3].usage.output_tokens == 4
