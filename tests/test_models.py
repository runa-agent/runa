"""Tests for `runa._models`: the two-backend router that replaces `openai-agents`/`openai`."""

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import httpx2 as httpx
import pytest
from pydantic import BaseModel

from runa import content
from runa._models import AnthropicModel, ModelProvider, ModelRequest, OpenAICompatibleModel
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
            ModelRequest(input=[{"role": "user", "content": "hi"}], tools=[_Tool("weather")])
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
        return await model.get_response(ModelRequest(input=[{"role": "user", "content": "hi"}]))

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
                ModelRequest(input=[{"role": "user", "content": "hi"}])
            )
        ]

    deltas = asyncio.run(collect())

    assert deltas[0].text == "Hi"
    assert deltas[1].tool_call_id == "call_1"
    assert deltas[1].tool_call_name == "weather"
    assert deltas[2].usage is not None
    assert deltas[2].usage.input_tokens == 3


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {}}
    )


def _get_response(model: OpenAICompatibleModel, output_schema: Any = None) -> Any:
    return asyncio.run(
        model.get_response(
            ModelRequest(input=[{"role": "user", "content": "hi"}], output_schema=output_schema)
        )
    )


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record retry delays instead of sleeping through them."""
    delays: list[float] = []

    async def sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("runa._models.chat_completions.asyncio.sleep", sleep)
    return delays


def test_openai_compatible_retries_rate_limits_and_server_errors(no_backoff: list[float]) -> None:
    """A 429 then a 503 are retried with backoff, honoring `retry-after`, until a 200 lands."""
    replies = [
        httpx.Response(429, headers={"retry-after": "3"}),
        httpx.Response(503),
        _ok_response(),
    ]
    model = OpenAICompatibleModel("gpt-5.4-nano", _mock_client(lambda _: replies.pop(0)))

    assert _get_response(model).output[0]["content"] == "ok"
    assert no_backoff[0] == 3.0
    assert len(no_backoff) == 2


def test_openai_compatible_gives_up_after_max_retries(no_backoff: list[float]) -> None:
    """Still failing after the retries, the last status surfaces as a `ModelBehaviorError`."""
    from runa.exceptions import ModelBehaviorError

    model = OpenAICompatibleModel("gpt-5.4-nano", _mock_client(lambda _: httpx.Response(500)))

    with pytest.raises(ModelBehaviorError, match="500"):
        _get_response(model)
    assert len(no_backoff) == 2


def test_openai_compatible_retries_connection_errors(no_backoff: list[float]) -> None:
    """A dropped connection is retried, then raised as a `ModelBehaviorError`, not a raw error."""
    from runa.exceptions import ModelBehaviorError

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    model = OpenAICompatibleModel("gpt-5.4-nano", _mock_client(handler))

    with pytest.raises(ModelBehaviorError, match="ConnectError"):
        _get_response(model)
    assert len(no_backoff) == 2


def test_openai_compatible_stream_retries_before_the_first_event(no_backoff: list[float]) -> None:
    """A stream that opens on a 503 is retried; nothing was emitted yet, so it's safe to."""
    replies = [
        httpx.Response(503),
        httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\ndata: [DONE]\n\n',
        ),
    ]
    model = OpenAICompatibleModel("gpt-5.4-nano", _mock_client(lambda _: replies.pop(0)))

    async def collect() -> list[Any]:
        return [
            d
            async for d in model.stream_response(
                ModelRequest(input=[{"role": "user", "content": "hi"}])
            )
        ]

    assert [d.text for d in asyncio.run(collect())] == ["Hi"]
    assert len(no_backoff) == 1


class _Answer(BaseModel):
    city: str
    temperature: int


def test_openai_compatible_asks_for_the_output_types_json_schema() -> None:
    """A structured `output_type` is sent as a closed `json_schema` response format."""
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _ok_response()

    _get_response(OpenAICompatibleModel("gpt-5.4-nano", _mock_client(handler)), _Answer)

    response_format = seen[0]["response_format"]
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]["schema"]
    assert set(schema["properties"]) == {"city", "temperature"}
    assert schema["additionalProperties"] is False


async def _events(events: list[Any]) -> AsyncIterator[Any]:
    for event in events:
        yield event


def _anthropic_message(content: list[Any] | None = None, usage: Any | None = None) -> Any:
    """A canned Anthropic `Message`, the shape the SDK hands back from `messages.create`."""
    return SimpleNamespace(
        id="msg_1",
        content=content if content is not None else [SimpleNamespace(type="text", text="ok")],
        usage=usage
        or SimpleNamespace(
            input_tokens=0,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


class _FakeAnthropic:
    """Stands in for `AsyncAnthropic`: records every request, answers with a canned reply."""

    def __init__(self, reply: Any = None, events: list[Any] | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self._reply = reply if reply is not None else _anthropic_message()
        self._events = events or []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **request: Any) -> Any:
        self.requests.append(request)
        return _events(self._events) if request.get("stream") else self._reply


def _anthropic_request(
    input: list[Any],
    model_settings: ModelSettings | None = None,
    tools: list[Any] | None = None,
    output_schema: type | None = None,
) -> dict[str, Any]:
    """The Anthropic request one `get_response` call puts on the wire."""
    client = _FakeAnthropic()
    model = AnthropicModel("claude-sonnet-5", client)  # type: ignore[arg-type]
    asyncio.run(
        model.get_response(
            ModelRequest(
                input=input,
                model_settings=model_settings or ModelSettings(),
                tools=tools or [],
                output_schema=output_schema,
            )
        )
    )
    return client.requests[0]


def test_anthropic_get_response_parses_text_and_tool_calls() -> None:
    """A Claude message's text and tool_use blocks come back as one chat-completions item."""
    reply = _anthropic_message(
        content=[
            SimpleNamespace(type="text", text="Checking the weather."),
            SimpleNamespace(type="tool_use", id="call_1", name="weather", input={"city": "NYC"}),
        ],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=2,
            cache_creation_input_tokens=1,
        ),
    )
    model = AnthropicModel("claude-sonnet-5", _FakeAnthropic(reply))  # type: ignore[arg-type]

    async def call() -> Any:
        return await model.get_response(ModelRequest(input=[{"role": "user", "content": "hi"}]))

    response = asyncio.run(call())

    assert response.output == [
        {
            "role": "assistant",
            "content": "Checking the weather.",
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
    assert response.usage.total_tokens == 15
    assert response.usage.input_tokens_details.cached_tokens == 2
    assert response.response_id == "msg_1"


def test_anthropic_stream_response_yields_text_and_tool_call_deltas() -> None:
    """Anthropic's SSE events become `StreamDelta`s with text, tool-call fragments, and usage."""
    client = _FakeAnthropic(
        events=[
            SimpleNamespace(
                type="message_start", message=SimpleNamespace(usage=SimpleNamespace(input_tokens=7))
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text="Hi"),
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
    )
    model = AnthropicModel("claude-sonnet-5", client)  # type: ignore[arg-type]

    async def collect() -> list[Any]:
        return [
            d
            async for d in model.stream_response(
                ModelRequest(input=[{"role": "user", "content": "hi"}])
            )
        ]

    deltas = asyncio.run(collect())

    assert deltas[0].text == "Hi"
    assert deltas[1].tool_call_id == "call_1"
    assert deltas[1].tool_call_name == "weather"
    assert deltas[2].tool_call_index == 1
    assert deltas[2].tool_call_arguments == '{"city":'
    assert deltas[3].usage is not None
    assert deltas[3].usage.input_tokens == 7
    assert deltas[3].usage.output_tokens == 4


def test_anthropic_asks_for_the_output_types_json_schema() -> None:
    """On Claude, a structured `output_type` becomes `output_config.format`; text sends none."""
    message: list[Any] = [{"role": "user", "content": "hi"}]

    output_format = _anthropic_request(message, output_schema=_Answer)["output_config"]["format"]

    assert output_format["type"] == "json_schema"
    assert output_format["schema"]["required"] == ["city", "temperature"]
    assert output_format["schema"]["additionalProperties"] is False
    assert "output_config" not in _anthropic_request(message, output_schema=str)


def test_anthropic_api_errors_surface_as_model_behavior_errors() -> None:
    """An `anthropic.APIError` the SDK's own retries couldn't fix becomes a `RunaError`."""
    import anthropic

    from runa.exceptions import ModelBehaviorError

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    async def create(**_: Any) -> Any:
        raise anthropic.APIConnectionError(request=request)

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    model = AnthropicModel("claude-sonnet-5", client=client)  # type: ignore[arg-type]

    with pytest.raises(ModelBehaviorError, match="model request failed"):
        asyncio.run(model.get_response(ModelRequest(input=[{"role": "user", "content": "hi"}])))


def test_anthropic_splits_system_and_merges_consecutive_tool_results() -> None:
    """Two tool replies to one multi-tool-call turn merge into a single Anthropic user message."""
    request = _anthropic_request(
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "weather in two cities?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "weather", "arguments": '{"city": "NYC"}'},
                    },
                    {
                        "id": "call_2",
                        "function": {"name": "weather", "arguments": '{"city": "SF"}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            {"role": "tool", "tool_call_id": "call_2", "content": "foggy"},
        ]
    )

    assert request["system"] == "Be terse."
    assert [turn["role"] for turn in request["messages"]] == ["user", "assistant", "user"]
    tool_results = request["messages"][2]["content"]
    assert [block["tool_use_id"] for block in tool_results] == ["call_1", "call_2"]
    assert [block["content"] for block in tool_results] == ["sunny", "foggy"]


def test_anthropic_assistant_turn_carries_text_and_tool_use_blocks() -> None:
    """An assistant message with both text and a tool call becomes two Anthropic content blocks."""
    request = _anthropic_request(
        [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "weather", "arguments": '{"city": "NYC"}'},
                    }
                ],
            }
        ]
    )

    assert request["messages"][0]["content"] == [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "call_1", "name": "weather", "input": {"city": "NYC"}},
    ]


def test_anthropic_user_turn_translates_text_and_image_parts() -> None:
    """A `runa.content` parts list keeps its text blocks and translates `image_url` parts."""
    request = _anthropic_request(
        [
            {
                "role": "user",
                "content": [
                    content.text("what is this?"),
                    content.image("https://host.test/c.png"),
                ],
            }
        ]
    )

    assert request["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image", "source": {"type": "url", "url": "https://host.test/c.png"}},
            ],
        }
    ]


def test_anthropic_sends_a_data_uri_image_as_a_base64_source() -> None:
    """A `data:` URI image part becomes Anthropic's base64 source, media type and data split out."""
    request = _anthropic_request(
        [{"role": "user", "content": [content.image("data:image/png;base64,aGVsbG8=")]}]
    )

    assert request["messages"][0]["content"] == [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="},
        }
    ]


def test_anthropic_drops_a_turn_with_no_content() -> None:
    """A string content becomes one text block; empty/`None` content sends no turn at all."""
    assert _anthropic_request([{"role": "user", "content": "hi"}])["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]
    assert _anthropic_request([{"role": "user", "content": ""}])["messages"] == []
    assert _anthropic_request([{"role": "user", "content": None}])["messages"] == []


def test_anthropic_translates_a_tools_function_schema() -> None:
    """A Runa tool reaches Anthropic in its own `input_schema` shape, not OpenAI's `parameters`."""
    request = _anthropic_request([{"role": "user", "content": "hi"}], tools=[_Tool("weather")])

    assert request["tools"] == [
        {
            "name": "weather",
            "description": "Get the weather.",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]


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
def test_anthropic_tool_choice(tool_choice: Any, expected: dict[str, Any] | None) -> None:
    """Each `ToolChoice` value maps to Anthropic's own `tool_choice` shape."""
    request = _anthropic_request(
        [{"role": "user", "content": "hi"}],
        ModelSettings(tool_choice=tool_choice),
        tools=[_Tool("weather")],
    )

    assert request.get("tool_choice") == expected


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        (None, {"type": "auto", "disable_parallel_tool_use": True}),
        ("auto", {"type": "auto", "disable_parallel_tool_use": True}),
        ("required", {"type": "any", "disable_parallel_tool_use": True}),
        ("weather", {"type": "tool", "name": "weather", "disable_parallel_tool_use": True}),
        ("none", {"type": "none"}),
    ],
)
def test_anthropic_parallel_tool_calls_false_disables_parallel_tool_use(
    tool_choice: Any, expected: dict[str, Any]
) -> None:
    """`parallel_tool_calls=False` becomes `disable_parallel_tool_use` on Anthropic's choice."""
    request = _anthropic_request(
        [{"role": "user", "content": "hi"}],
        ModelSettings(tool_choice=tool_choice, parallel_tool_calls=False),
        tools=[_Tool("weather")],
    )

    assert request["tool_choice"] == expected
