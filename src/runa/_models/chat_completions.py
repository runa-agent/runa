"""chat_completions.py: the chat-completions-shaped backend.

Covers OpenAI, Gemini, Llama, DeepSeek, and Qwen, every provider that speaks this wire format.
"""

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

import httpx2 as httpx

from runa._items import assistant_message, wire_items
from runa._models import transport
from runa._models.interface import (
    ModelRequest,
    StreamDelta,
    ToolSchema,
    output_json_schema,
)
from runa._types import (
    InputTokensDetails,
    ModelResponse,
    OutputTokensDetails,
    ToolChoice,
    Usage,
)
from runa.exceptions import ServiceUnavailable, UserError

_CHAT_COMPLETIONS_PATH = "chat/completions"


def _wire_tool(schema: ToolSchema) -> dict[str, Any]:
    """Wrap a `ToolSchema` in the chat-completions `{"type": "function", ...}` envelope."""
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.parameters,
        },
    }


def _openai_tool_choice(tool_choice: ToolChoice) -> Any:
    """Map Runa's `ToolChoice` to the chat-completions `tool_choice` shape.

    `"auto"`/`"required"`/`"none"`/`None` pass straight through; any other string is a specific
    tool name, which the wire format wants wrapped in `{"type": "function", "function": {...}}`.
    """
    if tool_choice is None or tool_choice in ("auto", "required", "none"):
        return tool_choice
    return {"type": "function", "function": {"name": tool_choice}}


def _usage_from_openai(usage: dict[str, Any]) -> Usage:
    """Build a `Usage` from a chat-completions response's `usage` object."""
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return Usage(
        requests=1,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        input_tokens_details=InputTokensDetails(
            cached_tokens=prompt_details.get("cached_tokens", 0)
        ),
        output_tokens_details=OutputTokensDetails(
            reasoning_tokens=completion_details.get("reasoning_tokens", 0)
        ),
    )


def _openai_deltas(chunk: dict[str, Any]) -> Iterator[StreamDelta]:
    """Turn one chat-completions streaming chunk into zero or more `StreamDelta`s."""
    usage = chunk.get("usage")
    if usage:
        yield StreamDelta(usage=_usage_from_openai(usage))
    choices = chunk.get("choices") or []
    if not choices:
        return
    delta = choices[0].get("delta") or {}
    if delta.get("content"):
        yield StreamDelta(text=delta["content"])
    for call in delta.get("tool_calls") or []:
        function = call.get("function") or {}
        yield StreamDelta(
            tool_call_index=call["index"],
            tool_call_id=call.get("id"),
            tool_call_name=function.get("name"),
            tool_call_arguments=function.get("arguments"),
        )


class OpenAICompatibleModel:
    """Talks to any chat-completions-shaped provider directly over HTTP, via `httpx2`.

    Built from where the provider lives and which environment variable holds its key, never from
    a client: each request goes through `runa._models.transport`'s client for the running event
    loop, so one instance serves `run_sync`'s loop, an app's own and a test's `asyncio.run` alike,
    with the same timeout and retries as every provider Runa names.
    """

    def __init__(self, model: str, *, base_url: str, api_key_env: str) -> None:
        """Call `model` at `base_url`, sending the key in `api_key_env` as a bearer token.

        The variable is read on each request, not here, so an instance is declarable on a class
        body before the key is set, and a rotated key is picked up by the next request.
        """
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env

    def _client(self) -> httpx.AsyncClient:
        """This loop's client for `base_url`; raises `UserError` when `api_key_env` is unset."""
        api_key = os.environ.get(self.api_key_env)
        if api_key is None:
            raise UserError(
                f"{self.api_key_env} is not set. Set it to call {self.model} at {self.base_url}."
            )
        return transport.client(self.base_url, api_key)

    def _body(self, request: ModelRequest) -> dict[str, Any]:
        """The JSON body one `request` puts on the wire."""
        settings = request.model_settings
        body: dict[str, Any] = {"model": self.model, "messages": wire_items(request.messages)}
        tools = [_wire_tool(schema) for schema in request.tools]
        if tools:
            body["tools"] = tools
        if settings.tool_choice is not None:
            body["tool_choice"] = _openai_tool_choice(settings.tool_choice)
        if settings.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = settings.parallel_tool_calls
        if settings.temperature is not None:
            body["temperature"] = settings.temperature
        if settings.top_p is not None:
            body["top_p"] = settings.top_p
        if settings.max_tokens is not None:
            body["max_tokens"] = settings.max_tokens
        if settings.reasoning_effort is not None:
            body["reasoning_effort"] = settings.reasoning_effort
        schema = output_json_schema(request)
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": schema},
            }
        return body

    async def get_response(self, request: ModelRequest) -> ModelResponse:
        """Send one turn to the model and return its full response."""
        body = self._body(request)
        client = self._client()
        response = await transport.send(
            lambda: client.post(_CHAT_COMPLETIONS_PATH, json=body),
            request.model_settings.max_retries,
        )
        transport.raise_for_status(response.status_code, response.text)
        data = response.json()
        choice = data["choices"][0]["message"]
        message = assistant_message(choice.get("content"), choice.get("tool_calls"))
        usage = _usage_from_openai(data.get("usage") or {})
        return ModelResponse(output=[message], usage=usage, response_id=data.get("id"))

    async def stream_response(self, request: ModelRequest) -> AsyncIterator[StreamDelta]:
        """Send one turn to the model and yield incremental `StreamDelta`s as it responds."""
        body = {
            **self._body(request),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        client = self._client()
        http_request = client.build_request("POST", _CHAT_COMPLETIONS_PATH, json=body)
        response = await transport.send(
            lambda: client.send(http_request, stream=True),
            request.model_settings.max_retries,
        )
        try:
            if response.status_code >= 400:
                body = await response.aread()
                transport.raise_for_status(
                    response.status_code, body.decode("utf-8", errors="replace")
                )
            async for sse in httpx.EventSource(response):
                if sse.data == "[DONE]":
                    continue
                for delta in _openai_deltas(cast(dict[str, Any], sse.json())):
                    yield delta
        except httpx.TransportError as exc:  # mid-stream: too late to retry, events went out
            raise ServiceUnavailable(f"model stream failed: {exc!r}") from exc
        finally:
            await response.aclose()


__all__ = ["OpenAICompatibleModel"]
