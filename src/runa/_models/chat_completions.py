"""chat_completions.py: the chat-completions-shaped backend.

Covers OpenAI, Gemini, Llama, DeepSeek, and Qwen, every provider that speaks this wire format.
"""

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, cast

import httpx2 as httpx

from runa._models.interface import (
    ModelRequest,
    StreamDelta,
    ToolSchema,
    output_json_schema,
    tool_schemas,
)
from runa._types import (
    InputTokensDetails,
    ModelResponse,
    OutputTokensDetails,
    ToolChoice,
    Usage,
)
from runa.exceptions import ModelBehaviorError

_CHAT_COMPLETIONS_PATH = "chat/completions"
DEFAULT_MAX_RETRIES = 2  # the same default as the `anthropic` SDK's, so both backends agree
_MAX_BACKOFF = 8.0


def _retryable(status_code: int) -> bool:
    return status_code in (408, 409, 429) or status_code >= 500


def _backoff(attempt: int, retry_after: str | None = None) -> float:
    """Seconds to wait before retry `attempt`: the server's `retry-after`, else jittered 2^n."""
    if retry_after is not None:
        try:
            return min(float(retry_after), _MAX_BACKOFF)
        except ValueError:
            pass
    return min(0.5 * 2**attempt, _MAX_BACKOFF) * random.uniform(0.75, 1.0)


async def _with_retries(
    send: Callable[[], Awaitable[httpx.Response]], max_retries: int | None = None
) -> httpx.Response:
    """`send()` until it succeeds, retrying connection errors and 408/409/429/5xx with backoff.

    `max_retries` is `ModelSettings.max_retries`: `None` means `DEFAULT_MAX_RETRIES`, and `0`
    disables retrying. Whatever still fails after the last attempt raises `ModelBehaviorError`.
    """
    retries = DEFAULT_MAX_RETRIES if max_retries is None else max(0, max_retries)
    attempt = 0
    while True:
        last = attempt == retries
        try:
            response = await send()
        except httpx.TransportError as exc:
            if last:
                raise ModelBehaviorError(f"model request failed: {exc!r}") from exc
            await asyncio.sleep(_backoff(attempt))
        else:
            if last or not _retryable(response.status_code):
                return response
            await response.aclose()
            await asyncio.sleep(_backoff(attempt, response.headers.get("retry-after")))
        attempt += 1


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


def _raise_for_status(status_code: int, body: str) -> None:
    """Raise `ModelBehaviorError` for a failed request; `body` is capped for legibility."""
    if status_code < 400:
        return
    raise ModelBehaviorError(f"model request failed with {status_code}: {body[:2000]}")


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
    """Talks to any chat-completions-shaped provider directly over HTTP, via `httpx2`."""

    def __init__(self, model: str, client: httpx.AsyncClient) -> None:
        """Store the model name and the shared, provider-scoped HTTP client to call it through."""
        self.model = model
        self._client = client

    def _body(self, request: ModelRequest) -> dict[str, Any]:
        """The JSON body one `request` puts on the wire."""
        settings = request.model_settings
        body: dict[str, Any] = {"model": self.model, "messages": request.messages}
        tools = [_wire_tool(schema) for schema in tool_schemas(request)]
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
        response = await _with_retries(
            lambda: self._client.post(_CHAT_COMPLETIONS_PATH, json=body),
            request.model_settings.max_retries,
        )
        _raise_for_status(response.status_code, response.text)
        data = response.json()
        choice = data["choices"][0]["message"]
        message: dict[str, Any] = {
            "role": "assistant",
            "content": choice.get("content"),
            "tool_calls": choice.get("tool_calls") or None,
        }
        usage = _usage_from_openai(data.get("usage") or {})
        return ModelResponse(output=[message], usage=usage, response_id=data.get("id"))

    async def stream_response(self, request: ModelRequest) -> AsyncIterator[StreamDelta]:
        """Send one turn to the model and yield incremental `StreamDelta`s as it responds."""
        body = {
            **self._body(request),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        http_request = self._client.build_request("POST", _CHAT_COMPLETIONS_PATH, json=body)
        response = await _with_retries(
            lambda: self._client.send(http_request, stream=True),
            request.model_settings.max_retries,
        )
        try:
            if response.status_code >= 400:
                body = await response.aread()
                _raise_for_status(response.status_code, body.decode("utf-8", errors="replace"))
            async for sse in httpx.EventSource(response):
                if sse.data == "[DONE]":
                    continue
                for delta in _openai_deltas(cast(dict[str, Any], sse.json())):
                    yield delta
        except httpx.TransportError as exc:  # mid-stream: too late to retry, events went out
            raise ModelBehaviorError(f"model stream failed: {exc!r}") from exc
        finally:
            await response.aclose()


__all__ = ["OpenAICompatibleModel"]
