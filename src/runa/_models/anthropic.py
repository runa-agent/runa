"""anthropic.py: the Claude backend, the one that needs real translation.

Anthropic's Messages API isn't chat-completions-shaped: content blocks instead of a `tool_calls`
array, a separate `system` param, strict user/assistant alternation, and its own streaming events.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

from anthropic import APIError, AsyncAnthropic

from runa._items import content_text, parsed_arguments
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
    ModelSettings,
    OutputTokensDetails,
    ToolChoice,
    Usage,
)
from runa.exceptions import ModelBehaviorError


def _to_anthropic_messages(messages: list[Any]) -> tuple[str | None, list[dict[str, Any]]]:
    """Split chat-completions messages into Anthropic's `system` string and `messages` turns.

    Anthropic requires strict user/assistant alternation, so consecutive same-role turns (most
    commonly several tool results answering one multi-tool-call assistant turn) are merged into
    one message rather than sent back to back.
    """
    system_parts: list[str] = []
    turns: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role in ("system", "developer"):
            text = content_text(message.get("content"))
            if text:
                system_parts.append(text)
            continue

        anthropic_role, blocks = _to_anthropic_turn(message)
        if not blocks:
            continue
        if turns and turns[-1]["role"] == anthropic_role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": anthropic_role, "content": blocks})

    return "\n".join(system_parts) or None, turns


def _to_anthropic_turn(message: Any) -> tuple[str, list[dict[str, Any]]]:
    role = message.get("role")
    if role == "tool":
        return "user", [
            {
                "type": "tool_result",
                "tool_use_id": message["tool_call_id"],
                "content": content_text(message.get("content")),
            }
        ]
    if role == "assistant":
        blocks: list[dict[str, Any]] = []
        text = content_text(message.get("content"))
        if text:
            blocks.append({"type": "text", "text": text})
        for call in message.get("tool_calls") or []:
            function = call["function"]
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call["id"],
                    "name": function["name"],
                    "input": parsed_arguments(function["arguments"]),
                }
            )
        return "assistant", blocks
    return "user", _to_anthropic_content(message.get("content"))


def _to_anthropic_content(content: Any) -> list[dict[str, Any]]:
    """Turn a chat-completions `content` field into Anthropic content blocks.

    A plain string becomes one text block. A list of parts keeps OpenAI-shaped `text` parts as
    text blocks and translates `image_url` parts (see `runa.content.image`) into Anthropic's own
    `image` block, whether the URL is a `data:` URI (decoded to a base64 source) or a plain
    `http(s)` URL (an Anthropic url source).
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            blocks.append({"type": "text", "text": part["text"]})
        elif part.get("type") == "image_url":
            blocks.append(_to_anthropic_image(part["image_url"]))
    return blocks


def _to_anthropic_image(image_url: Any) -> dict[str, Any]:
    """Map an OpenAI-shaped `image_url` part to Anthropic's `image` content block."""
    url = image_url["url"] if isinstance(image_url, dict) else image_url
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = header.removeprefix("data:").split(";")[0]
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _to_anthropic_tool(schema: ToolSchema) -> dict[str, Any]:
    """Map a `ToolSchema` to Anthropic's tool definition, which calls the schema `input_schema`."""
    return {
        "name": schema.name,
        "description": schema.description,
        "input_schema": schema.parameters,
    }


def _to_anthropic_tool_choice(
    tool_choice: ToolChoice, parallel_tool_calls: bool | None = None
) -> dict[str, Any] | None:
    """Map Runa's `ToolChoice` (and `parallel_tool_calls=False`) to Anthropic's `tool_choice`.

    Anthropic has no top-level parallel flag: `disable_parallel_tool_use` rides on `tool_choice`,
    defaulting it to `auto` when no choice was given.
    """
    if tool_choice == "none":
        return {"type": "none"}
    choice: dict[str, Any] | None
    if tool_choice is None:
        choice = {"type": "auto"} if parallel_tool_calls is False else None
    elif tool_choice == "auto":
        choice = {"type": "auto"}
    elif tool_choice == "required":
        choice = {"type": "any"}
    else:
        choice = {"type": "tool", "name": tool_choice}
    if choice is not None and parallel_tool_calls is False:
        choice["disable_parallel_tool_use"] = True
    return choice


def _to_usage(usage: Any) -> Usage:
    return Usage(
        requests=1,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.input_tokens + usage.output_tokens,
        input_tokens_details=InputTokensDetails(
            cached_tokens=usage.cache_read_input_tokens or 0,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
        ),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


class AnthropicModel:
    """Talks to Claude directly through the `anthropic` SDK's Messages API.

    Retries (connection errors, 408/409/429/5xx, with backoff) are the SDK's own, capped at
    `ModelSettings.max_retries` so both backends take the same knob; a request that still fails
    raises `ModelBehaviorError`, like the chat-completions backend.
    """

    def __init__(self, model: str, client: AsyncAnthropic) -> None:
        """Store the model name and the shared Anthropic client to call it through."""
        self.model = model
        self._client = client

    def _messages(self, model_settings: ModelSettings) -> Any:
        """The client's `messages` resource, with `ModelSettings.max_retries` applied if set."""
        if model_settings.max_retries is None:
            return self._client.messages
        return self._client.with_options(max_retries=max(0, model_settings.max_retries)).messages

    def _body(self, request: ModelRequest) -> dict[str, Any]:
        """The Messages API body one `request` puts on the wire."""
        settings = request.model_settings
        system, turns = _to_anthropic_messages(request.messages)
        tools = [_to_anthropic_tool(schema) for schema in tool_schemas(request)]

        body: dict[str, Any] = {
            "model": self.model,
            "messages": turns,
            "max_tokens": settings.max_tokens or 4096,
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = tools
        tool_choice = _to_anthropic_tool_choice(
            settings.tool_choice, settings.parallel_tool_calls if tools else None
        )
        if tool_choice:
            body["tool_choice"] = tool_choice
        if settings.temperature is not None:
            body["temperature"] = settings.temperature
        if settings.top_p is not None:
            body["top_p"] = settings.top_p
        schema = output_json_schema(request)
        if schema is not None:
            body["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        return body

    async def get_response(self, request: ModelRequest) -> ModelResponse:
        """Send one turn to the model and return its full response."""
        body = self._body(request)
        try:
            response = await self._messages(request.model_settings).create(**body)
        except APIError as exc:
            raise ModelBehaviorError(f"model request failed: {exc}") from exc

        message: dict[str, Any] = _to_chat_message(response)
        usage = _to_usage(response.usage)
        return ModelResponse(output=[message], usage=usage, response_id=response.id)

    async def stream_response(self, request: ModelRequest) -> AsyncIterator[StreamDelta]:
        """Send one turn to the model and yield incremental `StreamDelta`s as it responds."""
        body = self._body(request)
        try:
            raw_stream = await self._messages(request.model_settings).create(stream=True, **body)
            async for delta in _anthropic_deltas(raw_stream):
                yield delta
        except APIError as exc:
            raise ModelBehaviorError(f"model request failed: {exc}") from exc


def _to_chat_message(message: Any) -> dict[str, Any]:
    """Convert an Anthropic `Message` into a chat-completions-shaped assistant message dict."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in message.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": json.dumps(block.input)},
                }
            )
    return {
        "role": "assistant",
        "content": "".join(text_parts) or None,
        "tool_calls": tool_calls or None,
    }


async def _anthropic_deltas(stream: AsyncIterator[Any]) -> AsyncIterator[StreamDelta]:
    """Turn Anthropic's raw SSE events into `StreamDelta`s."""
    input_tokens = 0
    async for event in stream:
        if event.type == "message_start":
            input_tokens = event.message.usage.input_tokens
        elif event.type == "content_block_start" and event.content_block.type == "tool_use":
            block = event.content_block
            yield StreamDelta(
                tool_call_index=event.index,
                tool_call_id=block.id,
                tool_call_name=block.name,
                tool_call_arguments="",
            )
        elif event.type == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta":
                yield StreamDelta(text=delta.text)
            elif delta.type == "input_json_delta":
                yield StreamDelta(
                    tool_call_index=event.index, tool_call_arguments=delta.partial_json
                )
        elif event.type == "message_delta" and event.usage is not None:
            output_tokens = event.usage.output_tokens or 0
            yield StreamDelta(
                usage=Usage(
                    requests=1,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                )
            )


__all__ = ["AnthropicModel"]
