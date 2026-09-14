"""anthropic.py: the Claude backend, the one that needs real translation.

Anthropic's Messages API isn't chat-completions-shaped: content blocks instead of a `tool_calls`
array, a separate `system` param, strict user/assistant alternation, and its own streaming events.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

from runa._models.interface import StreamDelta, _handoff_dict, _tool_dict
from runa._types import (
    InputTokensDetails,
    ModelResponse,
    ModelSettings,
    OutputTokensDetails,
    ToolChoice,
    TResponseInputItem,
    Usage,
)
from runa.exceptions import UserError


def _check_plain_text_output(output_schema: type | None) -> None:
    """Reject structured output: `AnthropicModel` doesn't translate JSON response formats."""
    if output_schema is not None and output_schema is not str:
        raise UserError(
            "AnthropicModel does not support structured output schemas; use a plain-text "
            "output type (the default, or `str`) for Claude models."
        )


def _text_content(content: Any) -> str:
    """Flatten a chat-completions `content` field (string or a list of parts) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(
        part["text"]
        for part in content
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )


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
            text = _text_content(message.get("content"))
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
                "content": _text_content(message.get("content")),
            }
        ]
    if role == "assistant":
        blocks: list[dict[str, Any]] = []
        text = _text_content(message.get("content"))
        if text:
            blocks.append({"type": "text", "text": text})
        for call in message.get("tool_calls") or []:
            function = call["function"]
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call["id"],
                    "name": function["name"],
                    "input": json.loads(function["arguments"] or "{}"),
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


def _to_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool["function"]
    return {
        "name": function["name"],
        "description": function.get("description") or "",
        "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
    }


def _to_anthropic_tool_choice(tool_choice: ToolChoice) -> dict[str, Any] | None:
    """Map Runa's `ToolChoice` to Anthropic's `tool_choice` shape."""
    if tool_choice is None:
        return None
    if tool_choice == "auto":
        return {"type": "auto"}
    if tool_choice == "required":
        return {"type": "any"}
    if tool_choice == "none":
        return {"type": "none"}
    return {"type": "tool", "name": tool_choice}


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
    """Talks to Claude directly through the `anthropic` SDK's Messages API."""

    def __init__(self, model: str, client: AsyncAnthropic) -> None:
        """Store the model name and the shared Anthropic client to call it through."""
        self.model = model
        self._client = client

    def _request(
        self,
        system_instructions: str | None,
        input: list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Any],
        handoffs: list[Any],
    ) -> dict[str, Any]:
        messages = list(input)
        if system_instructions:
            messages = [{"role": "system", "content": system_instructions}, *messages]
        system, turns = _to_anthropic_messages(messages)

        wire_tools = [_tool_dict(t) for t in tools] + [_handoff_dict(h) for h in handoffs]

        request: dict[str, Any] = {
            "model": self.model,
            "messages": turns,
            "max_tokens": model_settings.max_tokens or 4096,
        }
        if system:
            request["system"] = system
        if wire_tools:
            request["tools"] = [_to_anthropic_tool(t) for t in wire_tools]
        tool_choice = _to_anthropic_tool_choice(model_settings.tool_choice)
        if tool_choice:
            request["tool_choice"] = tool_choice
        if model_settings.temperature is not None:
            request["temperature"] = model_settings.temperature
        if model_settings.top_p is not None:
            request["top_p"] = model_settings.top_p
        return request

    async def get_response(
        self,
        system_instructions: str | None,
        input: list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Any],
        output_schema: type | None,
        handoffs: list[Any],
    ) -> ModelResponse:
        """Send one turn to the model and return its full response."""
        _check_plain_text_output(output_schema)
        request = self._request(system_instructions, input, model_settings, tools, handoffs)
        response = await self._client.messages.create(**request)

        message: dict[str, Any] = _to_chat_message(response)
        usage = _to_usage(response.usage)
        return ModelResponse(output=[message], usage=usage, response_id=response.id)

    async def stream_response(
        self,
        system_instructions: str | None,
        input: list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Any],
        output_schema: type | None,
        handoffs: list[Any],
    ) -> AsyncIterator[StreamDelta]:
        """Send one turn to the model and yield incremental `StreamDelta`s as it responds."""
        _check_plain_text_output(output_schema)
        request = self._request(system_instructions, input, model_settings, tools, handoffs)
        raw_stream = await self._client.messages.create(stream=True, **request)
        async for delta in _anthropic_deltas(raw_stream):
            yield delta


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


__all__ = [
    "AnthropicModel",
    "_anthropic_deltas",
    "_check_plain_text_output",
    "_to_anthropic_content",
    "_to_anthropic_image",
    "_to_anthropic_messages",
    "_to_anthropic_tool",
    "_to_anthropic_tool_choice",
    "_to_chat_message",
    "_to_usage",
]
