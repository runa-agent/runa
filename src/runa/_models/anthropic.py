"""anthropic.py: the Claude backend, the one that needs real translation.

Anthropic's Messages API isn't chat-completions-shaped: content blocks instead of a `tool_calls`
array, a separate `system` param, strict user/assistant alternation, and its own streaming events.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

from anthropic import APIError, AsyncAnthropic

from runa._items import (
    answered_call_id,
    assistant_message,
    call_arguments,
    call_id,
    call_name,
    content_text,
    parsed_arguments,
    provider_blocks,
    tool_call,
    tool_calls,
)
from runa._models.interface import (
    ModelRequest,
    StreamDelta,
    ToolSchema,
    output_json_schema,
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
                "tool_use_id": answered_call_id(message),
                "content": content_text(message.get("content")),
            }
        ]
    if role == "assistant":
        # Thinking first, as Claude itself ordered it: the blocks this turn kept are the ones
        # `_to_chat_message` had no half of the chat-completions shape to put them in.
        blocks: list[dict[str, Any]] = list(provider_blocks(message))
        text = content_text(message.get("content"))
        if text:
            blocks.append({"type": "text", "text": text})
        for call in tool_calls(message):
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call_id(call),
                    "name": call_name(call),
                    "input": parsed_arguments(call_arguments(call)),
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
        output_tokens_details=OutputTokensDetails(reasoning_tokens=_thinking_tokens(usage)),
    )


def _thinking_tokens(usage: Any) -> int:
    """How much of `usage`'s output went on thinking, `0` when the response broke out nothing.

    Claude bills thinking inside `output_tokens` and itemizes it as `thinking_tokens`;
    chat-completions providers call the same figure `reasoning_tokens`. Both land on
    `Usage.output_tokens_details`, so one `reasoning_effort` doesn't report two different ways
    depending on which vendor `model` named.
    """
    details = usage.output_tokens_details
    return details.thinking_tokens if details else 0


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
        tools = [_to_anthropic_tool(schema) for schema in request.tools]

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

        # Claude spells both of these inside one `output_config`, so they're built together.
        output_config: dict[str, Any] = {}
        if settings.reasoning_effort is not None:
            # `effort` is the depth knob, and it only bites with thinking on -- which current
            # Claude models run adaptively, so there is no token budget for Runa to guess at.
            body["thinking"] = {"type": "adaptive"}
            output_config["effort"] = settings.reasoning_effort
        schema = output_json_schema(request)
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if output_config:
            body["output_config"] = output_config
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
    """Convert an Anthropic `Message` into a chat-completions-shaped assistant message dict.

    Words and calls go in the two halves that shape has; every other block kind -- `thinking`
    and `redacted_thinking` today, whatever Claude adds next without this function changing --
    is kept verbatim in the item's provider slot, as JSON the next request can hand straight
    back. Dropping one is not free: a thinking block carries the signature the model needs its
    own reasoning back by, so a turn that loses it pays to think from scratch on the next.
    """
    text_parts: list[str] = []
    calls: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    for block in message.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            calls.append(tool_call(block.id, block.name, json.dumps(block.input)))
        else:
            blocks.append(block.to_dict(mode="json"))
    return assistant_message("".join(text_parts), calls, blocks)


async def _anthropic_deltas(stream: AsyncIterator[Any]) -> AsyncIterator[StreamDelta]:
    """Turn Anthropic's raw SSE events into `StreamDelta`s.

    A block Runa has no half of the chat-completions shape for is assembled here rather than
    streamed: it only means anything whole (a thinking block's signature arrives last), so it's
    grown across its deltas and yielded once, at the `content_block_stop` that finishes it. What
    a stream appends to history then matches what `get_response` returns for the same turn,
    which is the property that keeps a streamed tool-calling loop replayable.
    """
    input_tokens = 0
    blocks: dict[int, dict[str, Any]] = {}
    async for event in stream:
        if event.type == "message_start":
            input_tokens = event.message.usage.input_tokens
        elif event.type == "content_block_start":
            block = event.content_block
            if block.type == "tool_use":
                yield StreamDelta(
                    tool_call_index=event.index,
                    tool_call_id=block.id,
                    tool_call_name=block.name,
                    tool_call_arguments="",
                )
            elif block.type != "text":
                blocks[event.index] = block.to_dict(mode="json")
        elif event.type == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta":
                yield StreamDelta(text=delta.text)
            elif delta.type == "input_json_delta":
                yield StreamDelta(
                    tool_call_index=event.index, tool_call_arguments=delta.partial_json
                )
            elif event.index in blocks:
                _grow(blocks[event.index], delta)
        elif event.type == "content_block_stop" and event.index in blocks:
            yield StreamDelta(provider_blocks=[blocks[event.index]])
        elif event.type == "message_delta" and event.usage is not None:
            output_tokens = event.usage.output_tokens or 0
            yield StreamDelta(
                usage=Usage(
                    requests=1,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    output_tokens_details=OutputTokensDetails(
                        reasoning_tokens=_thinking_tokens(event.usage)
                    ),
                )
            )


def _grow(block: dict[str, Any], delta: Any) -> None:
    """Fold one streamed delta into the block it belongs to, extending each field it names.

    A thinking block arrives as a `thinking_delta` per fragment and a `signature_delta` at the
    end, and the next kind Anthropic streams will arrive the same way, so the rule is the shape
    of the event rather than a branch per name: a delta's every string field is a piece of the
    same-named field of its block.
    """
    for name, value in delta.to_dict(mode="json").items():
        if name != "type" and isinstance(value, str):
            block[name] = block.get(name, "") + value


__all__ = ["AnthropicModel"]
