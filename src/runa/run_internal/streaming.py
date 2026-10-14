"""streaming.py: assembling a streamed model response, for `Agent.run_streamed`.

`run_streamed` runs the same turn loop as `run` (`run_loop._run_async`, with an `emit` callback),
so guardrails, approvals, tracing, hooks and sessions behave identically. The only
streaming-specific step is here: consuming `Model.stream_response` instead of `get_response`,
emitting each raw delta as it arrives.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from runa._items import assistant_message, tool_call
from runa._models import Model, ModelRequest
from runa._types import ModelResponse, Usage
from runa.stream_events import RawResponsesStreamEvent, StreamEvent

Emit = Callable[[StreamEvent], None]


@dataclass
class _PartialCall:
    """One tool call mid-arrival: `id`/`name` land once, `arguments` in however many pieces."""

    id: str = ""
    name: str = ""
    arguments: str = ""


async def _stream_response(model: Model, request: ModelRequest, emit: Emit) -> ModelResponse:
    """Consume `model.stream_response(request)` into a `ModelResponse`, emitting each delta."""
    text_parts: list[str] = []
    calls: dict[int, _PartialCall] = {}
    blocks: list[Any] = []
    usage = Usage()

    async for delta in model.stream_response(request):
        emit(RawResponsesStreamEvent(data=delta))
        if delta.text:
            text_parts.append(delta.text)
        if delta.provider_blocks:
            blocks.extend(delta.provider_blocks)
        if delta.tool_call_index is not None:
            call = calls.setdefault(delta.tool_call_index, _PartialCall())
            if delta.tool_call_id:
                call.id = delta.tool_call_id
            if delta.tool_call_name:
                call.name = delta.tool_call_name
            if delta.tool_call_arguments:
                call.arguments += delta.tool_call_arguments
        if delta.usage is not None:
            usage = delta.usage

    message = assistant_message(
        "".join(text_parts),
        [tool_call(calls[i].id, calls[i].name, calls[i].arguments) for i in sorted(calls)],
        blocks,
    )
    return ModelResponse(output=[message], usage=usage)


__all__ = ["Emit", "_stream_response"]
