"""streaming.py: assembling a streamed model response, for `Runner.run_streamed`.

`run_streamed` runs the same turn loop as `run` (`run_loop._run_async`, with an `emit` callback),
so guardrails, approvals, tracing, hooks and sessions behave identically. The only
streaming-specific step is here: consuming `Model.stream_response` instead of `get_response`,
emitting each raw delta as it arrives.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from runa._models import Model
from runa._types import ModelResponse, Usage
from runa.stream_events import RawResponsesStreamEvent, StreamEvent

Emit = Callable[[StreamEvent], None]


async def _stream_response(model: Model, request: tuple[Any, ...], emit: Emit) -> ModelResponse:
    """Consume `model.stream_response(*request)` into a `ModelResponse`, emitting each delta."""
    text_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    usage = Usage()

    async for delta in model.stream_response(*request):
        emit(RawResponsesStreamEvent(data=delta))
        if delta.text:
            text_parts.append(delta.text)
        if delta.tool_call_index is not None:
            entry = tool_calls.setdefault(
                delta.tool_call_index,
                {"id": None, "type": "function", "function": {"name": None, "arguments": ""}},
            )
            if delta.tool_call_id:
                entry["id"] = delta.tool_call_id
            if delta.tool_call_name:
                entry["function"]["name"] = delta.tool_call_name
            if delta.tool_call_arguments:
                entry["function"]["arguments"] += delta.tool_call_arguments
        if delta.usage is not None:
            usage = delta.usage

    message = {
        "role": "assistant",
        "content": "".join(text_parts) or None,
        "tool_calls": [tool_calls[i] for i in sorted(tool_calls)] or None,
    }
    return ModelResponse(output=[message], usage=usage)


__all__ = ["Emit", "_stream_response"]
