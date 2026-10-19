"""streaming.py: assembling a streamed model response, for `Agent.run_streamed`.

`run_streamed` runs the same turn loop as `run` (with an `emit` callback),
so guardrails, approvals, tracing, hooks and sessions behave identically. The only
streaming-specific step is here: consuming `Model.stream_response` instead of `get_response`,
emitting each piece of text as a `TextDelta` as it arrives. A provider's delta never reaches the
stream's consumer itself.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from runa._items import assistant_message, tool_call
from runa._models import Model, ModelRequest, StreamDelta
from runa._types import ModelResponse, Usage
from runa.stream_events import StreamEvent, TextDelta

Emit = Callable[[StreamEvent], None]


@dataclass
class _PartialCall:
    """One tool call mid-arrival: `id`/`name` land once, `arguments` in however many pieces."""

    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class _Assembly:
    """A streamed response put back together, one `StreamDelta` at a time."""

    text_parts: list[str] = field(default_factory=list)
    calls: dict[int, _PartialCall] = field(default_factory=dict)
    blocks: list[Any] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def add(self, delta: StreamDelta) -> None:
        """Fold `delta` into the response so far."""
        if delta.text:
            self.text_parts.append(delta.text)
        if delta.provider_blocks:
            self.blocks.extend(delta.provider_blocks)
        if delta.tool_call_index is not None:
            call = self.calls.setdefault(delta.tool_call_index, _PartialCall())
            if delta.tool_call_id:
                call.id = delta.tool_call_id
            if delta.tool_call_name:
                call.name = delta.tool_call_name
            if delta.tool_call_arguments:
                call.arguments += delta.tool_call_arguments
        if delta.usage is not None:
            self.usage = delta.usage

    def response(self) -> ModelResponse:
        """Every delta added so far, as the `ModelResponse` the same call would have returned."""
        calls = self.calls
        message = assistant_message(
            "".join(self.text_parts),
            [tool_call(calls[i].id, calls[i].name, calls[i].arguments) for i in sorted(calls)],
            self.blocks,
        )
        return ModelResponse(output=[message], usage=self.usage)


async def _stream_response(model: Model, request: ModelRequest, emit: Emit) -> ModelResponse:
    """Consume `model.stream_response(request)` into a `ModelResponse`, emitting its text."""
    assembly = _Assembly()
    async for delta in model.stream_response(request):
        assembly.add(delta)
        if delta.text:
            emit(TextDelta(delta.text))
    return assembly.response()


__all__ = ["Emit", "_Assembly", "_stream_response"]
