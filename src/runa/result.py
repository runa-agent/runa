"""result.py: `RunResult`/`RunResultStreaming`, what `Runner` returns to a caller."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from runa._types import RunContextWrapper, TResponseInputItem
from runa.run_state import Interruption, RunState
from runa.stream_events import StreamEvent
from runa.tracing.traces import Trace


@dataclass
class RunResult:
    """The outcome of one `Runner.run`/`run_sync` call."""

    final_output: Any
    context_wrapper: RunContextWrapper
    trace: Trace
    _original_input: list[TResponseInputItem]
    _generated_items: list[TResponseInputItem]
    interruptions: list[Interruption] = field(default_factory=list)
    _state: RunState | None = None
    input_guardrail_results: list[Any] = field(default_factory=list)
    output_guardrail_results: list[Any] = field(default_factory=list)
    tool_input_guardrail_results: list[Any] = field(default_factory=list)
    tool_output_guardrail_results: list[Any] = field(default_factory=list)

    def to_input_list(self) -> list[TResponseInputItem]:
        """Return `original_input + generated_items`: the full history after this run."""
        return [*self._original_input, *self._generated_items]

    def to_state(self) -> RunState:
        """Return the `RunState` to resolve `interruptions` against and resume with."""
        assert self._state is not None, "to_state() needs a run that actually paused"
        return self._state


class RunResultStreaming:
    """What `Runner.run_streamed` returns: an async iterator of `StreamEvent`s.

    Iterating runs the same turn loop as `Runner.run`, so guardrails, tracing, hooks and sessions
    behave identically. Once the iterator is fully consumed, `result` holds the finished
    `RunResult`; an error raised by the run is re-raised from the iterator.
    """

    def __init__(
        self,
        run: Callable[[Callable[[StreamEvent], None]], Awaitable[RunResult]],
        context_wrapper: RunContextWrapper,
    ) -> None:
        """Store `run` (the turn loop, given an `emit` callback) to start on first iteration."""
        self.context_wrapper = context_wrapper
        self.result: RunResult | None = None
        self._events = self._stream(run)

    async def _stream(
        self, run: Callable[[Callable[[StreamEvent], None]], Awaitable[RunResult]]
    ) -> AsyncGenerator[StreamEvent]:
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        task = asyncio.ensure_future(run(queue.put_nowait))
        task.add_done_callback(lambda _: queue.put_nowait(None))
        try:
            while (event := await queue.get()) is not None:
                yield event
            self.result = task.result()
        finally:
            task.cancel()

    def __aiter__(self) -> AsyncGenerator[StreamEvent]:
        """Iterate the `StreamEvent`s this run produces."""
        return self._events

    @property
    def final_output(self) -> Any:
        """The run's final output, once the stream is fully consumed (else `None`)."""
        return self.result.final_output if self.result is not None else None

    def to_input_list(self) -> list[TResponseInputItem]:
        """Return the full history after this run, once the stream is fully consumed."""
        return self.result.to_input_list() if self.result is not None else []


__all__ = ["RunResult", "RunResultStreaming"]
