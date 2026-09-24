"""`Run`: the result of `Agent.run()`/`run_sync()`, and `RunStream`, of `Agent.run_streamed()`."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from runa._types import Usage
from runa.exceptions import UserError
from runa.guardrail import GuardrailResult
from runa.run_state import Interruption, RunState
from runa.stream_events import StreamEvent
from runa.tracing import Trace

Status = Literal["completed", "paused", "error"]


@dataclass
class Run:
    """The outcome of a single `Agent.run()`/`run_sync()` call.

    `output` is the agent's final output (parsed into `Agent.output_type` when it declares one),
    `None` unless `status` is `"completed"`. `trace` is the hierarchical `Trace` (agent/LLM/tool/
    handoff/guardrail spans) captured for this call, see `runa.tracing`. `usage` is this call's
    token usage, same value as `Agent.last_usage` after the call.

    `status` is `"paused"` when a tool call needs human approval: resolve each of
    `interruptions` on `to_state()`, then pass that state back to `run`/`run_sync` in place of a
    message. It is `"error"` when a `RunaError` (a guardrail tripwire, `MaxTurnsExceeded`, a
    model error, ...) stopped the run; `error` then holds that exception's message.

    The four `*_guardrail_results` lists are the guardrail audit trail: every guardrail that ran
    (tripped or not, a delegate's included), by where it ran, whatever the `status`. The same
    four are on a paused `RunState`.
    """

    output: Any
    trace: Trace | None
    usage: Usage
    status: Status = "completed"
    error: str | None = None
    interruptions: list[Interruption] = field(default_factory=list)
    input_guardrail_results: list[GuardrailResult] = field(default_factory=list)
    output_guardrail_results: list[GuardrailResult] = field(default_factory=list)
    tool_input_guardrail_results: list[GuardrailResult] = field(default_factory=list)
    tool_output_guardrail_results: list[GuardrailResult] = field(default_factory=list)
    _state: RunState | None = field(default=None, repr=False)

    def to_state(self) -> RunState:
        """The paused run's `RunState`: approve or reject its `interruptions`, then resume."""
        if self._state is None:
            raise UserError(f"to_state() needs a paused run, this one is {self.status!r}")
        return self._state


class RunStream:
    """What `Agent.run_streamed()` returns: iterate it for `StreamEvent`s.

    Once iteration ends, `run` holds the same `Run` that `run()` would have returned, paused,
    completed, or errored.
    """

    def __init__(self, events: AsyncIterator[StreamEvent]) -> None:
        """Wrap `events`; `run` is set by the agent once they are exhausted."""
        self._events = events
        self.run: Run | None = None

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        """Iterate the run's `StreamEvent`s."""
        return self._events


__all__ = ["Run", "RunStream", "Status"]
