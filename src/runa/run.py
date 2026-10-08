"""`Run`: the result of `Agent.run()`/`run_sync()`, and `RunStream`, of `Agent.run_streamed()`.

One result shape, built once by the turn loop (`run_internal/run_loop._finish`) and handed back
unchanged. The loop used to return a `RunResult` of its own that the Agent translated field by
field into this one, which meant two dataclasses carrying the same ten values and a translator
that had to be kept honest between them. What the loop needs and a caller doesn't -- the context
wrapper, the input it started from, the items it generated -- is here too, underscore-prefixed:
private fields on the one result, rather than a second result shape to convert from.
"""

from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal

from pydantic import BaseModel

from runa._items import ConversationItem
from runa._types import RunContextWrapper, Usage
from runa.exceptions import UserError
from runa.guardrail import GuardrailAudit, GuardrailResults
from runa.run_state import Interruption, RunState
from runa.stream_events import StreamEvent
from runa.tool import ToolCall
from runa.tracing import Trace

Status = Literal["completed", "paused", "error"]


def _jsonable(value: Any) -> Any:
    """Render an agent's `output` for JSON, including a dataclass or Pydantic `output_type`."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return str(value)


@dataclass
class Run(GuardrailAudit):
    """The outcome of a single `Agent.run()`/`run_sync()` call.

    `output` is the agent's final output (parsed into `Agent.output_type` when it declares one),
    `None` unless `status` is `"completed"`. `trace` is the hierarchical `Trace` (agent/LLM/tool/
    handoff/guardrail spans) captured for this call, see `runa.tracing`. `usage` is this call's
    token usage, same value as `Agent.last_usage` after the call.

    `status` is `"paused"` when a tool call needs human approval: resolve each of
    `interruptions` on `to_state()`, then pass that state back to `run`/`run_sync` in place of a
    message. It is `"error"` when a `RunaError` (a guardrail tripwire, `MaxTurnsExceeded`, a
    model error, ...) stopped the run; `error` then holds that exception's message.

    `guardrail_results` is the guardrail audit trail: every guardrail that ran (tripped or not, a
    delegate's included), keyed by the `Phase` it ran in, whatever the `status`. The four
    `*_guardrail_results` lists `GuardrailAudit` reads off it are the documented way in. The same
    trail is on a paused `RunState`.

    `to_payload()` is how a transport renders all of this as JSON -- one wire shape, owned here
    rather than re-derived per transport.
    """

    output: Any
    trace: Trace | None
    usage: Usage
    status: Status = "completed"
    error: str | None = None
    interruptions: list[Interruption] = field(default_factory=list)
    guardrail_results: GuardrailResults = field(default_factory=GuardrailResults)
    _state: RunState | None = field(default=None, repr=False)
    _context_wrapper: RunContextWrapper | None = field(default=None, repr=False)
    _original_input: list[ConversationItem] = field(default_factory=list, repr=False)
    _generated_items: list[ConversationItem] = field(default_factory=list, repr=False)
    _tool_calls: list[ToolCall] = field(default_factory=list, repr=False)
    """Every tool call this run executed, as the turn loop saw it (see `ToolCall`). Private because
    `trace` is how a caller inspects a run: this is the unfiltered copy `runa.eval` grades, which
    a trace span can't be, since spans pass through the tracing privacy policy."""

    def to_state(self) -> RunState:
        """The paused run's `RunState`: approve or reject its `interruptions`, then resume."""
        if self._state is None:
            raise UserError(f"to_state() needs a paused run, this one is {self.status!r}")
        return self._state

    def to_payload(self) -> dict[str, Any]:
        """This run as plain JSON-compatible data: what a `Run` looks like over the wire.

        A transport (`runa serve`'s routes, `runa.web`, whatever ships next) renders a run by
        calling this, not by re-listing the fields itself. The shape is a fact about `Run`, so it
        belongs where `Run` is defined: the alternative is each transport re-deriving it, which is
        how `serve` came to report a `tool_name` that `Interruption` never had.

        `status` is the `Run`'s own, so a caller distinguishes "the agent answered" from "it needs
        an approval" from "it failed" without parsing prose. `trace_id` is the handle for
        `runa traces show`, which is what makes a production incident debuggable. A paused run
        carries `state`, `RunState.to_json()`'s blob: the one thing that can resolve the pause,
        handed back through `RunState.from_json` to resume (see `runa.run_state`). It is `None`
        for any other status.

        One-way, unlike `RunState.to_json()`: a `Run` holds live objects (an `Agent`, a `Trace`'s
        spans) and there is no `from_payload`. The guardrail audit trail is left out for the same
        reason `RunState` omits it -- a guardrail's `output_info` is arbitrary and not guaranteed
        JSON-safe. Plain dicts only, no transport imported here, so this stays a runtime concern.
        """
        return {
            "status": self.status,
            "output": _jsonable(self.output),
            "error": self.error,
            "trace_id": self.trace.id if self.trace else None,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.total_tokens,
                "requests": self.usage.requests,
            },
            "interruptions": [
                {
                    "name": item.name,
                    "arguments": item.arguments,
                    "call_id": item.call_id,
                    "agent": item.agent.name,
                }
                for item in self.interruptions
            ],
            "state": self._state.to_json() if self._state is not None else None,
        }

    def _history(self) -> list[ConversationItem]:
        """`original_input + generated_items`: the conversation as it stands after this run.

        What `Agent.run` writes back to `self.history` on a session-less run. Private because a
        session-backed run's history lives in the session, and `Agent.history` is the supported
        way to read the other kind.
        """
        return [*self._original_input, *self._generated_items]


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
