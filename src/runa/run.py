"""`Run`: the result of `Agent.run()`/`run_sync()`, and `RunStream`, of `Agent.run_streamed()`.

One result shape, built once by the turn loop and handed back unchanged. The run's `Transcript`,
which the loop needs and a caller doesn't, is an underscore-prefixed field on it.
"""

from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from runa._types import Usage
from runa.exceptions import UserError
from runa.guardrail import GuardrailAudit, GuardrailResults
from runa.stream_events import StreamEvent
from runa.tool import ToolCall
from runa.tracing import Trace
from runa.transcript import Transcript

if TYPE_CHECKING:
    from runa.run_state import Interruption, RunState

RunStatus = Literal["completed", "paused", "error"]
"""What `Run.status` can be: see `Run`."""


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
    handoff/guardrail spans) captured for this call, see `runa.tracing`; always present, whatever
    the `status`, because the trace is opened before anything a run can fail on. `usage` is this
    call's token usage, every model and embedding call made on its behalf included, same value
    as `Agent.last_usage` after the call.

    `status` is `"paused"` when a tool call needs human approval: resolve each of
    `interruptions` on `to_state()`, then pass that state back to `run`/`run_sync` in place of a
    message. The pause is already kept in `runa.db` under `run_id`, so another process resumes it
    from `RunState.load(agent, run_id)` (see `RunState`). It is `"error"` when a
    `RunaError` (a guardrail tripwire, `MaxTurnsExceeded`, a model error, ...) stopped the run;
    `error` then holds that exception's message.

    `guardrail_results` is the guardrail audit trail: every guardrail that ran (tripped or not, a
    delegate's included), keyed by the `Phase` it ran in, whatever the `status`. The four
    `*_guardrail_results` lists `GuardrailAudit` reads off it are the documented way in. The same
    trail is on a paused `RunState`.

    `to_payload()` is how a transport renders all of this as JSON -- one wire shape, owned here
    rather than re-derived per transport.
    """

    output: Any
    trace: Trace
    usage: Usage
    status: RunStatus = "completed"
    error: str | None = None
    interruptions: list[Interruption] = field(default_factory=list)
    guardrail_results: GuardrailResults = field(default_factory=GuardrailResults)
    _state: RunState | None = field(default=None, repr=False)
    _transcript: Transcript = field(default_factory=Transcript, repr=False)
    """The run's conversation (see `Transcript`): `history()` is what `Agent.run` writes back to
    `Agent.history`. Private because `trace` is how a caller inspects a run, and a session holds
    a session-backed run's history. Empty for a run that failed: an unfinished turn is not
    history."""

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """Every tool call this run answered, in the order they finished (see `ToolCall`).

        What the tool was actually given and returned, unlike a `"tool"` span in `trace`, whose
        input and output the tracing privacy policy may withhold or redact: this is the record to
        grade a run's tool use on. A failed call is one of them, its `error` saying why, and so
        is a `.delegate` call, the calls its own run made as its `children`.
        """
        return self._transcript.tool_calls

    @property
    def run_id(self) -> str | None:
        """The id a paused run is kept under in `runa.db`, for `RunState.load`; else `None`."""
        return self._state.run_id if self._state is not None else None

    def to_state(self) -> RunState:
        """The paused run's `RunState`: approve or reject its `interruptions`, then resume."""
        if self._state is None:
            raise UserError(f"to_state() needs a paused run, this one is {self.status!r}")
        return self._state

    def to_payload(self) -> dict[str, Any]:
        """This run as plain JSON-compatible data: what a `Run` looks like over the wire.

        Every transport (`runa serve`'s routes, `runa.web`) renders a run by calling this rather
        than listing the fields itself.

        `status` is the `Run`'s own, so a caller distinguishes "the agent answered" from "it needs
        an approval" from "it failed" without parsing prose. `trace_id` is the handle for
        `runa traces show`, which is what makes a production incident debuggable.

        A paused run's `RunState` is not part of it, only its `run_id`: the state carries the
        run's approval ledger, which stays in `runa.db`, so a client holds a handle to resume by.

        One-way: a `Run` holds live objects (an `Agent`, a `Trace`'s spans) and there is no
        `from_payload`. The guardrail audit trail is left out for the same
        reason `RunState` omits it -- every guardrail that ran is already a span in the trace
        `trace_id` names, tripped or not. Plain dicts only, no transport imported here, so this
        stays a runtime concern.
        """
        return {
            "status": self.status,
            "output": _jsonable(self.output),
            "error": self.error,
            "trace_id": self.trace.id,
            "run_id": self.run_id,
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
        }


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


__all__ = ["Run", "RunStatus", "RunStream"]
