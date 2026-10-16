"""active_run.py: `_Run`, the one value the turn loop and every step under it is passed.

A run is one object from the moment the loop starts it to the moment it becomes the caller's `Run`:
the agent, its `Transcript`, the `RunContext` user code is handed and the governance it isn't, the
span everything hangs under, and the hooks, config and `emit` this particular call was made with.
The turn loop and the steps it calls take this, not a re-listing of its fields, so something a run
starts carrying doesn't ripple through a dozen signatures.

It lives here rather than in `run_loop` for one reason: `tool_execution` takes a `_Run` too, and
it has no business importing the loop that calls it.
"""

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

from runa._governance import _Governance
from runa._types import RunContext
from runa.lifecycle import _Dispatch
from runa.run_internal.agent_shape import AgentShape
from runa.run_internal.run_config import RunConfig
from runa.run_internal.spans import _Spans
from runa.run_internal.streaming import Emit
from runa.run_state import RunState
from runa.session import Session
from runa.stream_events import StreamEvent
from runa.tracing.spans import Span
from runa.tracing.traces import Trace
from runa.transcript import Transcript


@dataclass
class _Run:
    """What a fresh run and a resumed one share once the turn loop starts.

    `shape` is the agent running now, resolved; `start` is the one the run began with. They are
    the same object until a handoff, and separate names afterwards because the two readings
    differ: who answered, and what the caller asked for. Memory extraction and compaction are
    the caller's agent's settings, so they read `start`; the turn itself reads `shape`.
    """

    shape: AgentShape
    transcript: Transcript
    run_context: RunContext
    governance: _Governance
    trace: Trace
    agent_span: Span
    session: Session | None
    hooks: _Dispatch[Any]
    run_config: RunConfig
    emit: Emit | None = None
    resumed: RunState | None = None
    """The paused state this run is continuing, if it is one: the unfinished tool-call message,
    the decisions that unblock its calls, and the results already computed for the rest of them.
    The loop reads those off the state itself rather than off a shape built from it -- a resumed
    turn needs no fact the state it came from doesn't already carry."""
    start: AgentShape = field(init=False)

    def __post_init__(self) -> None:
        self.start = self.shape

    @property
    def current_agent(self) -> Any:
        """The agent running now, as user code sees it: a hook's callback, a guardrail, a pause."""
        return self.shape.agent

    @property
    def spans(self) -> _Spans:
        """Where this run's steps put their spans: children of its one agent span."""
        return _Spans(self.trace, self.agent_span.id)

    def span(self, name: str, span_type: Any, *, input: Any = None) -> AbstractContextManager[Span]:
        """A span under this run's agent span, for a `with` block (see `_Spans.span`)."""
        return self.spans.span(name, span_type, input=input)

    def notify(self, event: StreamEvent) -> None:
        """Hand `event` to a streamed run's consumer; a non-streamed run has nowhere to put it."""
        if self.emit is not None:
            self.emit(event)


__all__ = ["_Run"]
