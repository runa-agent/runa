"""active_run.py: `_Run`, the one value the turn loop and every step under it is passed.

A run is one object from the moment `_run_async` builds it to the moment `_finish` turns it into
the caller's `Run`: the agent, the conversation so far, the context wrapper, the span everything
hangs under, and the hooks, config and `emit` this particular call was made with. `_run_turns`
and the steps it calls take this, not a re-listing of its fields, so something a run starts
carrying doesn't ripple through a dozen signatures.

It lives here rather than in `run_loop` for one reason: `tool_execution` takes a `_Run` too, and
it has no business importing the loop that calls it.
"""

from dataclasses import dataclass, field
from typing import Any

from runa._items import ConversationItem
from runa._types import RunContextWrapper
from runa.lifecycle import _Dispatch
from runa.run_internal.agent_shape import AgentShape
from runa.run_internal.run_config import RunConfig
from runa.run_internal.spans import _Spans
from runa.run_internal.streaming import Emit
from runa.run_state import RunState
from runa.session import Session
from runa.stream_events import StreamEvent
from runa.tool import ToolCall
from runa.tracing.spans import Span
from runa.tracing.traces import Trace


@dataclass
class _Run:
    """What a fresh run and a resumed one share once the turn loop starts.

    `shape` is the agent running now, resolved; `start` is the one the run began with. They are
    the same object until a handoff, and separate names afterwards because the two readings
    differ: who answered, and what the caller asked for. Memory extraction and compaction are
    the caller's agent's settings, so they read `start`; the turn itself reads `shape`.
    """

    shape: AgentShape
    input: str | list[ConversationItem]
    items: list[ConversationItem]
    context_wrapper: RunContextWrapper
    trace: Trace
    agent_span: Span
    original_input: list[ConversationItem]
    session: Session | None
    session_input: list[ConversationItem]
    generated: list[ConversationItem]
    hooks: _Dispatch[Any]
    run_config: RunConfig
    emit: Emit | None = None
    resumed: RunState | None = None
    """The paused state this run is continuing, if it is one: the unfinished tool-call message,
    the decisions that unblock its calls, and the results already computed for the rest of them.
    The loop reads those off the state itself rather than off a shape built from it -- a resumed
    turn needs no fact the state it came from doesn't already carry."""
    tool_calls: list[ToolCall] = field(default_factory=list)
    """Every tool call this run has executed, in the order they finished: what the tool was given
    and what it returned, before the tracing privacy policy gets a say (see `ToolCall`). A resumed
    run starts from the calls its paused `RunState` already carried."""
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

    def span(self, name: str, span_type: Any, *, input: Any = None) -> Span:
        """Open one span under this run's agent span."""
        return self.spans.open(name, span_type, input=input)

    def notify(self, event: StreamEvent) -> None:
        """Hand `event` to a streamed run's consumer; a non-streamed run has nowhere to put it."""
        if self.emit is not None:
            self.emit(event)


__all__ = ["_Run"]
