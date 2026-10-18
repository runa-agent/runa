"""run_loop.py: the turn loop (`_run_turns`), the `_Run` it works on, and run/resume orchestration.

The implementation behind `Agent.run`/`run_streamed`, not a seam of its own. Every path out
(`_finish` for completed and paused, `_errored` for failed) builds the `Run` the caller gets.
"""

import asyncio
import inspect
import time
from collections.abc import Awaitable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import TypeAdapter, ValidationError

from runa._governance import _Governance
from runa._items import (
    ConversationItem,
    item_text,
    latest_user_text,
    tool_calls,
    user_message,
)
from runa._log import logger
from runa._models import ModelRequest
from runa._types import RunContext, Usage
from runa.agent_like import AgentLike
from runa.context_source import ContextSource
from runa.exceptions import (
    MaxTokensExceeded,
    MaxTurnsExceeded,
    ModelBehaviorError,
    OperatorError,
    RunaError,
    RunTimeout,
    UnexpectedError,
    UserError,
)
from runa.guardrail import Phase
from runa.handoff import Delegate
from runa.lifecycle import Hooks, LoggingHooks, _Dispatch
from runa.run import Run
from runa.run_internal.agent_shape import AgentShape
from runa.run_internal.guardrails import _run_guardrails
from runa.run_internal.spans import _close_span, _Spans
from runa.run_internal.streaming import Emit, _stream_response
from runa.run_internal.tool_execution import (
    DelegatePaused,
    _run_message_tool_calls,
    _TurnOutcome,
)
from runa.run_state import RunState
from runa.session import Session
from runa.stream_events import AgentChanged, StreamEvent, ToolCalled, ToolOutput
from runa.tracing.config import export_trace_later
from runa.tracing.manual import current_trace
from runa.tracing.spans import Span
from runa.tracing.traces import Trace
from runa.tracing.util import gen_trace_id
from runa.transcript import Transcript


@dataclass
class _Frame:
    """What a run is known by before its agent is resolved, and all a failed run can report.

    The starting agent, the trace and agent span, the `RunContext` and the governance, and the
    session, hooks and `emit` of this call. Every entry point builds one, then hands it to
    `_within`. It exists before the agent is resolved, so a run that fails resolving still reports.
    """

    agent: AgentLike
    trace: Trace
    agent_span: Span
    run_context: RunContext
    governance: _Governance
    session: Session | None
    hooks: _Dispatch[Any]
    emit: Emit | None = None


@dataclass
class _Run:
    """One run, from the moment the loop starts it to the moment it becomes the caller's `Run`.

    Its `frame`, plus what needed the agent resolved: the `shape`, the `Transcript`, and the
    paused state it continues, if any.

    `shape` is the agent running now; `start` is the one the run began with, the same object
    until a handoff. Its context sources, compaction and the turn and token ceilings read `start`;
    the turn itself reads `shape`.
    """

    frame: _Frame
    shape: AgentShape
    transcript: Transcript
    resumed: RunState | None = None
    """The paused state this run is continuing, if it is one: the unfinished tool-call message,
    the decisions that unblock its calls, and the results already computed for the rest of them."""
    start: AgentShape = field(init=False)

    def __post_init__(self) -> None:
        self.start = self.shape

    @property
    def current_agent(self) -> AgentLike:
        """The agent running now, as user code sees it: a hook's callback, a guardrail, a pause."""
        return self.shape.agent

    @property
    def spans(self) -> _Spans:
        """Where this run's steps put their spans: children of its one agent span."""
        return _Spans(self.frame.trace, self.frame.agent_span.id)

    def span(self, name: str, span_type: Any, *, input: Any = None) -> AbstractContextManager[Span]:
        """A span under this run's agent span, for a `with` block."""
        return self.spans.span(name, span_type, input=input)

    def notify(self, event: StreamEvent) -> None:
        """Hand `event` to a streamed run's consumer; a non-streamed run has nowhere to put it."""
        if self.frame.emit is not None:
            self.frame.emit(event)

    async def delegate(self, delegate: Delegate, input: str, call_id: str, *, under: Span) -> Run:
        """Run `delegate`'s agent on `input` as a nested run of this one, and return its `Run`.

        The nested frame is this one's with the delegate's own agent (and its ceilings), a
        `RunContext` over the same `context` and `user_id` whose usage is merged back here, and
        an agent span under `under`. The governance, hooks and trace are shared, as is anything
        a frame starts carrying unless named below. This run's deadline encloses the delegate's.

        Not shared: the `session` (a delegate is handed fresh input, not a conversation) and
        `emit` (a stream event doesn't say which agent produced it, so the stream sees the
        delegate call and its output, not the delegate's tokens).

        A delegate that pauses for approval is stashed under `call_id` and raises
        `DelegatePaused`, so this run pauses on its interruptions; the same call reaching here
        again, on resume, continues it instead of starting it over, and `input` is not read
        again. The returned `Run` is completed or errored.
        """
        frame = self.frame
        paused = frame.governance.paused_delegates.pop(call_id, None)
        start: str | RunState
        if paused is None:
            target, message, start = delegate.agent, input, input
        else:
            target, message, start = paused.agent, None, paused
        nested = replace(
            frame,
            agent=target,
            agent_span=_Spans(frame.trace, under.id).open(target.name, "agent", input=message),
            run_context=RunContext(
                context=frame.run_context.context, user_id=frame.run_context.user_id
            ),
            session=None,
            emit=None,
        )
        run = await _within(nested, start)
        frame.run_context.usage.add(run.usage)
        if run.status == "paused":
            state = run.to_state()
            state.run_context.usage = Usage()  # already merged into this run's
            frame.governance.paused_delegates[call_id] = state
            raise DelegatePaused(state.pending)
        return run


async def _resolve_instructions(agent: AgentLike, run_context: RunContext) -> str | None:
    """Resolve `agent.instructions`: a string passes through, a callable is called and awaited."""
    instructions = agent.instructions
    if not callable(instructions):
        return instructions
    resolved: Any = instructions(run_context, agent)
    return await resolved if inspect.isawaitable(resolved) else resolved


def _parse_output(agent: AgentLike, text: str) -> Any:
    """The model's final `text`, validated into `agent.output_type` when it declares one.

    Strictly, through the schema the model was asked for, and deliberately not through
    `parsed_json_object`'s tolerance: a declared `output_type` is a contract the provider enforces
    on its side, not prose to mine a JSON object out of.
    """
    output_type = agent.output_type
    if output_type is None or output_type is str:
        return text
    try:
        return TypeAdapter(output_type).validate_json(text)
    except ValidationError as exc:
        raise ModelBehaviorError(
            f"final output doesn't match {getattr(output_type, '__name__', output_type)}: {exc}"
        ) from exc


def _context_sources(agent: AgentLike) -> list[tuple[str, ContextSource]]:
    """`agent`'s memory and knowledge, those it has, each with the name its spans go by."""
    sources = [("memory", agent.memory), ("knowledge", agent.knowledge)]
    return [(name, source) for name, source in sources if source is not None]


# A context source degrades to nothing when its backend doesn't answer, because a run is still
# worth finishing without it. These are not that: missing configuration (`UserError`, no
# embeddings key) and a custom source not shaped as `ContextSource` says. An empty block reads
# like a corpus with nothing relevant, so these raise instead. A `RuntimeError` ("Event loop is
# closed", a closed client) is a backend not answering, and degrades.
_NOT_A_BACKEND_FAULT = (UserError, AttributeError, NameError, TypeError)


@contextmanager
def _fails_open(span: Span, step: str, agent_name: str) -> Iterator[None]:
    """Let the block's backend fault go, recorded on `span` and logged; raise anything else."""
    try:
        yield
    except _NOT_A_BACKEND_FAULT:
        raise
    except Exception as exc:
        span.error = str(exc)
        logger.warning("%s failed for agent %s", step, agent_name, exc_info=True)


async def _consult(
    run: _Run, name: str, source: ContextSource, query: str
) -> ConversationItem | None:
    """`source`'s block for `query`, or `None`, in a `"retrieval"` span showing what came back."""
    block = None
    with (
        run.span(name, "retrieval", input=query) as span,
        _fails_open(span, f"{name} retrieval", run.start.agent.name),
    ):
        block = await source.context(query, run.frame.run_context)
        span.output = block
    return block


async def _inject_retrieved(run: _Run, query: str) -> None:
    """Consult the starting agent's context sources for `query` and fold their blocks into the run.

    The sources are consulted concurrently, and each block goes right before the message it was
    retrieved for, memory's before knowledge's.
    """
    sources = _context_sources(run.start.agent)
    blocks = await asyncio.gather(*(_consult(run, name, source, query) for name, source in sources))
    found = [block for block in blocks if block is not None]
    if found:
        run.transcript.add_context(*found)


def _note_compaction(spans: _Spans, tokens: int, dropped: int | None) -> None:
    """Record a compaction that changed the conversation (`dropped` is not `None`) as a span."""
    if dropped is not None:
        with spans.span("compact", "custom", input={"tokens": tokens}) as span:
            span.output = {"dropped": dropped}


def _spent(usage: Usage) -> int:
    """Total tokens `usage` represents, falling back to input+output if `total_tokens` is unset.

    Not every provider reports a total; the two components are always there.
    """
    return usage.total_tokens or (usage.input_tokens + usage.output_tokens)


def _check_token_budget(usage: Usage, max_tokens: int | None) -> None:
    """Raise `MaxTokensExceeded` once this run has spent more than `max_tokens`."""
    if max_tokens is None:
        return
    spent = _spent(usage)
    if spent > max_tokens:
        raise MaxTokensExceeded(f"max tokens ({max_tokens}) exceeded: {spent} used")


async def _record_tool_results(
    run: _Run, results: list[ConversationItem], switched: AgentLike | None
) -> None:
    """Fold one message's tool results into the run, following a handoff if there was one.

    The one way tool results enter a run, resumed or fresh.
    """
    run.transcript.record(*results)
    for result in results:
        run.notify(ToolOutput(result))
    if switched is not None:
        run.shape = await AgentShape.of(switched)
        run.notify(AgentChanged(switched))
        await run.frame.hooks.on_agent_start(run.frame.run_context, switched)


async def _run_turns(run: _Run) -> _TurnOutcome:
    """Call the model and run its tool calls until it answers, pauses, or runs out of turns.

    With `run.frame.emit` (a streamed run), the model is streamed and every step is emitted as a
    `StreamEvent`. Every generated item is appended to the run's transcript as it arrives.
    """
    transcript, hooks = run.transcript, run.frame.hooks
    max_turns, max_tokens = run.start.agent.max_turns, run.start.agent.max_tokens
    run_context = run.frame.run_context

    if run.resumed is not None:
        results, interruptions, switched = await _run_message_tool_calls(
            run, run.resumed.paused_message, run.resumed
        )
        if interruptions:
            return _TurnOutcome(None, interruptions, results)
        await _record_tool_results(run, results, switched)

    for _turn in range(max_turns):
        shape = run.shape
        agent = shape.agent
        model = shape.resolve_model()
        items = list(transcript.items)
        with run.span(str(agent.model), "llm", input=items) as llm_span:
            system_instructions = await _resolve_instructions(agent, run_context)
            await hooks.on_llm_start(run_context, agent, system_instructions, items)
            request = ModelRequest(
                input=items,
                system_instructions=system_instructions,
                model_settings=agent.model_settings,
                tools=shape.offered,
                output_schema=agent.output_type,
            )
            if run.frame.emit is None:
                response = await model.get_response(request)
            else:
                response = await _stream_response(model, request, run.frame.emit)
            run_context.usage.add(response.usage)
            llm_span.output = {"usage": response.usage.__dict__}
            _check_token_budget(run_context.usage, max_tokens)
        await hooks.on_llm_end(run_context, agent, response)
        context_tokens = response.usage.input_tokens + response.usage.output_tokens
        dropped = transcript.compact(agent.compactor, context_tokens)
        _note_compaction(run.spans, context_tokens, dropped)

        if not response.output:
            raise ModelBehaviorError("model returned no output items")
        message = response.output[0]
        transcript.record(message)

        calls = tool_calls(message)
        if not calls:
            text = item_text(message)
            await _run_guardrails(run, Phase.OUTPUT, text)
            return _TurnOutcome(_parse_output(agent, text), [], [], context_tokens)

        for call in calls:
            run.notify(ToolCalled(call))
        results, interruptions, switched = await _run_message_tool_calls(run, message)
        if interruptions:
            return _TurnOutcome(None, interruptions, results)
        await _record_tool_results(run, results, switched)

    raise MaxTurnsExceeded(f"max turns ({max_turns}) exceeded")


async def _traced(run: Awaitable[Run], trace: Trace) -> Run:
    """Await a run that owns `trace`, then stamp it finished and export it, however it came out.

    Only a run that started its trace comes through here, never a delegate, so each trace is
    exported exactly once. An `OperatorError` is exported too; a cancellation isn't held up by
    one. The export is queued, not awaited; `Agent.run_sync` flushes the queue before returning.
    """
    try:
        result = await run
    except OperatorError:
        _export(trace)
        raise
    _export(trace)
    return result


def _export(trace: Trace) -> None:
    trace.end_time = time.time()
    export_trace_later(trace)


def _errored(failure: RunaError, frame: _Frame) -> Run:
    """Close the run's agent span around `failure` and return the `Run` the caller was promised.

    Reports what the frame holds: the spans, the usage, the guardrails that ran. An unfinished
    turn is not history, so the `Run` carries an empty `Transcript`.
    """
    _close_span(frame.agent_span, error=str(failure))
    return Run(
        output=None,
        trace=frame.trace,
        usage=frame.run_context.usage,
        status="error",
        error=str(failure),
        guardrail_results=frame.governance.guardrail_results.snapshot(),
    )


async def _guarded(turn: Awaitable[Run], frame: _Frame) -> Run:
    """Await the whole run and return its `Run`, the `"error"` one included.

    The boundary where "errors never raise out of a run" is kept: it encloses everything a run
    does, application code included (a `Hooks` method, a `Session`, a `Compactor`), and catches
    `Exception`. Anything not a `RunaError` becomes `UnexpectedError`, its traceback logged.
    The frame agent's `timeout` bounds it, as `RunTimeout`. The frame's span is closed on every
    way out; exporting the trace is `_traced`'s.

    `BaseException` (cancellation) and `OperatorError` pass through, the span still closed.
    """
    timeout = frame.agent.timeout
    try:
        if timeout is None:
            result = await turn
        else:
            try:
                async with asyncio.timeout(timeout):
                    result = await turn
            except TimeoutError as exc:
                raise RunTimeout(f"run timed out after {timeout}s") from exc
    except OperatorError as exc:
        _close_span(frame.agent_span, error=str(exc))
        raise
    except RunaError as exc:
        failure = exc
    except Exception as exc:
        logger.error(
            "agent %s failed with an unhandled %s",
            frame.agent_span.name,
            type(exc).__name__,
            exc_info=True,
        )
        failure = UnexpectedError(exc)
    else:
        return result
    return _errored(failure, frame)


async def _within(frame: _Frame, input: str | list[ConversationItem] | RunState) -> Run:
    """Run `frame.agent` on `input`, or continue the paused `RunState` it is, inside the boundary.

    Every entry point ends here once its frame is built. Resolving the agent's shape (which may
    start an MCP subprocess) and reading the session are inside `_guarded` too, so their
    failures end the run like any other.
    """

    async def turn() -> Run:
        shape = await AgentShape.of(frame.agent)
        if isinstance(input, RunState):
            return await _drive(_Run(frame, shape, input.transcript.copy(), input), None)
        session = frame.session
        prior = await session.get_items() if session is not None else []
        return await _drive(_Run(frame, shape, Transcript.start(_turn_items(input), prior)), input)

    return await _guarded(turn(), frame)


def _turn_items(input: str | list[ConversationItem]) -> list[ConversationItem]:
    """The items one new turn's `input` adds to the conversation."""
    return [user_message(input)] if isinstance(input, str) else list(input)


async def _after_turn(run: _Run, final_output: Any) -> None:
    """Hand the completed exchange to each of the starting agent's context sources, in turn."""
    sources = _context_sources(run.start.agent)
    query = run.transcript.latest_user_text()
    if not sources or query is None:
        return
    model = run.start.resolve_model()
    # Beside the agent span, which is closed: at the root, or under a delegate's call.
    beside = _Spans(run.frame.trace, run.frame.agent_span.parent_id)
    for name, source in sources:
        with (
            beside.span(name, "custom", input=query) as span,
            _fails_open(span, f"{name} after_turn", run.start.agent.name),
        ):
            kept = await source.after_turn(query, final_output, run.frame.run_context, model)
            span.output = {"stored": len(kept)}


async def _finish(run: _Run, outcome: _TurnOutcome) -> Run:
    """Turn the loop's outcome into the caller's `Run`: a paused `RunState`, or a completed turn.

    A session-backed run persists nothing while paused: the whole turn is saved once it
    completes, whether that's straight away or after one or more resumes.

    `usage` is the `RunContext`'s, every turn and delegate included. Closes the agent span;
    exporting the trace is `_traced`'s.
    """
    frame, transcript = run.frame, run.transcript
    run_context, governance = frame.run_context, frame.governance
    _close_span(frame.agent_span, output=outcome.final_output)

    if outcome.interruptions:
        state = RunState(
            agent=run.current_agent,
            transcript=transcript,
            ready_results=outcome.ready_results,
            pending=outcome.interruptions,
            run_context=run_context,
            _governance=governance,
            trace=frame.trace,
            guardrail_results=governance.guardrail_results.snapshot(),
        )
        return Run(
            output=None,
            trace=frame.trace,
            usage=run_context.usage,
            status="paused",
            interruptions=outcome.interruptions,
            _state=state,
            _transcript=transcript,
            guardrail_results=governance.guardrail_results.snapshot(),
        )

    # The next call's history starts from the same cut compaction made mid-run.
    tokens = outcome.context_tokens
    dropped = transcript.compact_history(run.start.agent.compactor, tokens)
    _note_compaction(run.spans, tokens, dropped)
    if frame.session is not None:
        if dropped is None:
            await frame.session.add_items(transcript.unsaved())
        else:
            await frame.session.set_items(transcript.history())
    await _after_turn(run, outcome.final_output)

    await frame.hooks.on_agent_end(run_context, run.current_agent, outcome.final_output)
    return Run(
        output=outcome.final_output,
        trace=frame.trace,
        usage=run_context.usage,
        _transcript=transcript,
        guardrail_results=governance.guardrail_results.snapshot(),
    )


async def _drive(run: _Run, input: str | list[ConversationItem] | None) -> Run:
    """Start `run`'s agent and take it to its `Run`: a fresh run, a resumed one and a delegate's.

    A fresh run first consults memory/knowledge for its latest user message and checks `input`
    against the agent's input guardrails; a resumed one (`input=None`) did both before it paused.
    A resumed agent still gets `on_agent_start`, pairing the `on_agent_end` `_finish` fires.
    """
    await run.frame.hooks.on_agent_start(run.frame.run_context, run.current_agent)
    if run.resumed is None:
        query = run.transcript.latest_user_text()
        if query is not None:
            await _inject_retrieved(run, query)
        await _run_guardrails(run, Phase.INPUT, input)
    return await _finish(run, await _run_turns(run))


async def _run_async(
    agent: AgentLike,
    input: str | list[ConversationItem] | RunState,
    *,
    context: Any = None,
    hooks: Hooks[Any] | None = None,
    session: Session | None = None,
    emit: Emit | None = None,
) -> Run:
    """Run `agent` on `input`, or resume the paused `RunState` it is, and return its `Run`.

    The run behind `Agent.run`: it owns its trace and its governance, which a delegate's nested
    run shares with its caller instead (`_Run.delegate`). The trace is named for `agent`'s class
    and grouped under an enclosing `tracing.trace` block, else under `session`.
    """
    dispatch = _Dispatch(hooks or LoggingHooks())

    if isinstance(input, RunState):
        return await _resume(input, dispatch, session, emit)

    trace = Trace(
        id=gen_trace_id(),
        name=type(agent).__name__,
        start_time=time.time(),
        session_id=session.session_id if session is not None else None,
    )
    outer = current_trace()
    group_id = outer.id if outer is not None else trace.session_id
    if group_id is not None:
        trace.metadata = {"group_id": group_id}

    frame = _Frame(
        agent=agent,
        trace=trace,
        agent_span=_Spans(trace).open(
            agent.name, "agent", input=latest_user_text(_turn_items(input))
        ),
        run_context=RunContext(context=context, user_id=_user_of(session)),
        governance=_Governance(),
        session=session,
        hooks=dispatch,
        emit=emit,
    )
    return await _traced(_within(frame, input), trace)


def _user_of(session: Session | None) -> str | None:
    """Who a run on `session` is for: its `user_id`, or `None` for a session-less run."""
    return session.user_id if session is not None else None


async def _resume(
    state: RunState,
    dispatch: _Dispatch[Any],
    session: Session | None,
    emit: Emit | None = None,
) -> Run:
    """Continue a paused run once its interruptions are resolved."""
    frame = _Frame(
        agent=state.agent,
        trace=state.trace,
        agent_span=_Spans(state.trace).open(state.agent.name, "agent"),
        # The user is the resuming session's, as a fresh run's is: a serialized state doesn't
        # carry it.
        run_context=replace(state.run_context, user_id=_user_of(session)),
        governance=state._governance,
        session=session,
        hooks=dispatch,
        emit=emit,
    )
    return await _traced(_within(frame, state), state.trace)


__all__ = ["_resume", "_run_async", "_run_turns"]
