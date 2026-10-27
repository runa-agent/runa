"""run_loop.py: the turn loop (`_run_turns`), the `_Run` it works on, and run/resume orchestration.

The implementation behind `Agent.run`/`run_streamed`, not a seam of its own. Every path out
(`_finish` for completed and paused, `_errored` for failed) builds the `Run` the caller gets.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter, ValidationError

from runa._calling import call_app
from runa._items import (
    ConversationItem,
    item_text,
    latest_user_text,
    tool_calls,
    user_message,
)
from runa._log import logger
from runa._metering import Charge, charging, metered
from runa._models import ModelRequest, StreamDelta
from runa._types import ModelResponse, RunContext, Usage
from runa.context_source import ContextSource
from runa.exceptions import (
    GuardrailTripwireTriggered,
    MaxTokensExceeded,
    MaxTurnsExceeded,
    ModelBehaviorError,
    OperatorError,
    RunaError,
    RunAlreadyResumed,
    RunTimeout,
    ServiceUnavailable,
    UnexpectedError,
)
from runa.guardrail import GuardrailResult, Phase
from runa.handoff import Delegate
from runa.lifecycle import Hooks, LoggingHooks
from runa.run import Run, RunStatus
from runa.run_internal.agent_shape import AgentShape, RunPolicy, shape_of
from runa.run_internal.dispatch import _Dispatch
from runa.run_internal.record import RunRecord, RunTree, keep, paused, resuming
from runa.run_internal.spans import _Spans
from runa.run_internal.streaming import Emit, _Assembly, _emitting_text
from runa.run_internal.tool_execution import _answer_message
from runa.session import Session
from runa.stream_events import StreamEvent, ToolCalled
from runa.tool import Tool
from runa.tracing.export import export_trace_later
from runa.tracing.manual import current_trace
from runa.tracing.spans import Span, close_span
from runa.tracing.traces import Trace
from runa.tracing.util import gen_trace_id
from runa.transcript import Transcript

if TYPE_CHECKING:
    from runa.agent import Agent
    from runa.run_internal.record import Pause
    from runa.run_state import RunState


@dataclass
class _Run:
    """One run, from its entry point building it to the moment it becomes the caller's `Run`.

    Every entry point builds one, then hands it to `_within`. The record is what the run is, and
    the `tree` what it shares with every delegate's run, by reference; a pause hands both back.
    This call's agent span, session, hooks and `emit` are the call's own, and not resumed.

    `shape` is readable once `list_tools` has reached the running agent's MCP servers, so a run
    that fails reaching them still reports, from its record, tree and span. The run as a whole
    reads `policy`, the starting agent's even after a handoff, and every other setting from
    `shape`, the agent running now.

    `callers` are, for a delegate's run, the usages of the runs it is nested in, innermost first:
    each call it makes is charged to every one of them as well as to its own, as it is made. Empty
    for a root run.

    `claims` is, for a resumed root run, the pause kept in `runa.db` it continues, and claims just
    before its decided calls run. A delegate's run is never kept, so it claims nothing.
    """

    record: RunRecord
    tree: RunTree
    agent_span: Span
    session: Session | None
    hooks: _Dispatch[Any]
    emit: Emit | None = None
    callers: tuple[Usage, ...] = ()
    claims: Pause | None = None
    _listed: AgentShape | None = field(default=None, init=False, repr=False)

    @property
    def shape(self) -> AgentShape:
        """The shape of the agent running now, with its MCP tools."""
        listed = self._listed
        assert listed is not None and listed.agent is self.record.agent, "MCP tools not listed"
        return listed

    async def list_tools(self) -> None:
        """List the MCP tools of the agent running now, which may start a subprocess."""
        self._listed = await shape_of(self.record.agent).with_mcp_tools()

    async def hand_off(self, agent: Agent) -> None:
        """Make `agent` the one running now, its MCP tools listed."""
        self.record.agent = agent
        await self.list_tools()

    @property
    def delegated(self) -> bool:
        """Whether this is a delegate's run, whose pause is its caller's, not kept on its own."""
        return bool(self.callers)

    @property
    def payers(self) -> tuple[Usage, ...]:
        """What a call this run makes is charged to: its own usage, then each caller's."""
        return (self.run_context.usage, *self.callers)

    @property
    def current_agent(self) -> Agent:
        """The agent running now, as user code sees it: a hook's callback, a guardrail, a pause."""
        return self.record.agent

    @property
    def policy(self) -> RunPolicy:
        """What the run holds to throughout: its starting agent's, see `RunPolicy`."""
        return shape_of(self.record.start).policy

    @property
    def transcript(self) -> Transcript:
        """The run's conversation."""
        return self.record.transcript

    @property
    def run_context(self) -> RunContext[Any]:
        """The record's `RunContext`: what user code is told about the run."""
        return self.record.run_context

    @property
    def spans(self) -> _Spans:
        """Where this run's steps put their spans: children of its one agent span, charged to it."""
        return _Spans(self.tree.trace, self.agent_span.id, self.payers)

    def span(self, name: str, span_type: Any, *, input: Any = None) -> AbstractContextManager[Span]:
        """A span under this run's agent span, for a `with` block."""
        return self.spans.span(name, span_type, input=input)

    def reported(
        self, output: Any = None, *, status: RunStatus = "completed", error: str | None = None
    ) -> Run:
        """The caller's `Run` with `status`, over a snapshot of the record and tree as they are now.

        Never reads `shape`, which a run failing to list its MCP tools doesn't have. A paused
        `Run`'s `RunState` shares its audit trail.
        """
        tree = self.tree.snapshot()
        state = paused(self.record, tree, self.session) if status == "paused" else None
        return Run(output, status, error, self.record.snapshot(), tree, state)

    def notify(self, event: StreamEvent) -> None:
        """Hand `event` to a streamed run's consumer; a non-streamed run has nowhere to put it."""
        if self.emit is not None:
            self.emit(event)

    async def guard(
        self,
        phase: Phase,
        value: Any,
        *,
        tool: Tool | None = None,
        under: Span | None = None,
    ) -> None:
        """Run `phase`'s guardrails over `value` in list order, raising on the first that trips.

        The ordering rule is RUNA.md's (see Guardrail); what it leaves to this method is that
        every entry that already ran is in the run's audit trail, so a tripped run's
        trail is complete up to the stop.

        `phase` decides whose list to read (the running agent's, or `tool`'s), and which phase the
        audit trail and a tripwire report. Every entry is called with the one `value` the caller
        already has; what each phase makes of it is stated beside `Phase` in `runa.guardrail`. A
        tool phase passes the call's span as `under`, so its guardrails hang under the call they
        guard rather than under the run.
        """
        entries = (self.shape.guardrails if tool is None else tool.guardrails).get(phase, ())
        if not entries:
            return
        scope = self.spans if under is None else self.spans.under(under)
        for entry in entries:
            with scope.span(entry.get_name(), "guardrail") as span:
                verdict = await entry.guardrail_function(value)
                if verdict.tripped:
                    span.error = "tripwire triggered"
            result = GuardrailResult(entry, verdict, verdict.tripped, phase)
            self.tree.guardrail_results.record(result)
            if verdict.tripped:
                raise GuardrailTripwireTriggered(result)

    async def delegate(
        self, delegate: Delegate, input: str, call_id: str, *, under: Span
    ) -> Run | None:
        """Run `delegate`'s agent on `input` as call `call_id` of this run, nested in it.

        The call arrives claimed, like any tool call. A delegate that pauses for approval is
        `None`, its record suspended on this run's as the call's standing, for this run to pause
        on (see `RunRecord.suspend`), and continued by `resume_delegate`. Otherwise the returned
        `Run` is completed or errored.

        The nested run is this one with the delegate's own agent (and its ceilings), a record of
        its own over the same `context` and `user_id`, and an agent span under `under`. Each call
        it makes is charged to its own usage and to this run's (and so on up) as it is made, so
        its record's usage, like its turns, is the delegate's whole spend across its resumes. The
        tree and hooks are shared, as is anything a run starts carrying unless named below. Its
        MCP tools are its own, listed afresh. This run's deadline encloses the delegate's.

        Not shared: the `session` (a delegate is handed fresh input, not a conversation) and
        `emit` (a stream event doesn't say which agent produced it, so the stream sees the
        delegate call and its output, not the delegate's tokens).
        """
        context = self.run_context
        record = RunRecord(
            start=delegate.agent,
            agent=delegate.agent,
            run_context=RunContext(context=context.context, user_id=context.user_id),
        )
        return await self._nested(record, input, call_id, under)

    async def resume_delegate(
        self, suspended: RunRecord, call_id: str, *, under: Span
    ) -> Run | None:
        """Continue delegate call `call_id`'s run, which paused as `suspended`, where it stopped.

        Otherwise as `delegate`, and it may pause again the same way.
        """
        return await self._nested(suspended.resumed(), None, call_id, under)

    async def _nested(
        self, record: RunRecord, given: str | None, call_id: str, under: Span
    ) -> Run | None:
        """Run `record` on `given` as delegate call `call_id`'s nested run (see `delegate`)."""
        nested = replace(
            self,
            record=record,
            agent_span=_Spans(self.tree.trace, under.id).open(
                record.agent.name, "agent", input=given
            ),
            session=None,
            emit=None,
            callers=self.payers,
            claims=None,
        )
        run = await _within(nested, given)
        if run.status != "paused":
            return run
        self.record.suspend(call_id, record)
        return None


async def _resolve_instructions(shape: AgentShape, run_context: RunContext) -> str | None:
    """Resolve `shape.instructions`: a string passes through, a callable is run by `call_app`."""
    instructions = shape.instructions
    if not callable(instructions):
        return instructions
    return await call_app(instructions, run_context, shape.agent)


def _parse_output(shape: AgentShape, text: str) -> Any:
    """The model's final `text`, validated into `shape.output_type` when it declares one.

    Strictly, through the schema the model was asked for, and deliberately not through
    `parsed_json_object`'s tolerance: a declared `output_type` is a contract the provider enforces
    on its side, not prose to mine a JSON object out of.
    """
    output_type = shape.output_type
    if output_type is None or output_type is str:
        return text
    try:
        return TypeAdapter(output_type).validate_json(text)
    except ValidationError as exc:
        raise ModelBehaviorError(
            f"final output doesn't match {getattr(output_type, '__name__', output_type)}: {exc}"
        ) from exc


def _context_sources(policy: RunPolicy) -> list[tuple[str, ContextSource]]:
    """`policy`'s memory and knowledge, those it has, each with the name its spans go by."""
    sources = [("memory", policy.memory), ("knowledge", policy.knowledge)]
    return [(name, source) for name, source in sources if source is not None]


@contextmanager
def _fails_open(span: Span, step: str, agent_name: str) -> Iterator[None]:
    """Let the block's `ServiceUnavailable` go, recorded on `span` and logged; raise anything else.

    A run is still worth finishing without a source whose service didn't answer. Anything else
    ends it, because the empty block it would leave reads like a corpus with nothing relevant.
    """
    try:
        yield
    except ServiceUnavailable as exc:
        span.error = str(exc)
        logger.warning("%s failed for agent %s", step, agent_name, exc_info=True)


async def _consult(
    run: _Run, name: str, source: ContextSource, query: str
) -> ConversationItem | None:
    """`source`'s block for `query`, or `None`, in a `"retrieval"` span showing what came back."""
    block = None
    with (
        run.span(name, "retrieval", input=query) as span,
        _fails_open(span, f"{name} retrieval", run.record.start.name),
    ):
        block = await source.retrieve(query, run.run_context)
        span.output = block
    return block


async def _inject_retrieved(run: _Run, query: str) -> None:
    """Consult the starting agent's context sources for `query` and fold their blocks into the run.

    The sources are consulted concurrently, and each block goes right before the message it was
    retrieved for, memory's before knowledge's.
    """
    sources = _context_sources(run.policy)
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
    """Raise `MaxTokensExceeded` once this run has spent more than `max_tokens`.

    The loop checks after each of its own calls, so a charge made between two of them (an
    embedding in a tool, say) counts at the next.
    """
    if max_tokens is None:
        return
    spent = _spent(usage)
    if spent > max_tokens:
        raise MaxTokensExceeded(f"max tokens ({max_tokens}) exceeded: {spent} used")


@dataclass
class _TurnOutcome:
    """How the turn loop came out: an answer, or a pause on calls waiting on a human.

    Only what the loop couldn't record on the run as it went; the generated items, which agent
    ended up answering and what a pause waits on are read off the run itself.
    """

    final_output: Any
    paused: bool = False
    context_tokens: int = 0


async def _model_turn(run: _Run) -> tuple[ConversationItem, int]:
    """Call the model once, as one of the run's turns: its message, and the context it filled.

    A turn is counted on the record, so a resumed run goes on from the turns it already took;
    one past the starting agent's `max_turns` raises `MaxTurnsExceeded` instead. With
    `run.emit` (a streamed run), the model is streamed. The message is recorded on the
    run's transcript, after any compaction the response's usage calls for.
    """
    transcript, record = run.transcript, run.record
    max_turns = run.policy.max_turns
    if record.turns >= max_turns:
        raise MaxTurnsExceeded(f"max turns ({max_turns}) exceeded")
    record.turns += 1
    shape = run.shape
    request = ModelRequest(
        input=list(transcript.items),
        system_instructions=await _resolve_instructions(shape, run.run_context),
        model_settings=shape.model_settings,
        tools=shape.offered,
        output_schema=shape.output_type,
    )
    on_delta = None if run.emit is None else _emitting_text(run.emit)
    response = await _RunModel(run, shape, run.policy.max_tokens).respond(request, on_delta)
    context_tokens = response.usage.input_tokens + response.usage.output_tokens
    dropped = transcript.compact(run.policy.compactor, context_tokens)
    _note_compaction(run.spans, context_tokens, dropped)

    if not response.output:
        raise ModelBehaviorError("model returned no output items")
    message = response.output[0]
    transcript.record(message)
    return message, context_tokens


async def _run_turns(run: _Run) -> _TurnOutcome:
    """Call the model and run its tool calls until it answers, pauses, or runs out of turns.

    With `run.emit`, every step is emitted as a `StreamEvent`.
    """
    while True:
        message, context_tokens = await _model_turn(run)
        calls = tool_calls(message)
        if not calls:
            text = item_text(message)
            await run.guard(Phase.OUTPUT, text)
            return _TurnOutcome(_parse_output(run.shape, text), context_tokens=context_tokens)
        for call in calls:
            run.notify(ToolCalled(call))
        if not await _answer_message(run, message):
            return _TurnOutcome(None, paused=True)


async def _continue_paused(run: _Run) -> _TurnOutcome | None:
    """Answer the message a resumed run paused on, so the turn loop goes on from there.

    The message is the transcript's last, its calls already announced, so there is no model
    call; each call continues from where the pause left it (see `RunRecord.standings`). The root
    run's pause is claimed first, just before its decided calls run: a run that fails before
    then leaves the pause in `runa.db`, resumable again. `None` once the message is answered;
    the paused outcome if a call still waits on a human.
    """
    pause = run.claims
    if pause is not None and not await pause.claim():
        raise RunAlreadyResumed(pause.run_id)
    return (
        None
        if await _answer_message(run, run.record.paused_message)
        else _TurnOutcome(None, paused=True)
    )


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


def _errored(failure: RunaError, run: _Run) -> Run:
    """Close the run's agent span around `failure` and return the `Run` the caller was promised."""
    close_span(run.agent_span, error=str(failure))
    return run.reported(status="error", error=str(failure))


async def _timed(turn: Awaitable[Run], run: _Run) -> Run:
    """Await `turn` within the starting agent's `timeout`, less what the record already spent.

    Overrunning it is `RunTimeout`. The time spent here is added to the record on every way out,
    so a pause kept after this returns carries it.
    """
    record = run.record
    timeout = run.policy.timeout
    began = time.monotonic()
    try:
        if timeout is None:
            return await turn
        try:
            async with asyncio.timeout(timeout - record.elapsed):
                return await turn
        except TimeoutError as exc:
            raise RunTimeout(f"run timed out after {timeout}s") from exc
    finally:
        record.elapsed += time.monotonic() - began


async def _guarded(turn: Awaitable[Run], run: _Run) -> Run:
    """Await the whole run and return its `Run`, the `"error"` one included.

    The boundary where "errors never raise out of a run" is kept: it encloses everything a run
    does, application code included (a `Hooks` method, a `Session`, a `Compactor`), and catches
    `Exception`. Anything not a `RunaError` becomes `UnexpectedError`, its traceback logged.
    The run's span is closed on every way out; exporting the trace is `_traced`'s.

    `BaseException` (cancellation) and `OperatorError` pass through, the span still closed.
    """
    try:
        result = await turn
    except OperatorError as exc:
        close_span(run.agent_span, error=str(exc))
        raise
    except RunaError as exc:
        failure = exc
    except Exception as exc:
        logger.error(
            "agent %s failed with an unhandled %s",
            run.agent_span.name,
            type(exc).__name__,
            exc_info=True,
        )
        failure = UnexpectedError(exc)
    else:
        return result
    return _errored(failure, run)


type _Given = str | list[ConversationItem] | None
"""What a run is given: a fresh turn's input, or `None` to resume the pause its record holds."""


async def _within(run: _Run, given: _Given) -> Run:
    """Run the record's agent on `given`, a fresh turn's input or `None` to resume, guarded.

    Every entry point ends here once its run is built. Listing the MCP tools (which may start a
    subprocess), reading the session, claiming a resumed run's pause and keeping a pause in
    `runa.db` are inside `_guarded` too, so their failures end the run like any other. A pause
    is kept once the clock has stopped, so it carries all the time it spent, and never by a
    delegate's run: a delegate's pause is carried by its caller's.
    """

    async def turn() -> Run:
        await run.list_tools()
        return await _drive(run, given)

    async def kept() -> Run:
        result = await _timed(turn(), run)
        if result.status == "paused" and not run.delegated:
            await keep(result.to_state())
        return result

    return await _guarded(kept(), run)


def _turn_items(input: str | list[ConversationItem]) -> list[ConversationItem]:
    """The items one new turn's `input` adds to the conversation."""
    return [user_message(input)] if isinstance(input, str) else list(input)


async def _after_turn(run: _Run, final_output: Any) -> None:
    """Hand the completed exchange to each of the starting agent's context sources, in turn."""
    sources = _context_sources(run.policy)
    query = run.transcript.latest_user_text()
    if not sources or query is None:
        return
    start = shape_of(run.record.start)
    # Beside the agent span, which is closed: at the root, or under a delegate's call.
    beside = _Spans(run.tree.trace, run.agent_span.parent_id, run.payers)
    for name, source in sources:
        with (
            beside.span(name, "custom", input=query) as span,
            _fails_open(span, f"{name} after_turn", start.name),
        ):
            model = _RunModel(run, start)
            kept = await source.after_turn(query, final_output, run.run_context, model)
            span.output = {"stored": len(kept)}


@dataclass(frozen=True)
class _RunModel:
    """An agent's model as the run calls it, each call one of the run's.

    A call is an `"llm"` span in the run's trace, fires `on_llm_start` and `on_llm_end`, and is
    counted in the run's usage. With `max_tokens`, a call that takes the run over it ends its span
    in error and raises; without, the call is never held to a budget.
    """

    run: _Run
    shape: AgentShape
    max_tokens: int | None = None

    async def respond(
        self, request: ModelRequest, on_delta: Callable[[StreamDelta], None] | None = None
    ) -> ModelResponse:
        """One call to the agent's model, charged to the run, streamed when `on_delta` is given.

        A streamed call hands `on_delta` each delta as it arrives and returns them assembled.
        """
        if on_delta is None:
            return await self.get_response(request)
        assembly = _Assembly()
        async for delta in self._streamed(request, assembly):
            on_delta(delta)
        return assembly.response()

    async def get_response(self, request: ModelRequest) -> ModelResponse:
        """One call to the agent's model, unstreamed, charged to the run."""
        with metered(str(self.shape.model), input=list(request.input)) as charge:
            await self._on_llm_start(request)
            response = await self.shape.resolve_model().get_response(request)
            self._charge(charge, response)
        await self._on_llm_end(response)
        return response

    async def stream_response(self, request: ModelRequest) -> AsyncIterator[StreamDelta]:
        """One call to the agent's model, yielding each delta as it arrives.

        The call is charged, and held to `max_tokens`, once its last delta is in.
        """
        async for delta in self._streamed(request, _Assembly()):
            yield delta

    async def _streamed(
        self, request: ModelRequest, assembly: _Assembly
    ) -> AsyncIterator[StreamDelta]:
        """The call's deltas as they arrive, each added to `assembly`, then the call charged."""
        with metered(str(self.shape.model), input=list(request.input)) as charge:
            await self._on_llm_start(request)
            async for delta in self.shape.resolve_model().stream_response(request):
                assembly.add(delta)
                yield delta
            response = assembly.response()
            self._charge(charge, response)
        await self._on_llm_end(response)

    def _charge(self, charge: Charge, response: ModelResponse) -> None:
        charge(response.usage)
        _check_token_budget(self.run.run_context.usage, self.max_tokens)

    async def _on_llm_start(self, request: ModelRequest) -> None:
        await self.run.hooks.on_llm_start(
            self.run.run_context, self.shape.agent, request.system_instructions, list(request.input)
        )

    async def _on_llm_end(self, response: ModelResponse) -> None:
        await self.run.hooks.on_llm_end(self.run.run_context, self.shape.agent, response)


async def _finish(run: _Run, outcome: _TurnOutcome) -> Run:
    """Turn the loop's outcome into the caller's `Run`: a paused `RunState`, or a completed turn.

    A session-backed run persists nothing while paused: the whole turn is saved once it
    completes, whether that's straight away or after one or more resumes.

    `usage` is the `RunContext`'s, every turn and delegate included. Closes the agent span;
    exporting the trace is `_traced`'s.
    """
    transcript = run.transcript
    close_span(run.agent_span, output=outcome.final_output)

    if outcome.paused:
        return run.reported(status="paused")

    # The next call's history starts from the same cut compaction made mid-run.
    tokens = outcome.context_tokens
    dropped = transcript.compact_history(run.policy.compactor, tokens)
    _note_compaction(run.spans, tokens, dropped)
    if run.session is not None:
        if dropped is None:
            await run.session.add_items(transcript.unsaved())
        else:
            await run.session.set_items(transcript.history())
    await _after_turn(run, outcome.final_output)

    await run.hooks.on_agent_end(run.run_context, run.current_agent, outcome.final_output)
    return run.reported(outcome.final_output)


async def _drive(run: _Run, given: _Given) -> Run:
    """Start `run`'s agent and take it to its `Run`: a fresh run, a resumed one and a delegate's.

    A fresh run starts its conversation from the session's, consults memory/knowledge for its
    latest user message and checks its input against the agent's input guardrails. A resumed
    one did all three before it paused, and settles the message it paused on instead
    (`_continue_paused`). Either way the agent gets `on_agent_start`, pairing the `on_agent_end`
    `_finish` fires. Its agent span is open throughout, so a call charged to nothing narrower is
    charged under it.
    """
    with charging(run.spans):
        if given is None:
            await run.hooks.on_agent_start(run.run_context, run.current_agent)
            paused = await _continue_paused(run)
        else:
            session = run.session
            prior = await session.get_items() if session is not None else []
            run.record.transcript = Transcript.start(_turn_items(given), prior)
            await run.hooks.on_agent_start(run.run_context, run.current_agent)
            query = run.transcript.latest_user_text()
            if query is not None:
                await _inject_retrieved(run, query)
            await run.guard(Phase.INPUT, given)
            paused = None
        return await _finish(run, paused or await _run_turns(run))


async def _run_async(
    agent: Agent,
    input: str | list[ConversationItem] | RunState,
    *,
    context: Any = None,
    user_id: str | None = None,
    hooks: Hooks[Any] | None = None,
    session: Session | None = None,
    emit: Emit | None = None,
) -> Run:
    """Run `agent` on `input`, or resume the paused `RunState` it is, and return its `Run`.

    The run behind `Agent.run`: it owns its tree, which a delegate's nested run shares with its
    caller instead (`_Run.delegate`). The trace is
    named for `agent`'s declared `name` and grouped under an enclosing `tracing.trace` block, else
    under `session`.
    """
    dispatch = _Dispatch(hooks or LoggingHooks())

    if not isinstance(input, str | list):
        return await _resume(input, dispatch, session, emit)

    session_id = session.session_id if session is not None else None
    trace = Trace(id=gen_trace_id(), name=agent.name, start_time=time.time(), session_id=session_id)
    outer = current_trace()
    group_id = outer.id if outer is not None else trace.session_id
    if group_id is not None:
        trace.metadata = {"group_id": group_id}

    run = _Run(
        record=RunRecord(
            start=agent,
            agent=agent,
            run_context=RunContext(context=context, user_id=user_id),
            session_id=session_id,
        ),
        tree=RunTree(trace=trace),
        agent_span=_Spans(trace).open(
            agent.name, "agent", input=latest_user_text(_turn_items(input))
        ),
        session=session,
        hooks=dispatch,
        emit=emit,
    )
    return await _traced(_within(run, input), trace)


async def _resume(
    state: RunState,
    dispatch: _Dispatch[Any],
    session: Session | None,
    emit: Emit | None = None,
) -> Run:
    """Continue a paused run once its interruptions are resolved, on a copy of its record.

    The run claims the pause just before running anything it decided (`_continue_paused`). A
    pause the resumed run comes to is its own, kept under the agent the run started with.
    `session` is the one the run started in (see `RunState`), never a caller's choice.
    """
    pause = resuming(state)
    run = _Run(
        record=pause.record,
        tree=pause.tree,
        agent_span=_Spans(pause.tree.trace).open(pause.record.agent.name, "agent"),
        session=session,
        hooks=dispatch,
        emit=emit,
        claims=pause,
    )
    return await _traced(_within(run, None), pause.tree.trace)


__all__ = ["_resume", "_run_async", "_run_turns"]
