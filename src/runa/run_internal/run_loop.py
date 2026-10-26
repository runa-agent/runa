"""run_loop.py: the turn loop (`_run_turns`), the `_Run` it works on, and run/resume orchestration.

The implementation behind `Agent.run`/`run_streamed`, not a seam of its own. Every path out
(`_finish` for completed and paused, `_errored` for failed) builds the `Run` the caller gets.
"""

import asyncio
import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter, ValidationError

from runa._items import (
    ConversationItem,
    item_text,
    latest_user_text,
    tool_calls,
    user_message,
)
from runa._log import logger
from runa._metering import charging, metered
from runa._models import ModelRequest, StreamDelta
from runa._types import ModelResponse, RunContext, Usage
from runa.approval import ApprovalDecision
from runa.context_source import ContextSource
from runa.exceptions import (
    DuplicateToolCallError,
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
from runa.run_internal.agent_shape import AgentShape, shape_of
from runa.run_internal.dispatch import _Dispatch
from runa.run_internal.record import Interruption, RunRecord, RunTree, keep, paused, resuming
from runa.run_internal.spans import _Spans
from runa.run_internal.streaming import Emit, _Assembly, _stream_response
from runa.run_internal.tool_execution import _run_message_tool_calls, _TurnOutcome
from runa.session import Session
from runa.stream_events import AgentChanged, StreamEvent, ToolCalled, ToolOutput
from runa.tool import FunctionTool
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
    reads `start`: its `max_turns`, `max_tokens`, `timeout`, `memory`, `knowledge` and
    `compactor` stay the starting agent's after a handoff. Every other setting is read from
    `shape`, the agent running now.

    `callers` are, for a delegate's run, the usages of the runs it is nested in, innermost first:
    each call it makes is charged to every one of them as well as to its own, as it is made. Empty
    for a root run.
    """

    record: RunRecord
    tree: RunTree
    agent_span: Span
    session: Session | None
    hooks: _Dispatch[Any]
    emit: Emit | None = None
    callers: tuple[Usage, ...] = ()
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
    def start(self) -> AgentShape:
        """The shape of the agent the run began with, whose ceilings, sources and compaction are."""
        return shape_of(self.record.start)

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

    def claim(self, call_id: str, name: str) -> None:
        """Mark `call_id` as executing, raising `DuplicateToolCallError` if it already ran."""
        if not self.tree.approvals.claim(call_id):
            raise DuplicateToolCallError(call_id, name)

    async def decide(
        self, tool: FunctionTool, call_id: str, args: dict[str, Any]
    ) -> ApprovalDecision:
        """Whether this call to `tool` runs, is rejected, or waits on a human."""
        needs_approval = partial(tool.needs_approval, self.run_context, args, call_id)
        return await self.tree.approvals.decide(tool.name, call_id, needs_approval=needs_approval)

    async def guard(
        self,
        phase: Phase,
        value: Any,
        *,
        tool: FunctionTool | None = None,
        under: Span | None = None,
    ) -> None:
        """Run `phase`'s guardrails over `value` in list order, raising on the first that trips.

        The ordering rule is stated here and nowhere else: the first tripwire stops the entries
        after it, but every entry that already ran is in the run's audit trail, so a tripped run's
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
    ) -> Run | list[Interruption]:
        """Run `delegate`'s agent on `input` as call `call_id` of this run, nested in it.

        The whole of a delegate's pause protocol is here. The call arrives claimed, like any
        tool call. A delegate that pauses for approval gives the claim back, is stashed in this
        run's record under `call_id` and comes back as its pending `Interruption`s, for this run
        to pause on; the same call reaching here again, on resume, continues it instead of
        starting it over, and `input` is not read again. Otherwise the returned `Run` is completed
        or errored.

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
        paused = self.record.delegates.pop(call_id, None)
        message: str | None
        if paused is None:
            context = self.run_context
            record = RunRecord(
                start=delegate.agent,
                agent=delegate.agent,
                run_context=RunContext(context=context.context, user_id=context.user_id),
            )
            message = input
        else:
            record, message = paused.resumed(), None
        nested = replace(
            self,
            record=record,
            agent_span=_Spans(self.tree.trace, under.id).open(
                record.agent.name, "agent", input=message
            ),
            session=None,
            emit=None,
            callers=self.payers,
        )
        run = await _within(nested, message)
        if run.status != "paused":
            return run
        self.tree.approvals.release(call_id)
        self.record.delegates[call_id] = record
        return record.pending


async def _resolve_instructions(shape: AgentShape, run_context: RunContext) -> str | None:
    """Resolve `shape.instructions`: a string passes through, a callable is called and awaited."""
    instructions = shape.instructions
    if not callable(instructions):
        return instructions
    resolved: Any = instructions(run_context, shape.agent)
    return await resolved if inspect.isawaitable(resolved) else resolved


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


def _context_sources(shape: AgentShape) -> list[tuple[str, ContextSource]]:
    """`shape`'s memory and knowledge, those it has, each with the name its spans go by."""
    sources = [("memory", shape.memory), ("knowledge", shape.knowledge)]
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
        _fails_open(span, f"{name} retrieval", run.start.name),
    ):
        block = await source.retrieve(query, run.run_context)
        span.output = block
    return block


async def _inject_retrieved(run: _Run, query: str) -> None:
    """Consult the starting agent's context sources for `query` and fold their blocks into the run.

    The sources are consulted concurrently, and each block goes right before the message it was
    retrieved for, memory's before knowledge's.
    """
    sources = _context_sources(run.start)
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


async def _record_tool_results(
    run: _Run, results: list[ConversationItem], switched: Agent | None
) -> None:
    """Fold one message's tool results into the run, following a handoff if there was one.

    The one way tool results enter a run, resumed or fresh.
    """
    run.transcript.record(*results)
    for result in results:
        run.notify(ToolOutput(result))
    if switched is not None:
        await run.hand_off(switched)
        run.notify(AgentChanged(switched))
        await run.hooks.on_agent_start(run.run_context, run.current_agent)


async def _model_turn(run: _Run) -> tuple[ConversationItem, int]:
    """Call the model once, as one of the run's turns: its message, and the context it filled.

    A turn is counted on the record, so a resumed run goes on from the turns it already took;
    one past the starting agent's `max_turns` raises `MaxTurnsExceeded` instead. With
    `run.emit` (a streamed run), the model is streamed. The message is recorded on the
    run's transcript, after any compaction the response's usage calls for.
    """
    transcript, record = run.transcript, run.record
    max_turns = run.start.max_turns
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
    model = _RunModel(run, shape, run.start.max_tokens)
    if run.emit is None:
        response = await model.get_response(request)
    else:
        response = await _stream_response(model, request, run.emit)
    context_tokens = response.usage.input_tokens + response.usage.output_tokens
    dropped = transcript.compact(run.start.compactor, context_tokens)
    _note_compaction(run.spans, context_tokens, dropped)

    if not response.output:
        raise ModelBehaviorError("model returned no output items")
    message = response.output[0]
    transcript.record(message)
    return message, context_tokens


async def _run_turns(run: _Run) -> _TurnOutcome:
    """Call the model and run its tool calls until it answers, pauses, or runs out of turns.

    A resumed run's first message is the one it paused on, already in its transcript and its
    calls already announced: only the model call is skipped, and the results computed before the
    pause are reused. With `run.emit`, every step is emitted as a `StreamEvent`.
    """
    record = run.record
    while True:
        if record.pending:
            message, ready = record.paused_message, record.ready_results
            record.pending, record.ready_results = [], []
        else:
            message, context_tokens = await _model_turn(run)
            ready = []
            calls = tool_calls(message)
            if not calls:
                text = item_text(message)
                await run.guard(Phase.OUTPUT, text)
                return _TurnOutcome(_parse_output(run.shape, text), [], [], context_tokens)
            for call in calls:
                run.notify(ToolCalled(call))

        results, interruptions, switched = await _run_message_tool_calls(run, message, ready)
        if interruptions:
            return _TurnOutcome(None, interruptions, results)
        await _record_tool_results(run, results, switched)


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
    timeout = run.start.timeout
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


async def _within(
    run: _Run,
    input: str | list[ConversationItem] | None,
    *,
    claiming: Pause | None = None,
) -> Run:
    """Run the record's agent on `input`, or with `None`, continue its paused record, guarded.

    Every entry point ends here once its run is built. Listing the MCP tools (which may start a
    subprocess), reading the session, claiming the pause a resumed run is `claiming` (see
    `_drive`) and keeping a pause in `runa.db` are inside `_guarded` too, so their failures end
    the run like any other. A pause is kept once the clock has stopped, so it carries all the time
    it spent, and never by a delegate's run: a delegate's pause is carried by its caller's.
    """

    async def turn() -> Run:
        await run.list_tools()
        if input is not None:
            session = run.session
            prior = await session.get_items() if session is not None else []
            run.record.transcript = Transcript.start(_turn_items(input), prior)
        return await _drive(run, input, claiming)

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
    sources = _context_sources(run.start)
    query = run.transcript.latest_user_text()
    if not sources or query is None:
        return
    # Beside the agent span, which is closed: at the root, or under a delegate's call.
    beside = _Spans(run.tree.trace, run.agent_span.parent_id, run.payers)
    for name, source in sources:
        with (
            beside.span(name, "custom", input=query) as span,
            _fails_open(span, f"{name} after_turn", run.start.name),
        ):
            model = _RunModel(run, run.start)
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

    async def get_response(self, request: ModelRequest) -> ModelResponse:
        """One call to the agent's model, charged to the run."""
        with metered(str(self.shape.model), input=list(request.input)) as charge:
            await self._on_llm_start(request)
            response = await self.shape.resolve_model().get_response(request)
            charge(response.usage)
            _check_token_budget(self.run.run_context.usage, self.max_tokens)
        await self.run.hooks.on_llm_end(self.run.run_context, self.shape.agent, response)
        return response

    async def stream_response(self, request: ModelRequest) -> AsyncIterator[StreamDelta]:
        """One streamed call to the agent's model, charged to the run once the stream ends."""
        assembly = _Assembly()
        with metered(str(self.shape.model), input=list(request.input)) as charge:
            await self._on_llm_start(request)
            async for delta in self.shape.resolve_model().stream_response(request):
                assembly.add(delta)
                yield delta
            charge(assembly.usage)
            _check_token_budget(self.run.run_context.usage, self.max_tokens)
        await self.run.hooks.on_llm_end(self.run.run_context, self.shape.agent, assembly.response())

    async def _on_llm_start(self, request: ModelRequest) -> None:
        await self.run.hooks.on_llm_start(
            self.run.run_context, self.shape.agent, request.system_instructions, list(request.input)
        )


async def _finish(run: _Run, outcome: _TurnOutcome) -> Run:
    """Turn the loop's outcome into the caller's `Run`: a paused `RunState`, or a completed turn.

    A session-backed run persists nothing while paused: the whole turn is saved once it
    completes, whether that's straight away or after one or more resumes.

    `usage` is the `RunContext`'s, every turn and delegate included. Closes the agent span;
    exporting the trace is `_traced`'s.
    """
    record, transcript = run.record, run.transcript
    close_span(run.agent_span, output=outcome.final_output)

    if outcome.interruptions:
        record.pending, record.ready_results = outcome.interruptions, outcome.ready_results
        return run.reported(status="paused")

    # The next call's history starts from the same cut compaction made mid-run.
    tokens = outcome.context_tokens
    dropped = transcript.compact_history(run.start.compactor, tokens)
    _note_compaction(run.spans, tokens, dropped)
    if run.session is not None:
        if dropped is None:
            await run.session.add_items(transcript.unsaved())
        else:
            await run.session.set_items(transcript.history())
    await _after_turn(run, outcome.final_output)

    await run.hooks.on_agent_end(run.run_context, run.current_agent, outcome.final_output)
    return run.reported(outcome.final_output)


async def _drive(
    run: _Run, input: str | list[ConversationItem] | None, claiming: Pause | None = None
) -> Run:
    """Start `run`'s agent and take it to its `Run`: a fresh run, a resumed one and a delegate's.

    A fresh run first consults memory/knowledge for its latest user message and checks `input`
    against the agent's input guardrails; a resumed one (`input=None`) did both before it paused.
    A resumed agent still gets `on_agent_start`, pairing the `on_agent_end` `_finish` fires.
    Its agent span is open throughout, so a call charged to nothing narrower is charged under it.

    The pause a resumed root run is `claiming` is claimed last, just before its decided calls
    run: a run that fails before then leaves the pause in `runa.db`, resumable again.
    """
    with charging(run.spans):
        await run.hooks.on_agent_start(run.run_context, run.current_agent)
        if input is not None:
            query = run.transcript.latest_user_text()
            if query is not None:
                await _inject_retrieved(run, query)
            await run.guard(Phase.INPUT, input)
        if claiming is not None and not await claiming.claim():
            raise RunAlreadyResumed(claiming.run_id)
        return await _finish(run, await _run_turns(run))


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

    The run claims the pause before running anything it decided (see `_drive`), so a pause
    resumed already ends in `RunAlreadyResumed`. A
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
    )
    return await _traced(_within(run, None, claiming=pause), pause.tree.trace)


__all__ = ["_resume", "_run_async", "_run_turns"]
