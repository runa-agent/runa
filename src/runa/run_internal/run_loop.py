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
from runa._metering import Charge, charging
from runa._models import ModelRequest, StreamDelta
from runa._types import ModelResponse, RunContext, Usage
from runa.approval import ApprovalDecision, ApprovalLedger
from runa.context_source import ContextSource
from runa.exceptions import (
    DuplicateToolCallError,
    GuardrailTripwireTriggered,
    MaxTokensExceeded,
    MaxTurnsExceeded,
    ModelBehaviorError,
    OperatorError,
    RunaError,
    RunTimeout,
    UnexpectedError,
    UserError,
)
from runa.guardrail import GuardrailResult, GuardrailResults, Phase
from runa.handoff import Delegate
from runa.lifecycle import Hooks, LoggingHooks, _Dispatch
from runa.run import Run
from runa.run_internal.agent_shape import AgentShape
from runa.run_internal.spans import _close_span, _Spans
from runa.run_internal.streaming import Emit, _Assembly, _stream_response
from runa.run_internal.tool_execution import (
    DelegatePaused,
    _run_message_tool_calls,
    _TurnOutcome,
)
from runa.run_state import RunState, _RunRecord
from runa.session import Session
from runa.stream_events import AgentChanged, StreamEvent, ToolCalled, ToolOutput
from runa.tool import FunctionTool
from runa.tracing.config import export_trace_later
from runa.tracing.manual import current_trace
from runa.tracing.spans import Span
from runa.tracing.traces import Trace
from runa.tracing.util import gen_trace_id
from runa.transcript import Transcript

if TYPE_CHECKING:
    from runa.agent import Agent


@dataclass
class _Frame:
    """One call running a run: the run's record, and everything about the call that isn't resumed.

    Every entry point builds one, then hands it to `_within`. The record is what the run is, and
    what a pause hands back; the rest is ephemeral. The starting agent's shape and this call's
    agent span, session, hooks and `emit` are the call's own. The trace, the approval ledger and
    the guardrail audit trail are the run tree's, shared by reference with every delegate's frame,
    and a paused `RunState` carries them beside the record. The frame exists before the MCP
    servers are reached, so a run that fails reaching them still reports.
    """

    shape: AgentShape
    record: _RunRecord
    trace: Trace
    agent_span: Span
    approvals: ApprovalLedger
    guardrail_results: GuardrailResults
    session: Session | None
    hooks: _Dispatch[Any]
    emit: Emit | None = None

    @property
    def run_context(self) -> RunContext[Any]:
        """The record's `RunContext`: what user code is told about the run."""
        return self.record.run_context


@dataclass(frozen=True)
class _Meter:
    """Charges one model call to `run_context`'s usage, as an `"llm"` span in `spans`.

    Every call a run pays for goes through one: the loop's own, a context source's, and an
    embedding request made anywhere below the run. Holding the run to its `max_tokens` is the
    loop's, after each of its own calls, so a charge made between two of them counts at the next.
    """

    run_context: RunContext[Any]
    spans: _Spans

    @contextmanager
    def call(self, name: str, *, input: Any = None) -> Iterator[Charge]:
        """A span for the call to the model `name`, and what to charge its `Usage` to."""
        with self.spans.span(name, "llm", input=input) as span:

            def charge(usage: Usage) -> None:
                self.run_context.usage.add(usage)
                span.output = {"usage": usage.__dict__}

            yield charge


@dataclass
class _Run:
    """One run, from the moment the loop starts it to the moment it becomes the caller's `Run`.

    Its `frame`, plus what needed the MCP servers reached: the `shape` with their tools.

    `shape` is the agent running now and `start` the one the run began with, the same until a
    handoff. The run as a whole (its context sources, its compaction of history, its turn and
    token ceilings) belongs to `start`; each turn reads `shape`.
    """

    frame: _Frame
    shape: AgentShape
    start: AgentShape = field(init=False)

    def __post_init__(self) -> None:
        self.start = self.shape

    @property
    def record(self) -> _RunRecord:
        """What this run is, as it would be resumed."""
        return self.frame.record

    @property
    def transcript(self) -> Transcript:
        """The run's conversation."""
        return self.frame.record.transcript

    @property
    def current_agent(self) -> Agent:
        """The agent running now, as user code sees it: a hook's callback, a guardrail, a pause."""
        return self.shape.agent

    async def hand_off(self, agent: Agent) -> None:
        """Make `agent` the one running now, its MCP tools listed."""
        self.shape = await agent._shape.with_mcp_tools()
        self.record.agent = agent

    @property
    def spans(self) -> _Spans:
        """Where this run's steps put their spans: children of its one agent span."""
        return _Spans(self.frame.trace, self.frame.agent_span.id)

    def span(self, name: str, span_type: Any, *, input: Any = None) -> AbstractContextManager[Span]:
        """A span under this run's agent span, for a `with` block."""
        return self.spans.span(name, span_type, input=input)

    def meter(self, under: Span | None = None) -> _Meter:
        """What a call made on this run's behalf is charged to, its span under `under` if given."""
        return _Meter(self.run_context, self.spans if under is None else self.spans.under(under))

    def charging(self, under: Span | None = None) -> AbstractContextManager[None]:
        """Charge the calls made below the run inside the block to it, as `meter` would."""
        return charging(self.meter(under))

    @property
    def run_context(self) -> RunContext[Any]:
        """What user code is told about the run."""
        return self.frame.run_context

    def notify(self, event: StreamEvent) -> None:
        """Hand `event` to a streamed run's consumer; a non-streamed run has nowhere to put it."""
        if self.frame.emit is not None:
            self.frame.emit(event)

    async def fire(self, event: str, *args: Any, agent: Agent | None = None) -> None:
        """Fire the `Hooks` method named `event` with `(run_context, agent, *args)`.

        `agent` is the one acting, the agent running now unless given.
        """
        acting = self.current_agent if agent is None else agent
        await getattr(self.frame.hooks, event)(self.run_context, acting, *args)

    def claim(self, call_id: str, name: str) -> None:
        """Mark `call_id` as executing, raising `DuplicateToolCallError` if it already ran."""
        if not self.frame.approvals.claim(call_id):
            raise DuplicateToolCallError(call_id, name)

    def release(self, call_id: str) -> None:
        """Give `call_id`'s claim back: the call didn't finish, so resuming may run it again."""
        self.frame.approvals.release(call_id)

    async def decide(
        self, tool: FunctionTool, call_id: str, args: dict[str, Any]
    ) -> ApprovalDecision:
        """Whether this call to `tool` runs, is rejected, or waits on a human."""

        async def needs_approval() -> bool:
            if isinstance(tool.needs_approval, bool):
                return tool.needs_approval
            verdict = tool.needs_approval(self.run_context, args, call_id)
            return bool(await verdict if inspect.isawaitable(verdict) else verdict)

        return await self.frame.approvals.decide(tool.name, call_id, needs_approval=needs_approval)

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
            self.frame.guardrail_results.record(result)
            if verdict.tripped:
                raise GuardrailTripwireTriggered(result)

    async def delegate(self, delegate: Delegate, input: str, call_id: str, *, under: Span) -> Run:
        """Run `delegate`'s agent on `input` as a nested run of this one, and return its `Run`.

        The nested frame is this one's with the delegate's own agent (and its ceilings), a record
        of its own over the same `context` and `user_id`, whose usage is merged back here, and an
        agent span under `under`. The trace, approval ledger, guardrail audit trail and hooks are
        shared, as is anything a frame starts carrying unless named below. This run's deadline
        encloses the delegate's.

        Not shared: the `session` (a delegate is handed fresh input, not a conversation) and
        `emit` (a stream event doesn't say which agent produced it, so the stream sees the
        delegate call and its output, not the delegate's tokens).

        A delegate that pauses for approval is stashed in this run's record under `call_id` and
        raises `DelegatePaused`, so this run pauses on its interruptions; the same call reaching
        here again, on resume, continues it instead of starting it over, and `input` is not read
        again. The returned `Run` is completed or errored.
        """
        frame = self.frame
        paused = self.record.delegates.pop(call_id, None)
        message: str | None
        if paused is None:
            context = frame.run_context
            record = _RunRecord(
                agent=delegate.agent,
                run_context=RunContext(context=context.context, user_id=context.user_id),
            )
            message = input
        else:
            record, message = paused.resumed(), None
        shape = record.agent._shape
        nested = replace(
            frame,
            shape=shape,
            record=record,
            agent_span=_Spans(frame.trace, under.id).open(shape.name, "agent", input=message),
            session=None,
            emit=None,
        )
        run = await _within(nested, message)
        frame.run_context.usage.add(run.usage)
        if run.status == "paused":
            record.run_context.usage = Usage()  # already merged into this run's
            self.record.delegates[call_id] = record
            raise DelegatePaused(record.pending)
        return run


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
        _fails_open(span, f"{name} retrieval", run.start.name),
        run.charging(under=span),
    ):
        block = await source.context(query, run.run_context)
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
    """Raise `MaxTokensExceeded` once this run has spent more than `max_tokens`."""
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
        await run.fire("on_agent_start")


async def _run_turns(run: _Run) -> _TurnOutcome:
    """Call the model and run its tool calls until it answers, pauses, or runs out of turns.

    With `run.frame.emit` (a streamed run), the model is streamed and every step is emitted as a
    `StreamEvent`. Every generated item is appended to the run's transcript as it arrives.
    """
    transcript, record, run_context = run.transcript, run.record, run.run_context
    max_turns, max_tokens = run.start.max_turns, run.start.max_tokens

    if record.pending:
        results, interruptions, switched = await _run_message_tool_calls(
            run, record.paused_message, record.ready_results
        )
        record.pending, record.ready_results = [], []
        if interruptions:
            return _TurnOutcome(None, interruptions, results)
        await _record_tool_results(run, results, switched)

    for _turn in range(max_turns):
        shape = run.shape
        model = shape.resolve_model()
        items = list(transcript.items)
        with run.meter().call(str(shape.model), input=items) as charge:
            system_instructions = await _resolve_instructions(shape, run_context)
            await run.fire("on_llm_start", system_instructions, items)
            request = ModelRequest(
                input=items,
                system_instructions=system_instructions,
                model_settings=shape.model_settings,
                tools=shape.offered,
                output_schema=shape.output_type,
            )
            if run.frame.emit is None:
                response = await model.get_response(request)
            else:
                response = await _stream_response(model, request, run.frame.emit)
            charge(response.usage)
            _check_token_budget(run_context.usage, max_tokens)
        await run.fire("on_llm_end", response)
        context_tokens = response.usage.input_tokens + response.usage.output_tokens
        dropped = transcript.compact(shape.compactor, context_tokens)
        _note_compaction(run.spans, context_tokens, dropped)

        if not response.output:
            raise ModelBehaviorError("model returned no output items")
        message = response.output[0]
        transcript.record(message)

        calls = tool_calls(message)
        if not calls:
            text = item_text(message)
            await run.guard(Phase.OUTPUT, text)
            return _TurnOutcome(_parse_output(shape, text), [], [], context_tokens)

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
        guardrail_results=frame.guardrail_results.snapshot(),
    )


async def _guarded(turn: Awaitable[Run], frame: _Frame) -> Run:
    """Await the whole run and return its `Run`, the `"error"` one included.

    The boundary where "errors never raise out of a run" is kept: it encloses everything a run
    does, application code included (a `Hooks` method, a `Session`, a `Compactor`), and catches
    `Exception`. Anything not a `RunaError` becomes `UnexpectedError`, its traceback logged.
    The starting agent's `timeout` bounds it, as `RunTimeout`, less what the record has already
    spent before a pause; the time spent here is added to the record on every way out. The frame's
    span is closed on every way out too; exporting the trace is `_traced`'s.

    `BaseException` (cancellation) and `OperatorError` pass through, the span still closed.
    """
    timeout, record = frame.shape.timeout, frame.record
    began = time.monotonic()
    try:
        if timeout is None:
            result = await turn
        else:
            try:
                async with asyncio.timeout(timeout - record.elapsed):
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
    finally:
        record.elapsed += time.monotonic() - began
    return _errored(failure, frame)


async def _within(frame: _Frame, input: str | list[ConversationItem] | None) -> Run:
    """Run `frame.shape`'s agent on `input`, or with `None`, continue its paused record, guarded.

    Every entry point ends here once its frame is built. Listing the MCP tools (which may start a
    subprocess) and reading the session are inside `_guarded` too, so their failures end the run
    like any other.
    """

    async def turn() -> Run:
        shape = await frame.shape.with_mcp_tools()
        if input is not None:
            session = frame.session
            prior = await session.get_items() if session is not None else []
            frame.record.transcript = Transcript.start(_turn_items(input), prior)
        return await _drive(_Run(frame, shape), input)

    return await _guarded(turn(), frame)


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
    beside = _Spans(run.frame.trace, run.frame.agent_span.parent_id)
    for name, source in sources:
        with (
            beside.span(name, "custom", input=query) as span,
            _fails_open(span, f"{name} after_turn", run.start.name),
        ):
            meter = _Meter(run.run_context, beside.under(span))
            with charging(meter):
                model = _SourceModel(run, meter)
                kept = await source.after_turn(query, final_output, run.run_context, model)
            span.output = {"stored": len(kept)}


@dataclass(frozen=True)
class _SourceModel:
    """The starting agent's model as a context source is handed it, each call one of the run's.

    A call is an `"llm"` span under the source's own, fires the hooks the loop's calls fire, and
    is counted in the run's usage, but isn't held to the run's `max_tokens`: a source calls it
    once the answer is in, and a finished answer isn't failed for what was spent after it.
    """

    run: _Run
    meter: _Meter

    async def get_response(self, request: ModelRequest) -> ModelResponse:
        """One call to the agent's model, charged to the run."""
        start = self.run.start
        with self.meter.call(str(start.model), input=list(request.input)) as charge:
            await self._fire("on_llm_start", request.system_instructions, list(request.input))
            response = await start.resolve_model().get_response(request)
            charge(response.usage)
        await self._fire("on_llm_end", response)
        return response

    async def stream_response(self, request: ModelRequest) -> AsyncIterator[StreamDelta]:
        """One streamed call to the agent's model, charged to the run once the stream ends."""
        start, assembly = self.run.start, _Assembly()
        with self.meter.call(str(start.model), input=list(request.input)) as charge:
            await self._fire("on_llm_start", request.system_instructions, list(request.input))
            async for delta in start.resolve_model().stream_response(request):
                assembly.add(delta)
                yield delta
            charge(assembly.usage)
        await self._fire("on_llm_end", assembly.response())

    async def _fire(self, event: str, *args: Any) -> None:
        await self.run.fire(event, *args, agent=self.run.start.agent)


async def _finish(run: _Run, outcome: _TurnOutcome) -> Run:
    """Turn the loop's outcome into the caller's `Run`: a paused `RunState`, or a completed turn.

    A session-backed run persists nothing while paused: the whole turn is saved once it
    completes, whether that's straight away or after one or more resumes.

    `usage` is the `RunContext`'s, every turn and delegate included. Closes the agent span;
    exporting the trace is `_traced`'s.
    """
    frame, record, transcript = run.frame, run.record, run.transcript
    run_context = frame.run_context
    _close_span(frame.agent_span, output=outcome.final_output)

    if outcome.interruptions:
        record.pending, record.ready_results = outcome.interruptions, outcome.ready_results
        guardrail_results = frame.guardrail_results.snapshot()
        return Run(
            output=None,
            trace=frame.trace,
            usage=run_context.usage,
            status="paused",
            interruptions=outcome.interruptions,
            _state=RunState(record, frame.trace, frame.approvals, guardrail_results),
            _transcript=transcript,
            guardrail_results=guardrail_results,
        )

    # The next call's history starts from the same cut compaction made mid-run.
    tokens = outcome.context_tokens
    dropped = transcript.compact_history(run.start.compactor, tokens)
    _note_compaction(run.spans, tokens, dropped)
    if frame.session is not None:
        if dropped is None:
            await frame.session.add_items(transcript.unsaved())
        else:
            await frame.session.set_items(transcript.history())
    await _after_turn(run, outcome.final_output)

    await run.fire("on_agent_end", outcome.final_output)
    return Run(
        output=outcome.final_output,
        trace=frame.trace,
        usage=run_context.usage,
        _transcript=transcript,
        guardrail_results=frame.guardrail_results.snapshot(),
    )


async def _drive(run: _Run, input: str | list[ConversationItem] | None) -> Run:
    """Start `run`'s agent and take it to its `Run`: a fresh run, a resumed one and a delegate's.

    A fresh run first consults memory/knowledge for its latest user message and checks `input`
    against the agent's input guardrails; a resumed one (`input=None`) did both before it paused.
    A resumed agent still gets `on_agent_start`, pairing the `on_agent_end` `_finish` fires.
    """
    with run.charging():
        await run.fire("on_agent_start")
        if input is not None:
            query = run.transcript.latest_user_text()
            if query is not None:
                await _inject_retrieved(run, query)
            await run.guard(Phase.INPUT, input)
        return await _finish(run, await _run_turns(run))


async def _run_async(
    agent: Agent,
    input: str | list[ConversationItem] | RunState,
    *,
    context: Any = None,
    hooks: Hooks[Any] | None = None,
    session: Session | None = None,
    emit: Emit | None = None,
) -> Run:
    """Run `agent` on `input`, or resume the paused `RunState` it is, and return its `Run`.

    The run behind `Agent.run`: it owns its trace, approval ledger and guardrail audit trail,
    which a delegate's nested run shares with its caller instead (`_Run.delegate`). The trace is
    named for `agent`'s class and grouped under an enclosing `tracing.trace` block, else under
    `session`.
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

    shape = agent._shape
    user_id = session.user_id if session is not None else None
    frame = _Frame(
        shape=shape,
        record=_RunRecord(agent=agent, run_context=RunContext(context=context, user_id=user_id)),
        trace=trace,
        agent_span=_Spans(trace).open(
            shape.name, "agent", input=latest_user_text(_turn_items(input))
        ),
        approvals=ApprovalLedger(),
        guardrail_results=GuardrailResults(),
        session=session,
        hooks=dispatch,
        emit=emit,
    )
    return await _traced(_within(frame, input), trace)


async def _resume(
    state: RunState,
    dispatch: _Dispatch[Any],
    session: Session | None,
    emit: Emit | None = None,
) -> Run:
    """Continue a paused run once its interruptions are resolved, on a copy of its record."""
    shape = state.agent._shape
    frame = _Frame(
        shape=shape,
        record=state._record.resumed(),
        trace=state.trace,
        agent_span=_Spans(state.trace).open(shape.name, "agent"),
        approvals=state._approvals,
        guardrail_results=state.guardrail_results.snapshot(),
        session=session,
        hooks=dispatch,
        emit=emit,
    )
    return await _traced(_within(frame, None), state.trace)


__all__ = ["_resume", "_run_async", "_run_turns"]
