"""run_loop.py: the turn loop (`_run_turns`), the `_Run` it works on, and run/resume orchestration.

`Agent.run`/`run_streamed` are the only public way in: this is the Agent's implementation, not a
seam of its own, which is why everything here is underscore-prefixed. `_finish` and `_errored`
return the same `Run` the caller gets -- completed, paused, or failed -- so there is one result
shape from the loop's last line to the caller's hands, built here on every path and nothing in
between translating one into another.
"""

import asyncio
import inspect
import time
from collections.abc import Awaitable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import TypeAdapter, ValidationError

from runa._governance import _Governance
from runa._items import (
    ConversationItem,
    item_text,
    latest_user_text,
    system_message,
    tool_calls,
    user_message,
)
from runa._log import logger
from runa._models import ModelRequest
from runa._types import RunContext, Usage
from runa.agent_like import AgentLike
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
from runa.run_internal.run_config import RunConfig
from runa.run_internal.spans import _close_span, _Spans
from runa.run_internal.streaming import Emit, _stream_response
from runa.run_internal.tool_execution import (
    DelegatePaused,
    _run_message_tool_calls,
    _TurnOutcome,
)
from runa.run_state import RunState
from runa.session import Session
from runa.stream_events import AgentUpdatedStreamEvent, RunItemStreamEvent, StreamEvent
from runa.tracing.config import export_trace_later
from runa.tracing.spans import Span
from runa.tracing.traces import Trace
from runa.tracing.util import gen_trace_id
from runa.transcript import Transcript


@dataclass
class _Run:
    """One run, from the moment the loop starts it to the moment it becomes the caller's `Run`.

    The agent, its `Transcript`, the `RunContext` user code is handed and the governance it isn't,
    the span everything hangs under, and the hooks, config and `emit` this particular call was
    made with. The turn loop and the steps it calls take this, not a re-listing of its fields, so
    something a run starts carrying doesn't ripple through a dozen signatures.

    `shape` is the agent running now, resolved; `start` is the one the run began with. They are
    the same object until a handoff, and separate names afterwards because the two readings
    differ: who answered, and what the caller asked for. Memory extraction and compaction are
    the caller's agent's settings, so they read `start`; the turn itself reads `shape`. Either
    one's settings are its `agent`'s, read as `shape.agent.<setting>`.
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
    def current_agent(self) -> AgentLike:
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

    async def delegate(self, delegate: Delegate, input: str, call_id: str, *, under: Span) -> Run:
        """Run `delegate`'s agent on `input` as a nested run of this one, and return its `Run`.

        The nested run is this one, with what is a delegate's own replaced: the agent and a fresh
        transcript, a `RunContext` over the same `context` and `user_id` (its usage starts at zero
        and is merged into this run's here), an agent span under the call's `under` span, and the
        agent's own turn, token and time ceilings. Everything else is shared, and anything a run
        starts carrying is shared too unless it is named below: the governance (one approval
        ledger, one guardrail audit trail, one stash of paused delegates), the hooks, the trace,
        the model provider. This run's own deadline encloses the delegate's, as it does any tool
        call.

        Two things are deliberately not shared. A `session`: a delegate is handed fresh input,
        not a conversation. And `emit`: a stream event doesn't say which agent produced it, so a
        consumer would read a delegate's tokens as this run's answer; the stream sees the
        delegate call itself, as a tool call and its output.

        A delegate that pauses for approval is stashed under `call_id` and raises
        `DelegatePaused`, so this run pauses on its interruptions; the same call reaching here
        again, on resume, continues it instead of starting it over (`runa.paused_delegates` owns
        that protocol), and `input` is not read again. The returned `Run` is completed or errored.
        """
        agent = delegate.agent
        run_context = RunContext(context=self.run_context.context, user_id=self.run_context.user_id)
        paused = self.governance.paused_delegates.take(call_id)
        if paused is None:
            message: str | None = input
            target, transcript = agent, Transcript.start([user_message(input)])
        else:
            message, target, transcript = None, paused.agent, paused.transcript.copy()
        agent_span = _Spans(self.trace, under.id).open(target.name, "agent", input=message)
        run_config = replace(
            self.run_config,
            max_turns=agent.max_turns,
            max_tokens=agent.max_tokens,
            timeout=agent.timeout,
        )

        async def turn() -> Run:
            nested = replace(
                self,
                shape=await AgentShape.of(target),  # inside the boundary: see `_run_async`
                transcript=transcript,
                run_context=run_context,
                agent_span=agent_span,
                session=None,
                run_config=run_config,
                emit=None,
                resumed=paused,
            )
            return await _drive(nested, message)

        run = await _guarded(
            turn(),
            trace=self.trace,
            agent_span=agent_span,
            run_context=run_context,
            governance=self.governance,
            timeout=run_config.timeout,
        )
        self.run_context.usage.add(run.usage)
        if run.status == "paused":
            state = run.to_state()
            state.run_context.usage = Usage()  # already merged into this run's
            self.governance.paused_delegates.stash(call_id, state)
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


def _memory_block(matches: list[Any]) -> ConversationItem:
    """A small, clearly labeled system message carrying retrieved `MemoryMatch`es."""
    lines = "\n".join(f"- {match.text}" for match in matches)
    return system_message(f"Relevant memories:\n{lines}")


def _knowledge_block(matches: list[Any]) -> ConversationItem:
    """A small, clearly labeled system message carrying retrieved `KnowledgeMatch`es."""
    lines = "\n".join(f"- {match.text}" for match in matches)
    return system_message(f"Relevant knowledge:\n{lines}")


async def _no_matches() -> list[Any]:
    return []


# What fail-open deliberately does *not* cover. Memory and knowledge degrade to no matches when
# the backend doesn't answer -- an embeddings endpoint timing out, a vector store refusing a
# connection -- because a run is still worth finishing without them. These shapes are not that:
# they are Runa's own code being wrong, or the app's configuration being incomplete. Retrieval is
# the one step whose failure leaves no mark on the output (an empty block reads exactly like a
# corpus with nothing relevant in it), so a bug swallowed here is a bug nobody ever reports.
#
# Two groups, and nothing else: `UserError`, which the layer below raises when the app hasn't
# configured what retrieval needs (no embeddings key), and the three exceptions Python raises when
# an object isn't shaped the way the calling code assumed -- which at this seam is a custom
# `MemoryLike`/`KnowledgeLike` whose `search` doesn't match the contract, and which would
# otherwise read as a memory that simply never has anything to say.
#
# `RuntimeError` is deliberately not a third group, because it's the one failure the rest of
# Runa works hardest to handle: it's what async plumbing raises for lifetime faults --
# httpx's "client has been closed", asyncio's "Event loop is closed", a pool reached from a loop
# that didn't open it. That is the backend not answering, in its most literal form. `runa._loop`
# exists for those faults precisely because `run_sync` opens a loop per call; a custom store
# holding its own client has none of that protection, and shouldn't end a run by hitting them.
_NOT_A_BACKEND_FAULT = (UserError, AttributeError, NameError, TypeError)


async def _retrieve(
    source: Any,
    query: str,
    *,
    label: str,
    agent_name: str,
    spans: _Spans,
    **search_kwargs: Any,
) -> list[Any]:
    """Search `source` for `query`, degrading to no matches (and a logged warning) if it raises.

    Wrapped in a `"retrieval"` span so a trace shows whether memory/knowledge were consulted, what
    came back, and any failure -- not just the `llm`/`agent` spans around it.
    """
    with spans.span(label, "retrieval", input=query) as span:
        try:
            matches = await source.search(query, **search_kwargs)
        except _NOT_A_BACKEND_FAULT:
            raise
        except Exception as exc:
            span.error = str(exc)
            logger.warning("%s retrieval failed for agent %s", label, agent_name, exc_info=True)
            return []
        span.output = {"count": len(matches)}
        return matches


async def _inject_retrieved(run: _Run, query: str) -> None:
    """Search `agent.memory`/`agent.knowledge` for `query` and fold the matches into the run.

    Both are searched concurrently, and each block goes right before the message it was retrieved
    for. Called from inside `_guarded`, so a retrieval failure that isn't the backend's fault
    (see `_NOT_A_BACKEND_FAULT`) ends the run with its span closed and its trace exported, like
    any other failure the loop can hit.
    """
    start = run.start.agent
    memory, knowledge = start.memory, start.knowledge
    if memory is None and knowledge is None:
        return
    name, spans = start.name, run.spans
    memory_matches, knowledge_matches = await asyncio.gather(
        _retrieve(
            memory,
            query,
            label="memory",
            agent_name=name,
            spans=spans,
            user_id=run.run_context.user_id,
        )
        if memory is not None
        else _no_matches(),
        _retrieve(knowledge, query, label="knowledge", agent_name=name, spans=spans)
        if knowledge is not None
        else _no_matches(),
    )
    blocks = [
        *([_memory_block(memory_matches)] if memory_matches else []),
        *([_knowledge_block(knowledge_matches)] if knowledge_matches else []),
    ]
    if blocks:
        run.transcript.add_context(*blocks)


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

    The loop reaches this twice -- after a resumed message's calls, and after each turn's own --
    and a result recorded one way but not the other is how a resumed run drifts from a fresh one.
    """
    run.transcript.record(*results)
    for result in results:
        run.notify(RunItemStreamEvent(name="tool_output", item=result))
    if switched is not None:
        run.shape = await AgentShape.of(switched)
        run.notify(AgentUpdatedStreamEvent(new_agent=switched))
        await run.hooks.on_agent_start(run.run_context, switched)


async def _run_turns(run: _Run) -> _TurnOutcome:
    """Call the model and run its tool calls until it answers, pauses, or runs out of turns.

    With `run.emit` (a streamed run), the model is streamed and every step is emitted as a
    `StreamEvent`. Every generated item is appended to the run's transcript as it arrives.
    """
    transcript, hooks, run_config = run.transcript, run.hooks, run.run_config
    run_context = run.run_context

    if run.resumed is not None:
        results, interruptions, switched = await _run_message_tool_calls(
            run, run.resumed.paused_message, run.resumed
        )
        if interruptions:
            return _TurnOutcome(None, interruptions, results)
        await _record_tool_results(run, results, switched)

    for _turn in range(run_config.max_turns):
        shape = run.shape
        agent = shape.agent
        model = shape.resolve_model(run_config.model_provider)
        items = list(transcript.items)
        with run.span(str(agent.model), "llm", input=items) as llm_span:
            system_instructions = await _resolve_instructions(agent, run_context)
            await hooks.on_llm_start(run_context, agent, system_instructions, items)
            request = ModelRequest(
                input=items,
                system_instructions=system_instructions,
                model_settings=agent.model_settings,
                tools=shape.tools,
                output_schema=agent.output_type,
                handoffs=list(shape.handoffs.values()),
                delegates=list(shape.delegates.values()),
            )
            if run.emit is None:
                response = await model.get_response(request)
            else:
                response = await _stream_response(model, request, run.emit)
            run_context.usage.add(response.usage)
            llm_span.output = {"usage": response.usage.__dict__}
            _check_token_budget(run_context.usage, run_config.max_tokens)
        await hooks.on_llm_end(run_context, agent, response)
        context_tokens = response.usage.input_tokens + response.usage.output_tokens
        dropped = transcript.compact(agent.compactor, context_tokens)
        _note_compaction(run.spans, context_tokens, dropped)

        if not response.output:
            raise ModelBehaviorError("model returned no output items")
        message = response.output[0]
        transcript.record(message)
        run.notify(RunItemStreamEvent(name="message_output_created", item=message))

        calls = tool_calls(message)
        if not calls:
            text = item_text(message)
            await _run_guardrails(run, Phase.OUTPUT, text)
            return _TurnOutcome(_parse_output(agent, text), [], [], context_tokens)

        for call in calls:
            run.notify(RunItemStreamEvent(name="tool_called", item=call))
        results, interruptions, switched = await _run_message_tool_calls(run, message)
        if interruptions:
            return _TurnOutcome(None, interruptions, results)
        await _record_tool_results(run, results, switched)

    raise MaxTurnsExceeded(f"max turns ({run_config.max_turns}) exceeded")


async def _traced(run: Awaitable[Run], trace: Trace) -> Run:
    """Await a run that owns `trace`, then stamp it finished and export it, however it came out.

    Once per trace, which is why only a run that started one comes through here and a delegate,
    whose spans are in its caller's trace, does not: a trace exported by the step that produced it
    and again by the run around it is two traces to whoever reads them, and one exported by
    neither is a run nobody can explain. `run` is a `_guarded` one, so it comes back as a `Run`
    or raises one of the two things that boundary lets through: an `OperatorError`, exported
    like any other way out, or a cancellation, which isn't held up by an export.

    Queued, not awaited: exporting is I/O (a database write, an HTTP POST to Langfuse), and
    neither this run's caller nor the other requests a `runa serve` process is serving should
    wait on one backend's bookkeeping. `Agent.run_sync` flushes the queue before it returns, so
    a synchronous caller still finds its trace written.
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


def _errored(
    failure: RunaError,
    *,
    trace: Trace,
    agent_span: Span,
    run_context: RunContext,
    governance: _Governance,
) -> Run:
    """Close the run's agent span around `failure` and return the `Run` the caller was promised.

    The third way out of the loop, beside `_finish`'s completed and paused `Run`s, and built here
    for the same reason those are built there: one result shape, produced by the loop on every
    path, rather than a payload bolted onto the exception for the layer above to rebuild one from.

    What a failed run can say is exactly what the trace, the `RunContext` and the `_Governance`
    already hold -- the spans around the failure, what it spent, which guardrails ran -- so this
    needs nothing the run hadn't put somewhere a caller can read. The items it generated before
    failing are not among them: an unfinished turn is not history, so the `Run` carries an empty
    `Transcript`.
    """
    _close_span(agent_span, error=str(failure))
    return Run(
        output=None,
        trace=trace,
        usage=run_context.usage,
        status="error",
        error=str(failure),
        guardrail_results=governance.guardrail_results.snapshot(),
    )


async def _guarded(
    turn: Awaitable[Run],
    *,
    trace: Trace,
    agent_span: Span,
    run_context: RunContext,
    governance: _Governance,
    timeout: float | None,
) -> Run:
    """Await the whole run and return its `Run`, the `"error"` one included.

    This is the boundary the "errors never raise out of a run" promise is kept at, which is why
    it encloses everything a run does -- resolving the agent's shape, loading a `session`,
    dispatching hooks, the turn loop, and `_finish`'s saving and memory extraction -- and why it
    catches `Exception` rather than Runa's own root. Half of what a run calls is application code
    reached through a sanctioned extension point (a `Hooks` method, a custom `Session`, a
    `Compactor`), and a contract that held for `MaxTurnsExceeded` but not for a locked SQLite file
    would be no contract at all: `Agent.run` would hand the caller a traceback where
    `Run(status="error")` was documented, and the failure would skip the close-out below, leaving
    the one run worth inspecting with its agent span open. So anything unrecognized becomes
    `UnexpectedError`, logged with its traceback on the way past, and the `Run` says what happened.

    It takes the things a run is known by before its first turn rather than the `_Run` the
    loop works on, because resolving the agent's shape is itself inside here: listing an MCP
    server's tools starts a subprocess, which can fail. The trace and the span exist first and the
    shape is resolved under them, so every way out of a run ends with its span closed. Exporting
    the trace is not this boundary's: a delegate is guarded the same way inside its caller's
    trace, which `_traced` exports once, around the run that started it.

    `timeout` (`Agent.timeout`) bounds the whole thing in wall-clock seconds. Exceeding it is
    translated into `RunTimeout`, a `RunaError` like any other, so a timed-out run closes its
    span and comes back as `Run(status="error")` rather than leaving a half-finished trace behind.

    Two kinds of exception deliberately pass through instead. `asyncio.CancelledError` and the
    rest of `BaseException` are not caught at all: a caller that cancels a run (a dropped HTTP
    connection, a shutting-down worker) wants it to stop, not to be turned into an error result.
    And an `OperatorError` is a sibling of `RunaError` precisely so a run never reports one as its
    own failure (see `runa.exceptions`); the span is still closed around it.
    """
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
        _close_span(agent_span, error=str(exc))
        raise
    except RunaError as exc:
        failure = exc
    except Exception as exc:
        logger.error(
            "agent %s failed with an unhandled %s",
            agent_span.name,
            type(exc).__name__,
            exc_info=True,
        )
        failure = UnexpectedError(exc)
    else:
        return result
    return _errored(
        failure,
        trace=trace,
        agent_span=agent_span,
        run_context=run_context,
        governance=governance,
    )


async def _extract_memory(run: _Run, final_output: Any) -> None:
    """Store what's worth remembering from this turn in `agent.memory`, if it has one."""
    memory = run.start.agent.memory
    query = run.transcript.latest_user_text()
    if memory is None or query is None:
        return
    # Beside the agent span, which is closed: at the root, or under a delegate's call.
    beside = _Spans(run.trace, run.agent_span.parent_id)
    with beside.span("memory", "custom", input=query) as span:
        try:
            stored = await memory.remember_from_conversation(
                f"User: {query}\nAssistant: {final_output}",
                user_id=run.run_context.user_id,
                model=run.start.resolve_model(run.run_config.model_provider),
            )
        except _NOT_A_BACKEND_FAULT:
            raise
        except Exception as exc:
            span.error = str(exc)
            logger.warning(
                "memory extraction failed for agent %s", run.start.agent.name, exc_info=True
            )
            return
        span.output = {"stored": len(stored)}


async def _finish(run: _Run, outcome: _TurnOutcome) -> Run:
    """Turn the loop's outcome into the caller's `Run`: a paused `RunState`, or a completed turn.

    A session-backed run persists nothing while paused: the whole turn is saved once it
    completes, whether that's straight away or after one or more resumes.

    `usage` is the `RunContext`'s, which is where every turn (and every delegate's) has been
    accumulating it all along; `Agent.run` records that same value to `last_usage` rather than
    recomputing it.

    Closing the agent span is this function's, ending and exporting the trace is not: that is
    `_traced`'s, on the way out, so one run is one exported trace whether it completed, paused
    or failed here -- saving to a session is as able to fail as any other step.
    """
    _close_span(run.agent_span, output=outcome.final_output)
    run_context, governance, transcript = run.run_context, run.governance, run.transcript

    if outcome.interruptions:
        state = RunState(
            agent=run.current_agent,
            transcript=transcript,
            ready_results=outcome.ready_results,
            pending=outcome.interruptions,
            run_context=run_context,
            _governance=governance,
            trace=run.trace,
            guardrail_results=governance.guardrail_results.snapshot(),
        )
        return Run(
            output=None,
            trace=run.trace,
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
    if run.session is not None:
        if dropped is None:
            await run.session.add_items(transcript.unsaved())
        else:
            await run.session.set_items(transcript.history())
    await _extract_memory(run, outcome.final_output)

    await run.hooks.on_agent_end(run_context, run.current_agent, outcome.final_output)
    return Run(
        output=outcome.final_output,
        trace=run.trace,
        usage=run_context.usage,
        _transcript=transcript,
        guardrail_results=governance.guardrail_results.snapshot(),
    )


async def _drive(run: _Run, input: str | list[ConversationItem] | None) -> Run:
    """Start `run`'s agent and take it to its `Run`: a fresh run, a resumed one and a delegate's.

    A fresh run first consults memory/knowledge for its latest user message and checks `input`
    against the agent's input guardrails; a resumed one (`input=None`) did both before it paused.
    A resumed agent still gets its own `on_agent_start`: `_finish` always fires `on_agent_end`,
    and a resumed run that skipped the start would emit an unpaired end.
    """
    await run.hooks.on_agent_start(run.run_context, run.current_agent)
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
    run_config: RunConfig | None = None,
    session: Session | None = None,
    emit: Emit | None = None,
) -> Run:
    """Run `agent` on `input`, or resume the paused `RunState` it is, and return its `Run`.

    `Agent.run` is the public door; this is the room behind it. A run that comes in here owns its
    trace and starts with a `_Governance` of its own; a `.delegate` subagent's never does, being a
    nested run of its caller's (`_Run.delegate`).
    """
    run_config = run_config or RunConfig()
    dispatch = _Dispatch(hooks or LoggingHooks())

    if isinstance(input, RunState):
        return await _resume(input, dispatch, run_config, session, emit)

    run_context = RunContext(context=context, user_id=_user_of(session))
    governance = _Governance()
    trace = Trace(
        id=gen_trace_id(),
        name=run_config.workflow_name,
        start_time=time.time(),
        session_id=session.session_id if session is not None else None,
    )
    if run_config.group_id is not None or run_config.trace_metadata is not None:
        trace.metadata = {**(run_config.trace_metadata or {}), "group_id": run_config.group_id}

    turn_input = [user_message(input)] if isinstance(input, str) else list(input)
    query = latest_user_text(turn_input)
    agent_span = _Spans(trace).open(agent.name, "agent", input=query)

    async def turn() -> Run:
        # The agent's shape is resolved here, not above, for the reason the session's prior turns
        # are read here and the hooks below are dispatched here: all three are the run's work, and
        # all three reach something that can refuse. `AgentShape.of` lists an MCP server's tools,
        # which starts its subprocess; a `Session` is the application's to implement, and a locked
        # SQLite file under `runa serve` is its most ordinary failure. Inside the boundary each
        # ends the run the way every other failure does, with a trace to read, instead of
        # escaping it.
        shape = await AgentShape.of(agent)
        prior = await session.get_items() if session is not None else []
        run = _Run(
            shape=shape,
            transcript=Transcript.start(turn_input, prior),
            run_context=run_context,
            governance=governance,
            trace=trace,
            agent_span=agent_span,
            session=session,
            hooks=dispatch,
            run_config=run_config,
            emit=emit,
        )
        return await _drive(run, input)

    guarded = _guarded(
        turn(),
        trace=trace,
        agent_span=agent_span,
        run_context=run_context,
        governance=governance,
        timeout=run_config.timeout,
    )
    return await _traced(guarded, trace)


def _user_of(session: Session | None) -> str | None:
    """Who a run on `session` is for: its `user_id`, or `None` for a session-less run."""
    return session.user_id if session is not None else None


async def _resume(
    state: RunState,
    dispatch: _Dispatch[Any],
    run_config: RunConfig,
    session: Session | None,
    emit: Emit | None = None,
) -> Run:
    """Continue a paused run once its interruptions are resolved."""
    agent_span = _Spans(state.trace).open(state.agent.name, "agent")
    # The user is the resuming session's, as a fresh run's is: a serialized state doesn't carry it.
    run_context = replace(state.run_context, user_id=_user_of(session))

    async def turn() -> Run:
        shape = await AgentShape.of(state.agent)  # inside the boundary: see `_run_async`
        run = _Run(
            shape=shape,
            transcript=state.transcript.copy(),
            run_context=run_context,
            governance=state._governance,
            trace=state.trace,
            agent_span=agent_span,
            session=session,
            hooks=dispatch,
            run_config=run_config,
            emit=emit,
            resumed=state,
        )
        return await _drive(run, None)

    guarded = _guarded(
        turn(),
        trace=state.trace,
        agent_span=agent_span,
        run_context=run_context,
        governance=state._governance,
        timeout=run_config.timeout,
    )
    return await _traced(guarded, state.trace)


__all__ = ["_resume", "_run_async", "_run_turns"]
