"""run_loop.py: the turn loop (`_run_turns`), and run/resume orchestration.

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
from typing import Any

from pydantic import TypeAdapter, ValidationError

from runa._governance import _Governance
from runa._items import (
    ConversationItem,
    item_text,
    latest_user_index,
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
from runa.lifecycle import Hooks, LoggingHooks, _Dispatch
from runa.run import Run
from runa.run_internal.active_run import _Run
from runa.run_internal.agent_shape import AgentShape
from runa.run_internal.guardrails import _run_guardrails
from runa.run_internal.run_config import RunConfig
from runa.run_internal.spans import _close_span, _Spans
from runa.run_internal.streaming import Emit, _stream_response
from runa.run_internal.tool_execution import _run_message_tool_calls, _TurnOutcome
from runa.run_state import RunState
from runa.session import Session
from runa.stream_events import AgentUpdatedStreamEvent, RunItemStreamEvent
from runa.tracing.config import export_trace_off_loop
from runa.tracing.spans import Span
from runa.tracing.traces import Trace
from runa.tracing.util import gen_trace_id
from runa.transcript import Transcript


async def _resolve_instructions(shape: AgentShape, run_context: RunContext) -> str | None:
    """Resolve `agent.instructions`: a string passes through, a callable is called and awaited."""
    instructions = shape.instructions
    if not callable(instructions):
        return instructions
    resolved: Any = instructions(run_context, shape.agent)
    return await resolved if inspect.isawaitable(resolved) else resolved


def _parse_output(shape: AgentShape, text: str) -> Any:
    """The model's final `text`, validated into `agent.output_type` when it declares one.

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
# `RuntimeError` is not a third group, and adding it back inverts this for the one failure the
# rest of Runa works hardest to handle: it's what async plumbing raises for lifetime faults --
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
    memory, knowledge = run.start.memory, run.start.knowledge
    if memory is None and knowledge is None:
        return
    name, spans, session = run.start.name, run.spans, run.session
    memory_matches, knowledge_matches = await asyncio.gather(
        _retrieve(
            memory,
            query,
            label="memory",
            agent_name=name,
            spans=spans,
            user_id=session.user_id if session is not None else None,
        )
        if memory is not None
        else _no_matches(),
        _retrieve(knowledge, query, label="knowledge", agent_name=name, spans=spans)
        if knowledge is not None
        else _no_matches(),
    )
    # Right before the message they were retrieved for, wherever it sits in `items`.
    items = run.transcript.items
    at = latest_user_index(items)
    assert at is not None  # `query` came from that same message
    if knowledge_matches:
        items.insert(at, _knowledge_block(knowledge_matches))
    if memory_matches:
        items.insert(at, _memory_block(memory_matches))


def _maybe_compact(
    shape: AgentShape, items: list[ConversationItem], usage_tokens: int, spans: _Spans
) -> None:
    """Run `agent.compact`'s `Compactor`, if any, and replace `items` in place if it trims them.

    Called twice: mid-`_run_turns`, on `items`, so a single run's own repeated calls (a long
    tool-calling loop) don't keep resending an ever-growing prompt; and once more when the run
    finishes (on `original_input` for a no-session run, or in `_save_to_session` on the session's
    full history) so the *next* call's history reflects the same cut too.
    """
    compactor = shape.compactor
    if compactor is None:
        return
    replacement = compactor(items, usage_tokens)
    if replacement is None or len(replacement) == len(items):
        return
    with spans.span("compact", "custom", input={"tokens": usage_tokens}) as span:
        span.output = {"dropped": len(items) - len(replacement)}
        items[:] = replacement


async def _save_to_session(
    shape: AgentShape,
    session: Session,
    new_tail: list[ConversationItem],
    context_tokens: int,
    spans: _Spans,
) -> None:
    """Append `new_tail` to `session`, rewriting its history instead if compaction trimmed it."""
    history = await session.get_items()
    full_history = [*history, *new_tail]
    _maybe_compact(shape, full_history, context_tokens, spans)
    if len(full_history) == len(history) + len(new_tail):
        await session.add_items(new_tail)
    else:
        await session.set_items(full_history)


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
    run.transcript.items.extend(results)
    run.transcript.generated.extend(results)
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
    items, hooks, run_config = run.transcript.items, run.hooks, run.run_config
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
        with run.span(str(shape.model), "llm", input=list(items)) as llm_span:
            system_instructions = await _resolve_instructions(shape, run_context)
            await hooks.on_llm_start(run_context, agent, system_instructions, items)
            request = ModelRequest(
                input=items,
                system_instructions=system_instructions,
                model_settings=shape.model_settings,
                tools=shape.tools,
                output_schema=shape.output_type,
                handoffs=list(shape.handoffs.values()),
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
        _maybe_compact(shape, items, context_tokens, run.spans)

        if not response.output:
            raise ModelBehaviorError("model returned no output items")
        message = response.output[0]
        items.append(message)
        run.transcript.generated.append(message)
        run.notify(RunItemStreamEvent(name="message_output_created", item=message))

        calls = tool_calls(message)
        if not calls:
            text = item_text(message)
            await _run_guardrails(run, Phase.OUTPUT, text)
            return _TurnOutcome(_parse_output(shape, text), [], [], context_tokens)

        for call in calls:
            run.notify(RunItemStreamEvent(name="tool_called", item=call))
        results, interruptions, switched = await _run_message_tool_calls(run, message)
        if interruptions:
            return _TurnOutcome(None, interruptions, results)
        await _record_tool_results(run, results, switched)

    raise MaxTurnsExceeded(f"max turns ({run_config.max_turns}) exceeded")


async def _close_trace(trace: Trace) -> None:
    """Stamp `trace` as finished and hand it to the exporters: once per run, however it came out.

    `_guarded` is the only caller, on each of its ways out, because "once" is the point: a trace
    exported by the step that produced it and again by the boundary that caught its failure is two
    traces to whoever reads them, and one exported by neither is a run nobody can explain.

    Awaited, because exporting is I/O (`tracing/config.py`'s `export_trace_off_loop`): a database
    write, or an HTTP POST to Langfuse. The run waits for its own, so a `run_sync` that has
    returned has a trace already written; the loop does not, so the other requests a `runa serve`
    process is serving are not held up by this one's bookkeeping.
    """
    trace.end_time = time.time()
    await export_trace_off_loop(trace)


async def _errored(
    failure: RunaError,
    *,
    trace: Trace,
    agent_span: Span,
    run_context: RunContext,
    governance: _Governance,
) -> Run:
    """Close the run out around `failure` and return it as the `Run` the caller was promised.

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
    await _close_trace(trace)
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
    the one run worth inspecting with no exported trace. So anything unrecognized becomes
    `UnexpectedError`, logged with its traceback on the way past, and the `Run` says what happened.

    It takes the things a run is known by before its first turn rather than the `_Run` the
    loop works on, because resolving the agent's shape is itself inside here: listing an MCP
    server's tools starts a subprocess, and a server that won't start used to be the one failure
    that happened before a run had a trace -- which is the whole reason `Run.trace` was once
    optional. The trace and the span exist first, the shape is resolved under them, and every way
    out of a run now ends with a trace to read.

    `timeout` (`Agent.timeout`) bounds the whole thing in wall-clock seconds. Exceeding it is
    translated into `RunTimeout`, a `RunaError` like any other, so a timed-out run closes its
    span, exports its partial trace, and comes back as `Run(status="error")` rather than leaving
    a half-finished trace behind.

    Two kinds of exception deliberately pass through instead. `asyncio.CancelledError` and the
    rest of `BaseException` are not caught at all: a caller that cancels a run (a dropped HTTP
    connection, a shutting-down worker) wants it to stop, not to be turned into an error result.
    And an `OperatorError` is a sibling of `RunaError` precisely so a run never reports one as its
    own failure (see `runa.exceptions`); the trace is still closed out around it.
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
        await _close_trace(trace)
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
        await _close_trace(trace)
        return result
    return await _errored(
        failure,
        trace=trace,
        agent_span=agent_span,
        run_context=run_context,
        governance=governance,
    )


async def _extract_memory(run: _Run, final_output: Any) -> None:
    """Store what's worth remembering from this turn in `agent.memory`, if it has one."""
    memory = run.start.memory
    query = latest_user_text([*run.transcript.original_input, *run.transcript.session_input])
    if memory is None or query is None:
        return
    with _Spans(run.trace).span("memory", "custom", input=query) as span:  # agent span is closed
        try:
            stored = await memory.remember_from_conversation(
                f"User: {query}\nAssistant: {final_output}",
                user_id=run.session.user_id if run.session is not None else None,
                model=run.start.resolve_model(run.run_config.model_provider),
            )
        except _NOT_A_BACKEND_FAULT:
            raise
        except Exception as exc:
            span.error = str(exc)
            logger.warning("memory extraction failed for agent %s", run.start.name, exc_info=True)
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
    `_guarded`'s, on the way out, so one run is one exported trace whether it completed, paused
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
            _governance=governance,
            _transcript=transcript,
            guardrail_results=governance.guardrail_results.snapshot(),
        )

    # The next call's history starts from the same cut compaction made mid-run: the session's
    # stored history, or `original_input` (what `Agent.history` is written from) without one.
    if run.session is not None:
        await _save_to_session(
            run.start,
            run.session,
            [*transcript.session_input, *transcript.generated],
            outcome.context_tokens,
            run.spans,
        )
    else:
        _maybe_compact(run.start, transcript.original_input, outcome.context_tokens, run.spans)
    await _extract_memory(run, outcome.final_output)

    await run.hooks.on_agent_end(run_context, run.current_agent, outcome.final_output)
    return Run(
        output=outcome.final_output,
        trace=run.trace,
        usage=run_context.usage,
        _governance=governance,
        _transcript=transcript,
        guardrail_results=governance.guardrail_results.snapshot(),
    )


async def _run_async(
    agent: AgentLike | AgentShape,
    input: str | list[ConversationItem] | RunState,
    *,
    context: Any = None,
    hooks: Hooks[Any] | None = None,
    run_config: RunConfig | None = None,
    session: Session | None = None,
    governance: _Governance | None = None,
    emit: Emit | None = None,
) -> Run:
    """Run `agent` on `input`, or resume the paused `RunState` it is, and return its `Run`.

    `Agent.run` is the public door; this is the room behind it, and the one other caller is a
    `.delegate` subagent's nested run (`runa.handoff.agent_as_tool`), which comes here directly
    rather than through `Agent.run` so that no internal channel shows up in a public signature.
    It passes its caller's `governance`, shared rather than started afresh, and its caller's
    `context`, which the nested run gets a `RunContext` of its own over -- usage starting at zero
    (see `runa._governance`). Every other run starts with a `_Governance` of its own.
    """
    run_config = run_config or RunConfig()
    dispatch = _Dispatch(hooks or LoggingHooks())

    if isinstance(input, RunState):
        return await _resume(input, dispatch, run_config, session, emit)

    run_context = RunContext(context=context)
    governance = governance if governance is not None else _Governance()
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
        transcript = (
            Transcript(session_input=turn_input, items=[*await session.get_items(), *turn_input])
            if session is not None
            else Transcript(original_input=list(turn_input), items=list(turn_input))
        )
        run = _Run(
            shape=shape,
            transcript=transcript,
            run_context=run_context,
            governance=governance,
            trace=trace,
            agent_span=agent_span,
            session=session,
            hooks=dispatch,
            run_config=run_config,
            emit=emit,
        )
        await dispatch.on_agent_start(run_context, shape.agent)
        if query is not None:
            await _inject_retrieved(run, query)
        await _run_guardrails(run, Phase.INPUT, input)
        return await _finish(run, await _run_turns(run))

    return await _guarded(
        turn(),
        trace=trace,
        agent_span=agent_span,
        run_context=run_context,
        governance=governance,
        timeout=run_config.timeout,
    )


async def _resume(
    state: RunState,
    dispatch: _Dispatch[Any],
    run_config: RunConfig,
    session: Session | None,
    emit: Emit | None = None,
) -> Run:
    """Continue a paused run once its interruptions are resolved.

    The agent starts again here, so it gets its own `on_agent_start`: `_finish` always fires
    `on_agent_end`, and a resumed run that skipped the start would emit an unpaired end.
    """
    agent_span = _Spans(state.trace).open(state.agent.name, "agent")

    async def turn() -> Run:
        shape = await AgentShape.of(state.agent)  # inside the boundary: see `_run_async`
        run = _Run(
            shape=shape,
            transcript=state.transcript.copy(),
            run_context=state.run_context,
            governance=state._governance,
            trace=state.trace,
            agent_span=agent_span,
            session=session,
            hooks=dispatch,
            run_config=run_config,
            emit=emit,
            resumed=state,
        )
        await dispatch.on_agent_start(state.run_context, state.agent)
        return await _finish(run, await _run_turns(run))

    return await _guarded(
        turn(),
        trace=state.trace,
        agent_span=agent_span,
        run_context=state.run_context,
        governance=state._governance,
        timeout=run_config.timeout,
    )


__all__ = ["_resume", "_run_async", "_run_turns"]
