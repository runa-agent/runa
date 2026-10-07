"""run_loop.py: the turn loop (`_run_turns`), and run/resume orchestration.

`Agent.run`/`run_streamed` are the only public way in: this is the Agent's implementation, not a
seam of its own, which is why everything here is underscore-prefixed. `_finish` returns the same
`Run` the caller gets, so there is one result shape from the loop's last line to the caller's
hands and nothing in between translating one into another.
"""

import asyncio
import inspect
import time
from collections.abc import Awaitable
from typing import Any

from pydantic import TypeAdapter, ValidationError

from runa._types import RunContextWrapper, TResponseInputItem, Usage
from runa.exceptions import (
    MaxTokensExceeded,
    MaxTurnsExceeded,
    ModelBehaviorError,
    RunaError,
    RunErrorDetails,
    RunTimeout,
)
from runa.guardrail import Phase
from runa.lifecycle import LoggingRunHooks, RunHooks, _Dispatch, logger
from runa.run import Run
from runa.run_config import RunConfig
from runa.run_internal.active_run import _Pending, _Run
from runa.run_internal.agent_shape import AgentShape
from runa.run_internal.guardrails import _run_guardrails
from runa.run_internal.spans import _close_span, _export, _Spans
from runa.run_internal.streaming import Emit, _stream_response
from runa.run_internal.tool_execution import _run_message_tool_calls, _TurnOutcome
from runa.run_state import RunState
from runa.session import SessionABC
from runa.stream_events import AgentUpdatedStreamEvent, RunItemStreamEvent
from runa.tracing.traces import Trace
from runa.tracing.util import gen_trace_id


async def _resolve_instructions(
    shape: AgentShape, context_wrapper: RunContextWrapper
) -> str | None:
    """Resolve `agent.instructions`: a string passes through, a callable is called and awaited."""
    instructions = shape.instructions
    if not callable(instructions):
        return instructions
    resolved: Any = instructions(context_wrapper, shape.agent)
    return await resolved if inspect.isawaitable(resolved) else resolved


def _parse_output(shape: AgentShape, text: str) -> Any:
    """The model's final `text`, validated into `agent.output_type` when it declares one."""
    output_type = shape.output_type
    if output_type is None or output_type is str:
        return text
    try:
        return TypeAdapter(output_type).validate_json(text)
    except ValidationError as exc:
        raise ModelBehaviorError(
            f"final output doesn't match {getattr(output_type, '__name__', output_type)}: {exc}"
        ) from exc


def _latest_user_index(items: list[TResponseInputItem]) -> int | None:
    """Where the most recent plain-text user message in `items` is, if any."""
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        if item.get("role") == "user" and isinstance(item.get("content"), str):
            return index
    return None


def _latest_user_text(items: list[TResponseInputItem]) -> str | None:
    """The most recent plain-text user message in `items`, Memory's default search query."""
    index = _latest_user_index(items)
    return items[index]["content"] if index is not None else None


def _memory_block(matches: list[Any]) -> TResponseInputItem:
    """A small, clearly labeled system message carrying retrieved `MemoryMatch`es."""
    lines = "\n".join(f"- {match.text}" for match in matches)
    return {"role": "system", "content": f"Relevant memories:\n{lines}"}


def _knowledge_block(matches: list[Any]) -> TResponseInputItem:
    """A small, clearly labeled system message carrying retrieved `KnowledgeMatch`es."""
    lines = "\n".join(f"- {match.text}" for match in matches)
    return {"role": "system", "content": f"Relevant knowledge:\n{lines}"}


async def _no_matches() -> list[Any]:
    return []


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
    span = spans.open(label, "retrieval", input=query)
    try:
        matches = await source.search(query, **search_kwargs)
        _close_span(span, output={"count": len(matches)})
        return matches
    except Exception as exc:
        _close_span(span, error=str(exc))
        logger.warning("%s retrieval failed for agent %s", label, agent_name, exc_info=True)
        return []


def _maybe_compact(
    shape: AgentShape, items: list[TResponseInputItem], usage_tokens: int, spans: _Spans
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
    span = spans.open("compact", "custom", input={"tokens": usage_tokens})
    dropped = len(items) - len(replacement)
    items[:] = replacement
    _close_span(span, output={"dropped": dropped})


async def _save_to_session(
    shape: AgentShape,
    session: SessionABC,
    new_tail: list[TResponseInputItem],
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


async def _record_tool_results(run: _Run, results: list[TResponseInputItem], switched: Any) -> None:
    """Fold one message's tool results into the run, following a handoff if there was one.

    The loop reaches this twice -- after a resumed message's calls, and after each turn's own --
    and a result recorded one way but not the other is how a resumed run drifts from a fresh one.
    """
    run.items.extend(results)
    run.generated.extend(results)
    for result in results:
        run.notify(RunItemStreamEvent(name="tool_output", item=result))
    if switched is not None:
        run.shape = await AgentShape.of(switched)
        run.notify(AgentUpdatedStreamEvent(new_agent=switched))
        await run.hooks.on_agent_start(run.context_wrapper, switched)


async def _run_turns(run: _Run) -> _TurnOutcome:
    """Call the model and run its tool calls until it answers, pauses, or runs out of turns.

    With `run.emit` (a streamed run), the model is streamed and every step is emitted as a
    `StreamEvent`. Every generated item is appended to `run.generated`, which outlives the loop,
    so a run that errors can still report what it produced.
    """
    items, hooks, run_config = run.items, run.hooks, run.run_config
    context_wrapper = run.context_wrapper

    if run.pending is not None:
        results, interruptions, switched = await _run_message_tool_calls(
            run, run.pending.message, run.pending
        )
        if interruptions:
            return _TurnOutcome(None, interruptions, results)
        await _record_tool_results(run, results, switched)

    for _turn in range(run_config.max_turns):
        shape = run.shape
        agent = shape.agent
        model = shape.resolve_model(run_config.model_provider)
        llm_span = run.span(str(shape.model), "llm", input=list(items))
        system_instructions = await _resolve_instructions(shape, context_wrapper)
        await hooks.on_llm_start(context_wrapper, agent, system_instructions, items)
        request = (
            system_instructions,
            items,
            shape.model_settings,
            shape.tools,
            shape.output_type,
            list(shape.handoffs.values()),
        )
        if run.emit is None:
            response = await model.get_response(*request)
        else:
            response = await _stream_response(model, request, run.emit)
        context_wrapper.usage.add(response.usage)
        _check_token_budget(context_wrapper.usage, run_config.max_tokens)
        _close_span(llm_span, output={"usage": response.usage.__dict__})
        await hooks.on_llm_end(context_wrapper, agent, response)
        context_tokens = response.usage.input_tokens + response.usage.output_tokens
        _maybe_compact(shape, items, context_tokens, run.spans)

        if not response.output:
            raise ModelBehaviorError("model returned no output items")
        message = response.output[0]
        items.append(message)
        run.generated.append(message)
        run.notify(RunItemStreamEvent(name="message_output_created", item=message))

        if not message.get("tool_calls"):
            text = message.get("content") or ""
            await _run_guardrails(run, Phase.OUTPUT, text)
            return _TurnOutcome(_parse_output(shape, text), [], [], context_tokens)

        for call in message["tool_calls"]:
            run.notify(RunItemStreamEvent(name="tool_called", item=call))
        results, interruptions, switched = await _run_message_tool_calls(run, message)
        if interruptions:
            return _TurnOutcome(None, interruptions, results)
        await _record_tool_results(run, results, switched)

    raise MaxTurnsExceeded(f"max turns ({run_config.max_turns}) exceeded")


async def _guarded(run: _Run, turns: Awaitable[_TurnOutcome]) -> _TurnOutcome:
    """Await `turns`; on a `RunaError`, close the trace and attach what the run had so far.

    `run_config.timeout` (`Agent.timeout`) bounds the whole thing in wall-clock seconds. Exceeding
    it is translated into `RunTimeout`, a `RunaError` like any other, so a timed-out run closes its
    span, exports its partial trace, and comes back as `Run(status="error")` rather than leaving
    a half-finished trace behind. `asyncio.CancelledError` is deliberately not caught: a caller
    that cancels a run (a dropped HTTP connection, a shutting-down worker) wants it to stop, not
    to be turned into an error result.
    """
    timeout = run.run_config.timeout
    try:
        if timeout is None:
            return await turns
        try:
            async with asyncio.timeout(timeout):
                return await turns
        except TimeoutError as exc:
            raise RunTimeout(f"run timed out after {timeout}s") from exc
    except RunaError as exc:
        _close_span(run.agent_span, error=str(exc))
        run.trace.end_time = time.time()
        _export(run.trace)
        exc.run_data = RunErrorDetails(
            input=run.input,
            new_items=list(run.generated),
            raw_responses=[],
            last_agent=run.start.agent,
            context_wrapper=run.context_wrapper,
            trace=run.trace,
            guardrail_results=run.context_wrapper.guardrail_results.snapshot(),
        )
        raise


async def _extract_memory(run: _Run, final_output: Any) -> None:
    """Store what's worth remembering from this turn in `agent.memory`, if it has one."""
    memory = run.start.memory
    query = _latest_user_text([*run.original_input, *run.session_input])
    if memory is None or query is None:
        return
    span = _Spans(run.trace).open("memory", "custom", input=query)  # the agent span is closed
    try:
        stored = await memory.remember_from_conversation(
            f"User: {query}\nAssistant: {final_output}",
            user_id=run.session.user_id if run.session is not None else None,
            model=run.start.resolve_model(run.run_config.model_provider),
        )
        _close_span(span, output={"stored": len(stored)})
    except Exception as exc:
        _close_span(span, error=str(exc))
        logger.warning("memory extraction failed for agent %s", run.start.name, exc_info=True)


async def _finish(run: _Run, outcome: _TurnOutcome) -> Run:
    """Turn the loop's outcome into the caller's `Run`: a paused `RunState`, or a completed turn.

    A session-backed run persists nothing while paused: the whole turn is saved once it
    completes, whether that's straight away or after one or more resumes.

    `usage` is the context wrapper's, which is where every turn (and every delegate's) has been
    accumulating it all along; `Agent.run` records that same value to `last_usage` rather than
    recomputing it.
    """
    _close_span(run.agent_span, output=outcome.final_output)
    run.trace.end_time = time.time()
    context_wrapper = run.context_wrapper

    if outcome.interruptions:
        state = RunState(
            agent=run.current_agent,
            original_input=run.original_input,
            generated_items=list(run.items),
            ready_results=outcome.ready_results,
            pending=outcome.interruptions,
            context_wrapper=context_wrapper,
            trace=run.trace,
            new_items=list(run.generated),
            session_input=run.session_input,
            guardrail_results=context_wrapper.guardrail_results.snapshot(),
        )
        for interruption in outcome.interruptions:
            interruption.owner = interruption.owner or state  # a delegate's keeps its own
        _export(run.trace)
        return Run(
            output=None,
            trace=run.trace,
            usage=context_wrapper.usage,
            status="paused",
            interruptions=outcome.interruptions,
            _state=state,
            _context_wrapper=context_wrapper,
            _original_input=run.original_input,
            _generated_items=list(run.generated),
            guardrail_results=context_wrapper.guardrail_results.snapshot(),
        )

    # The next call's history starts from the same cut compaction made mid-run: the session's
    # stored history, or `original_input` (what `to_input_list()` returns) without one.
    if run.session is not None:
        await _save_to_session(
            run.start,
            run.session,
            [*run.session_input, *run.generated],
            outcome.context_tokens,
            run.spans,
        )
    else:
        _maybe_compact(run.start, run.original_input, outcome.context_tokens, run.spans)
    await _extract_memory(run, outcome.final_output)

    await run.hooks.on_agent_end(context_wrapper, run.current_agent, outcome.final_output)
    _export(run.trace)
    return Run(
        output=outcome.final_output,
        trace=run.trace,
        usage=context_wrapper.usage,
        _context_wrapper=context_wrapper,
        _original_input=run.original_input,
        _generated_items=list(run.generated),
        guardrail_results=context_wrapper.guardrail_results.snapshot(),
    )


async def _run_async(
    agent: Any,
    input: str | list[TResponseInputItem] | RunState,
    *,
    context: Any = None,
    hooks: RunHooks[Any] | None = None,
    run_config: RunConfig | None = None,
    session: SessionABC | None = None,
    _context_wrapper: RunContextWrapper[Any] | None = None,
    emit: Emit | None = None,
) -> Run:
    run_config = run_config or RunConfig()
    dispatch = _Dispatch(hooks or LoggingRunHooks())

    if isinstance(input, RunState):
        return await _resume(input, dispatch, run_config, session, emit)

    context_wrapper = (
        _context_wrapper if _context_wrapper is not None else RunContextWrapper(context=context)
    )
    trace = Trace(
        id=gen_trace_id(),
        name=run_config.workflow_name,
        start_time=time.time(),
        session_id=session.session_id if session is not None else None,
    )
    if run_config.group_id is not None or run_config.trace_metadata is not None:
        trace.metadata = {**(run_config.trace_metadata or {}), "group_id": run_config.group_id}

    turn_input = [{"role": "user", "content": input}] if isinstance(input, str) else list(input)
    history = await session.get_items() if session is not None else []
    items = [*history, *turn_input]
    query = _latest_user_text(turn_input)
    shape = await AgentShape.of(agent)
    run = _Run(
        shape=shape,
        input=input if isinstance(input, str) else list(input),
        items=items,
        context_wrapper=context_wrapper,
        trace=trace,
        agent_span=_Spans(trace).open(shape.name, "agent", input=query),
        original_input=[] if session is not None else list(turn_input),
        session=session,
        session_input=turn_input if session is not None else [],
        generated=[],
        hooks=dispatch,
        run_config=run_config,
        emit=emit,
    )

    memory = shape.memory
    knowledge = shape.knowledge
    if query is not None and (memory is not None or knowledge is not None):
        memory_matches, knowledge_matches = await asyncio.gather(
            _retrieve(
                memory,
                query,
                label="memory",
                agent_name=shape.name,
                spans=run.spans,
                user_id=session.user_id if session is not None else None,
            )
            if memory is not None
            else _no_matches(),
            _retrieve(knowledge, query, label="knowledge", agent_name=shape.name, spans=run.spans)
            if knowledge is not None
            else _no_matches(),
        )
        # Right before the message they were retrieved for, wherever it sits in `items`.
        at = _latest_user_index(items)
        assert at is not None  # `query` came from that same message
        if knowledge_matches:
            items.insert(at, _knowledge_block(knowledge_matches))
        if memory_matches:
            items.insert(at, _memory_block(memory_matches))

    async def turns() -> _TurnOutcome:
        await _run_guardrails(run, Phase.INPUT, input)
        return await _run_turns(run)

    await dispatch.on_agent_start(context_wrapper, shape.agent)
    outcome = await _guarded(run, turns())
    return await _finish(run, outcome)


async def _resume(
    state: RunState,
    dispatch: _Dispatch[Any],
    run_config: RunConfig,
    session: SessionABC | None,
    emit: Emit | None = None,
) -> Run:
    """Continue a paused run once its interruptions are resolved.

    The agent starts again here, so it gets its own `on_agent_start`: `_finish` always fires
    `on_agent_end`, and a resumed run that skipped the start would emit an unpaired end.
    """
    shape = await AgentShape.of(state.agent)
    run = _Run(
        shape=shape,
        input=state.original_input,
        items=list(state.generated_items),
        context_wrapper=state.context_wrapper,
        trace=state.trace,
        agent_span=_Spans(state.trace).open(shape.name, "agent"),
        original_input=state.original_input,
        session=session,
        session_input=state.session_input,
        generated=list(state.new_items),
        hooks=dispatch,
        run_config=run_config,
        emit=emit,
        pending=_Pending(
            message=state.generated_items[-1],
            approvals=state.approvals,
            rejection_messages=state.rejection_messages,
            ready_results=state.ready_results,
        ),
    )
    await dispatch.on_agent_start(state.context_wrapper, state.agent)
    outcome = await _guarded(run, _run_turns(run))
    return await _finish(run, outcome)


__all__ = ["_resume", "_run_async", "_run_turns"]
