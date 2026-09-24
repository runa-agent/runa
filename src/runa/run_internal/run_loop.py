"""run_loop.py: the turn loop (`_run_turns`), and run/resume orchestration.

The public `Runner` that calls into this lives in `runa.runner`, matching openaisdk's own
`run_internal/run_loop.py` docstring: only execution-time utilities belong here; public-facing
APIs belong at the top level.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, TypedDict

from runa._types import RunContextWrapper, TResponseInputItem
from runa.compact import Compactor, default_compactor
from runa.exceptions import (
    MaxTurnsExceeded,
    ModelBehaviorError,
    RunaError,
    RunErrorDetails,
)
from runa.lifecycle import RunHooks, logger
from runa.result import RunResult
from runa.run_config import RunConfig
from runa.run_internal.agent_runner_helpers import (
    _agent_tools,
    _model_settings,
    _normalized_handoffs,
    _resolve_instructions,
    _resolve_model,
)
from runa.run_internal.guardrails import _run_input_guardrails, _run_output_guardrails
from runa.run_internal.spans import _close_span, _export, _new_span
from runa.run_internal.streaming import Emit, _stream_response
from runa.run_internal.tool_execution import _run_message_tool_calls, _TurnOutcome
from runa.run_state import RunState
from runa.session import SessionABC
from runa.stream_events import AgentUpdatedStreamEvent, RunItemStreamEvent
from runa.tracing.spans import Span
from runa.tracing.traces import Trace
from runa.tracing.util import gen_trace_id


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
    trace: Trace,
    parent_id: str,
    **search_kwargs: Any,
) -> list[Any]:
    """Search `source` for `query`, degrading to no matches (and a logged warning) if it raises.

    Wrapped in a `"retrieval"` span so a trace shows whether memory/knowledge were consulted, what
    came back, and any failure -- not just the `llm`/`agent` spans around it.
    """
    span = _new_span(trace, parent_id, label, "retrieval", input=query)
    try:
        matches = await source.search(query, **search_kwargs)
        _close_span(span, output={"count": len(matches)})
        return matches
    except Exception as exc:
        _close_span(span, error=str(exc))
        logger.warning("%s retrieval failed for agent %s", label, agent_name, exc_info=True)
        return []


def _resolve_compactor(agent: Any) -> Compactor | None:
    """`agent.compact` to the `Compactor` to run, or `None` if compaction is off.

    `True` means Runa's own `default_compactor`; anything else truthy is trusted as already
    being a `Compactor` -- `Agent(compact=...)`'s escape hatch, the same shape as
    `memory=`/`knowledge=` accepting an instance instead of `"auto"`.
    """
    compact = getattr(agent, "compact", False)
    if not compact:
        return None
    return default_compactor if compact is True else compact


def _maybe_compact(
    agent: Any, items: list[TResponseInputItem], usage_tokens: int, trace: Trace, parent_id: str
) -> None:
    """Run `agent.compact`'s `Compactor`, if any, and replace `items` in place if it trims them.

    Called twice: mid-`_run_turns`, on `items`, so a single run's own repeated calls (a long
    tool-calling loop) don't keep resending an ever-growing prompt; and once more when the run
    finishes (on `original_input` for a no-session run, or in `_save_to_session` on the session's
    full history) so the *next* call's history reflects the same cut too.
    """
    compactor = _resolve_compactor(agent)
    if compactor is None:
        return
    replacement = compactor(items, usage_tokens)
    if replacement is None or len(replacement) == len(items):
        return
    span = _new_span(trace, parent_id, "compact", "custom", input={"tokens": usage_tokens})
    dropped = len(items) - len(replacement)
    items[:] = replacement
    _close_span(span, output={"dropped": dropped})


async def _save_to_session(
    agent: Any,
    session: SessionABC,
    new_tail: list[TResponseInputItem],
    context_tokens: int,
    trace: Trace,
    parent_id: str,
) -> None:
    """Append `new_tail` to `session`, rewriting its history instead if compaction trimmed it."""
    history = await session.get_items()
    full_history = [*history, *new_tail]
    _maybe_compact(agent, full_history, context_tokens, trace, parent_id)
    if len(full_history) == len(history) + len(new_tail):
        await session.add_items(new_tail)
    else:
        await session.set_items(full_history)


def _ignore(_event: Any) -> None:
    """The `emit` of a non-streamed run: events go nowhere."""


async def _run_turns(
    current_agent: Any,
    items: list[TResponseInputItem],
    context_wrapper: RunContextWrapper,
    hooks: RunHooks[Any],
    run_config: RunConfig,
    trace: Trace,
    agent_span_id: str,
    *,
    max_turns: int,
    generated: list[TResponseInputItem],
    pending_resume: tuple[
        TResponseInputItem, dict[str, bool], dict[str, str], list[TResponseInputItem]
    ]
    | None = None,
    emit: Emit | None = None,
) -> _TurnOutcome:
    """Call the model and run its tool calls until it answers, pauses, or runs out of turns.

    With `emit` (a streamed run), the model is streamed and every step is emitted as a
    `StreamEvent`. Every generated item is appended to `generated`, owned by the caller so a
    run that errors can still report what it produced.
    """
    notify = emit or _ignore

    if pending_resume is not None:
        last_message, approvals, rejection_messages, ready_results = pending_resume
        results, interruptions, switched = await _run_message_tool_calls(
            last_message,
            current_agent,
            context_wrapper,
            hooks,
            trace,
            agent_span_id,
            approvals,
            rejection_messages,
            ready_results,
        )
        if interruptions:
            return _TurnOutcome(None, generated, interruptions, results, current_agent)
        items.extend(results)
        generated.extend(results)
        for result in results:
            notify(RunItemStreamEvent(name="tool_output", item=result))
        if switched is not None:
            current_agent = switched
            notify(AgentUpdatedStreamEvent(new_agent=current_agent))

    for _turn in range(max_turns):
        model = _resolve_model(current_agent, run_config.model_provider)
        llm_span = _new_span(
            trace, agent_span_id, str(current_agent.model), "llm", input=list(items)
        )
        system_instructions = await _resolve_instructions(current_agent, context_wrapper)
        await hooks.on_llm_start(context_wrapper, current_agent, system_instructions, items)
        request = (
            system_instructions,
            items,
            _model_settings(current_agent),
            await _agent_tools(current_agent),
            getattr(current_agent, "output_type", None),
            list(_normalized_handoffs(getattr(current_agent, "handoffs", [])).values()),
        )
        if emit is None:
            response = await model.get_response(*request)
        else:
            response = await _stream_response(model, request, emit)
        context_wrapper.usage.add(response.usage)
        _close_span(llm_span, output={"usage": response.usage.__dict__})
        await hooks.on_llm_end(context_wrapper, current_agent, response)
        context_tokens = response.usage.input_tokens + response.usage.output_tokens
        _maybe_compact(current_agent, items, context_tokens, trace, agent_span_id)

        if not response.output:
            raise ModelBehaviorError("model returned no output items")
        message = response.output[0]
        items.append(message)
        generated.append(message)
        notify(RunItemStreamEvent(name="message_output_created", item=message))

        if not message.get("tool_calls"):
            text = message.get("content") or ""
            await _run_output_guardrails(current_agent, context_wrapper, text, trace, agent_span_id)
            return _TurnOutcome(text, generated, [], [], current_agent, context_tokens)

        for call in message["tool_calls"]:
            notify(RunItemStreamEvent(name="tool_called", item=call))
        results, interruptions, switched = await _run_message_tool_calls(
            message, current_agent, context_wrapper, hooks, trace, agent_span_id, None
        )
        if interruptions:
            return _TurnOutcome(None, generated, interruptions, results, current_agent)

        items.extend(results)
        generated.extend(results)
        for result in results:
            notify(RunItemStreamEvent(name="tool_output", item=result))
        if switched is not None:
            current_agent = switched
            notify(AgentUpdatedStreamEvent(new_agent=current_agent))
            await hooks.on_agent_start(context_wrapper, current_agent)

    raise MaxTurnsExceeded(f"max turns ({max_turns}) exceeded")


def _default_hooks() -> RunHooks[Any]:
    from runa.lifecycle import LoggingRunHooks

    return LoggingRunHooks()


@dataclass
class _Run:
    """What a fresh run and a resumed one share once the turn loop starts."""

    agent: Any
    input: str | list[TResponseInputItem]
    items: list[TResponseInputItem]
    context_wrapper: RunContextWrapper
    trace: Trace
    span: Span
    original_input: list[TResponseInputItem]
    session: SessionABC | None
    session_input: list[TResponseInputItem]
    generated: list[TResponseInputItem]


class _GuardrailResults(TypedDict):
    input_guardrail_results: list[Any]
    output_guardrail_results: list[Any]
    tool_input_guardrail_results: list[Any]
    tool_output_guardrail_results: list[Any]


def _guardrail_results(context_wrapper: RunContextWrapper) -> _GuardrailResults:
    """The run's four guardrail audit lists, as keyword arguments for a result or state."""
    return _GuardrailResults(
        input_guardrail_results=list(context_wrapper.input_guardrail_results),
        output_guardrail_results=list(context_wrapper.output_guardrail_results),
        tool_input_guardrail_results=list(context_wrapper.tool_input_guardrail_results),
        tool_output_guardrail_results=list(context_wrapper.tool_output_guardrail_results),
    )


async def _guarded(run: _Run, turns: Awaitable[_TurnOutcome]) -> _TurnOutcome:
    """Await `turns`; on a `RunaError`, close the trace and attach what the run had so far."""
    try:
        return await turns
    except RunaError as exc:
        _close_span(run.span, error=str(exc))
        run.trace.end_time = time.time()
        _export(run.trace)
        exc.run_data = RunErrorDetails(
            input=run.input,
            new_items=list(run.generated),
            raw_responses=[],
            last_agent=run.agent,
            context_wrapper=run.context_wrapper,
            trace=run.trace,
            **_guardrail_results(run.context_wrapper),
        )
        raise


async def _extract_memory(run: _Run, final_output: Any, run_config: RunConfig) -> None:
    """Store what's worth remembering from this turn in `agent.memory`, if it has one."""
    memory = getattr(run.agent, "memory", None)
    query = _latest_user_text([*run.original_input, *run.session_input])
    if memory is None or query is None:
        return
    span = _new_span(run.trace, None, "memory", "custom", input=query)
    try:
        stored = await memory.remember_from_conversation(
            f"User: {query}\nAssistant: {final_output}",
            user_id=getattr(run.session, "user_id", None),
            model=_resolve_model(run.agent, run_config.model_provider),
        )
        _close_span(span, output={"stored": len(stored)})
    except Exception as exc:
        _close_span(span, error=str(exc))
        logger.warning("memory extraction failed for agent %s", run.agent.name, exc_info=True)


async def _finish(
    run: _Run, outcome: _TurnOutcome, hooks: RunHooks[Any], run_config: RunConfig
) -> RunResult:
    """Turn the loop's outcome into a `RunResult`: a paused `RunState`, or a completed turn.

    A session-backed run persists nothing while paused: the whole turn is saved once it
    completes, whether that's straight away or after one or more resumes.
    """
    _close_span(run.span, output=outcome.final_output)
    run.trace.end_time = time.time()
    context_wrapper = run.context_wrapper

    if outcome.interruptions:
        state = RunState(
            agent=outcome.current_agent,
            original_input=run.original_input,
            generated_items=list(run.items),
            ready_results=outcome.ready_results,
            pending=outcome.interruptions,
            context_wrapper=context_wrapper,
            trace=run.trace,
            new_items=list(run.generated),
            session_input=run.session_input,
            **_guardrail_results(context_wrapper),
        )
        _export(run.trace)
        return RunResult(
            final_output=None,
            context_wrapper=context_wrapper,
            trace=run.trace,
            _original_input=run.original_input,
            _generated_items=list(run.generated),
            interruptions=outcome.interruptions,
            _state=state,
            **_guardrail_results(context_wrapper),
        )

    # The next call's history starts from the same cut compaction made mid-run: the session's
    # stored history, or `original_input` (what `to_input_list()` returns) without one.
    if run.session is not None:
        await _save_to_session(
            run.agent,
            run.session,
            [*run.session_input, *run.generated],
            outcome.context_tokens,
            run.trace,
            run.span.id,
        )
    else:
        _maybe_compact(
            run.agent, run.original_input, outcome.context_tokens, run.trace, run.span.id
        )
    await _extract_memory(run, outcome.final_output, run_config)

    await hooks.on_agent_end(context_wrapper, outcome.current_agent, outcome.final_output)
    _export(run.trace)
    return RunResult(
        final_output=outcome.final_output,
        context_wrapper=context_wrapper,
        trace=run.trace,
        _original_input=run.original_input,
        _generated_items=list(run.generated),
        **_guardrail_results(context_wrapper),
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
) -> RunResult:
    run_config = run_config or RunConfig()
    hooks = hooks or _default_hooks()

    if isinstance(input, RunState):
        return await _resume(input, hooks, run_config, session, emit)

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
    run = _Run(
        agent=agent,
        input=input if isinstance(input, str) else list(input),
        items=items,
        context_wrapper=context_wrapper,
        trace=trace,
        span=_new_span(trace, None, agent.name, "agent", input=query),
        original_input=[] if session is not None else list(turn_input),
        session=session,
        session_input=turn_input if session is not None else [],
        generated=[],
    )

    memory = getattr(agent, "memory", None)
    knowledge = getattr(agent, "knowledge", None)
    if query is not None and (memory is not None or knowledge is not None):
        memory_matches, knowledge_matches = await asyncio.gather(
            _retrieve(
                memory,
                query,
                label="memory",
                agent_name=agent.name,
                trace=trace,
                parent_id=run.span.id,
                user_id=getattr(session, "user_id", None),
            )
            if memory is not None
            else _no_matches(),
            _retrieve(
                knowledge,
                query,
                label="knowledge",
                agent_name=agent.name,
                trace=trace,
                parent_id=run.span.id,
            )
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
        await _run_input_guardrails(agent, context_wrapper, input, trace, run.span.id)
        return await _run_turns(
            agent,
            items,
            context_wrapper,
            hooks,
            run_config,
            trace,
            run.span.id,
            max_turns=run_config.max_turns,
            generated=run.generated,
            emit=emit,
        )

    await hooks.on_agent_start(context_wrapper, agent)
    outcome = await _guarded(run, turns())
    return await _finish(run, outcome, hooks, run_config)


async def _resume(
    state: RunState,
    hooks: RunHooks[Any],
    run_config: RunConfig,
    session: SessionABC | None,
    emit: Emit | None = None,
) -> RunResult:
    """Continue a paused run once its interruptions are resolved."""
    run = _Run(
        agent=state.agent,
        input=state.original_input,
        items=list(state.generated_items),
        context_wrapper=state.context_wrapper,
        trace=state.trace,
        span=_new_span(state.trace, None, state.agent.name, "agent"),
        original_input=state.original_input,
        session=session,
        session_input=state.session_input,
        generated=list(state.new_items),
    )
    turns = _run_turns(
        state.agent,
        run.items,
        state.context_wrapper,
        hooks,
        run_config,
        state.trace,
        run.span.id,
        max_turns=run_config.max_turns,
        generated=run.generated,
        emit=emit,
        pending_resume=(
            state.generated_items[-1],
            state.approvals,
            state.rejection_messages,
            state.ready_results,
        ),
    )
    outcome = await _guarded(run, turns)
    return await _finish(run, outcome, hooks, run_config)


__all__ = ["_default_hooks", "_resume", "_run_async", "_run_turns"]
