"""run_loop.py: the turn loop (`_run_turns`), and run/resume orchestration.

The public `Runner` that calls into this lives in `runa.runner`, matching openaisdk's own
`run_internal/run_loop.py` docstring: only execution-time utilities belong here; public-facing
APIs belong at the top level.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from runa._types import RunContextWrapper, TResponseInputItem
from runa.compact import Compactor, default_compactor
from runa.exceptions import MaxTurnsExceeded, ModelBehaviorError, RunaError
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
from runa.run_internal.tool_execution import _run_message_tool_calls, _TurnOutcome
from runa.run_state import RunState
from runa.session import SessionABC
from runa.tracing.traces import Trace
from runa.tracing.util import gen_trace_id


def _latest_user_text(items: list[TResponseInputItem]) -> str | None:
    """The most recent plain-text user message in `items`, Memory's default search query."""
    for item in reversed(items):
        if item.get("role") == "user" and isinstance(item.get("content"), str):
            return item["content"]
    return None


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
    tool-calling loop) don't keep resending an ever-growing prompt; and once more in `_run_async`
    (on `original_input` for a no-session run, or on the full `[*history, *new_tail]` view for a
    session-backed one) so the *next* call's history reflects the same cut too.
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
    pending_resume: tuple[TResponseInputItem, dict[str, bool], dict[str, str]] | None = None,
) -> _TurnOutcome:
    generated: list[TResponseInputItem] = []

    if pending_resume is not None:
        last_message, approvals, rejection_messages = pending_resume
        results, interruptions, switched = await _run_message_tool_calls(
            last_message,
            current_agent,
            context_wrapper,
            hooks,
            trace,
            agent_span_id,
            approvals,
            rejection_messages,
        )
        if interruptions:
            return _TurnOutcome(None, [], interruptions, results, current_agent)
        items.extend(results)
        generated.extend(results)
        if switched is not None:
            current_agent = switched

    for _turn in range(max_turns):
        model = _resolve_model(current_agent, run_config.model_provider)
        llm_span = _new_span(
            trace, agent_span_id, str(current_agent.model), "llm", input=list(items)
        )
        system_instructions = await _resolve_instructions(current_agent, context_wrapper)
        await hooks.on_llm_start(context_wrapper, current_agent, system_instructions, items)
        response = await model.get_response(
            system_instructions,
            items,
            _model_settings(current_agent),
            await _agent_tools(current_agent),
            getattr(current_agent, "output_type", None),
            list(_normalized_handoffs(getattr(current_agent, "handoffs", [])).values()),
        )
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

        if not message.get("tool_calls"):
            text = message.get("content") or ""
            await _run_output_guardrails(current_agent, context_wrapper, text, trace, agent_span_id)
            return _TurnOutcome(text, generated, [], [], current_agent, context_tokens)

        results, interruptions, switched = await _run_message_tool_calls(
            message, current_agent, context_wrapper, hooks, trace, agent_span_id, None
        )
        if interruptions:
            return _TurnOutcome(None, generated, interruptions, results, current_agent)

        items.extend(results)
        generated.extend(results)
        if switched is not None:
            current_agent = switched
            await hooks.on_agent_start(context_wrapper, current_agent)

    raise MaxTurnsExceeded(f"max turns ({max_turns}) exceeded")


def _default_hooks() -> RunHooks[Any]:
    from runa.lifecycle import LoggingRunHooks

    return LoggingRunHooks()


async def _run_async(
    agent: Any,
    input: str | list[TResponseInputItem] | RunState,
    *,
    context: Any = None,
    hooks: RunHooks[Any] | None = None,
    run_config: RunConfig | None = None,
    session: SessionABC | None = None,
    _context_wrapper: RunContextWrapper[Any] | None = None,
) -> RunResult:
    run_config = run_config or RunConfig()
    hooks = hooks or _default_hooks()

    if isinstance(input, RunState):
        return await _resume(input, hooks, run_config)

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

    history: list[TResponseInputItem] = []
    if session is not None:
        history = await session.get_items()
        new_message = {"role": "user", "content": input} if isinstance(input, str) else None
        items = [*history, new_message] if new_message else [*history, *input]
        original_input: list[TResponseInputItem] = []
    else:
        items = [{"role": "user", "content": input}] if isinstance(input, str) else list(input)
        original_input = list(items) if not isinstance(input, str) else []

    memory = getattr(agent, "memory", None)
    knowledge = getattr(agent, "knowledge", None)
    user_id = getattr(session, "user_id", None) if session is not None else None
    memory_query = _latest_user_text(items)

    agent_span = _new_span(trace, None, agent.name, "agent", input=memory_query)

    if memory_query is not None and (memory is not None or knowledge is not None):
        memory_matches, knowledge_matches = await asyncio.gather(
            _retrieve(
                memory,
                memory_query,
                label="memory",
                agent_name=agent.name,
                trace=trace,
                parent_id=agent_span.id,
                user_id=user_id,
            )
            if memory is not None
            else _no_matches(),
            _retrieve(
                knowledge,
                memory_query,
                label="knowledge",
                agent_name=agent.name,
                trace=trace,
                parent_id=agent_span.id,
            )
            if knowledge is not None
            else _no_matches(),
        )
        if memory_matches:
            items.insert(len(items) - 1, _memory_block(memory_matches))
        if knowledge_matches:
            items.insert(len(items) - 1, _knowledge_block(knowledge_matches))

    await hooks.on_agent_start(context_wrapper, agent)
    try:
        await _run_input_guardrails(agent, context_wrapper, input, trace, agent_span.id)
        outcome = await _run_turns(
            agent,
            items,
            context_wrapper,
            hooks,
            run_config,
            trace,
            agent_span.id,
            max_turns=run_config.max_turns,
        )
    except RunaError as exc:
        _close_span(agent_span, error=str(exc))
        trace.end_time = time.time()
        _export(trace)
        from runa.exceptions import RunErrorDetails

        exc.run_data = RunErrorDetails(
            input=input if isinstance(input, str) else list(input),
            new_items=[],
            raw_responses=[],
            last_agent=agent,
            context_wrapper=context_wrapper,
            input_guardrail_results=list(context_wrapper.input_guardrail_results),
            output_guardrail_results=list(context_wrapper.output_guardrail_results),
            tool_input_guardrail_results=list(context_wrapper.tool_input_guardrail_results),
            tool_output_guardrail_results=list(context_wrapper.tool_output_guardrail_results),
            trace=trace,
        )
        raise

    _close_span(agent_span, error=None, output=outcome.final_output)
    trace.end_time = time.time()

    if outcome.interruptions:
        state = RunState(
            agent=outcome.current_agent,
            original_input=original_input,
            generated_items=[*items[: len(items) - len(outcome.ready_results)]]
            if outcome.ready_results
            else list(items),
            ready_results=outcome.ready_results,
            pending=outcome.interruptions,
            context_wrapper=context_wrapper,
            trace=trace,
            input_guardrail_results=list(context_wrapper.input_guardrail_results),
            output_guardrail_results=list(context_wrapper.output_guardrail_results),
            tool_input_guardrail_results=list(context_wrapper.tool_input_guardrail_results),
            tool_output_guardrail_results=list(context_wrapper.tool_output_guardrail_results),
        )
        _export(trace)
        return RunResult(
            final_output=None,
            context_wrapper=context_wrapper,
            trace=trace,
            _original_input=original_input,
            _generated_items=outcome.generated,
            interruptions=outcome.interruptions,
            _state=state,
            input_guardrail_results=list(context_wrapper.input_guardrail_results),
            output_guardrail_results=list(context_wrapper.output_guardrail_results),
            tool_input_guardrail_results=list(context_wrapper.tool_input_guardrail_results),
            tool_output_guardrail_results=list(context_wrapper.tool_output_guardrail_results),
        )

    if session is not None:
        to_persist = [{"role": "user", "content": input}] if isinstance(input, str) else list(input)
        new_tail = [*to_persist, *outcome.generated]
        full_history = [*history, *new_tail]
        _maybe_compact(agent, full_history, outcome.context_tokens, trace, agent_span.id)
        if len(full_history) == len(history) + len(new_tail):
            await session.add_items(new_tail)
        else:
            await session.set_items(full_history)

    if memory is not None and memory_query is not None:
        extraction_span = _new_span(trace, None, "memory", "custom", input=memory_query)
        try:
            conversation = f"User: {memory_query}\nAssistant: {outcome.final_output}"
            resolved_model = _resolve_model(agent, run_config.model_provider)
            stored = await memory.remember_from_conversation(
                conversation, user_id=user_id, model=resolved_model
            )
            _close_span(extraction_span, output={"stored": len(stored)})
        except Exception as exc:
            _close_span(extraction_span, error=str(exc))
            logger.warning("memory extraction failed for agent %s", agent.name, exc_info=True)

    # `original_input` is what `agent.history` becomes via `to_input_list()` for a no-session run
    # (the session case is compacted above, against `session`'s own stored history) -- compact it
    # too, so the *next* call starts from the same cut `items` already made mid-run, not the full
    # pre-compaction history.
    if session is None:
        _maybe_compact(agent, original_input, outcome.context_tokens, trace, agent_span.id)

    await hooks.on_agent_end(context_wrapper, outcome.current_agent, outcome.final_output)
    _export(trace)
    return RunResult(
        final_output=outcome.final_output,
        context_wrapper=context_wrapper,
        trace=trace,
        _original_input=original_input,
        _generated_items=outcome.generated,
        input_guardrail_results=list(context_wrapper.input_guardrail_results),
        output_guardrail_results=list(context_wrapper.output_guardrail_results),
        tool_input_guardrail_results=list(context_wrapper.tool_input_guardrail_results),
        tool_output_guardrail_results=list(context_wrapper.tool_output_guardrail_results),
    )


async def _resume(state: RunState, hooks: RunHooks[Any], run_config: RunConfig) -> RunResult:
    items = list(state.generated_items)
    agent_span = _new_span(state.trace, None, state.agent.name, "agent")

    pending_message = state.generated_items[-1]
    try:
        outcome = await _run_turns(
            state.agent,
            items,
            state.context_wrapper,
            hooks,
            run_config,
            state.trace,
            agent_span.id,
            max_turns=run_config.max_turns,
            pending_resume=(pending_message, state.approvals, state.rejection_messages),
        )
    except RunaError as exc:
        _close_span(agent_span, error=str(exc))
        state.trace.end_time = time.time()
        _export(state.trace)
        from runa.exceptions import RunErrorDetails

        exc.run_data = RunErrorDetails(
            input=state.original_input,
            new_items=[],
            raw_responses=[],
            last_agent=state.agent,
            context_wrapper=state.context_wrapper,
            input_guardrail_results=list(state.context_wrapper.input_guardrail_results),
            output_guardrail_results=list(state.context_wrapper.output_guardrail_results),
            tool_input_guardrail_results=list(state.context_wrapper.tool_input_guardrail_results),
            tool_output_guardrail_results=list(state.context_wrapper.tool_output_guardrail_results),
            trace=state.trace,
        )
        raise

    _close_span(agent_span, output=outcome.final_output)
    state.trace.end_time = time.time()

    if outcome.interruptions:
        new_state = RunState(
            agent=outcome.current_agent,
            original_input=state.original_input,
            generated_items=items,
            ready_results=outcome.ready_results,
            pending=outcome.interruptions,
            context_wrapper=state.context_wrapper,
            trace=state.trace,
            input_guardrail_results=list(state.context_wrapper.input_guardrail_results),
            output_guardrail_results=list(state.context_wrapper.output_guardrail_results),
            tool_input_guardrail_results=list(state.context_wrapper.tool_input_guardrail_results),
            tool_output_guardrail_results=list(state.context_wrapper.tool_output_guardrail_results),
        )
        _export(state.trace)
        return RunResult(
            final_output=None,
            context_wrapper=state.context_wrapper,
            trace=state.trace,
            _original_input=state.original_input,
            _generated_items=outcome.generated,
            interruptions=outcome.interruptions,
            _state=new_state,
            input_guardrail_results=list(state.context_wrapper.input_guardrail_results),
            output_guardrail_results=list(state.context_wrapper.output_guardrail_results),
            tool_input_guardrail_results=list(state.context_wrapper.tool_input_guardrail_results),
            tool_output_guardrail_results=list(state.context_wrapper.tool_output_guardrail_results),
        )

    await hooks.on_agent_end(state.context_wrapper, outcome.current_agent, outcome.final_output)
    _export(state.trace)
    return RunResult(
        final_output=outcome.final_output,
        context_wrapper=state.context_wrapper,
        trace=state.trace,
        _original_input=state.original_input,
        _generated_items=[*state.generated_items[:-1], *outcome.generated],
        input_guardrail_results=list(state.context_wrapper.input_guardrail_results),
        output_guardrail_results=list(state.context_wrapper.output_guardrail_results),
        tool_input_guardrail_results=list(state.context_wrapper.tool_input_guardrail_results),
        tool_output_guardrail_results=list(state.context_wrapper.tool_output_guardrail_results),
    )


__all__ = ["_default_hooks", "_resume", "_run_async", "_run_turns"]
