"""tool_execution.py: executing one message's tool calls, handoffs, approval gating, guardrails."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from runa._types import RunContextWrapper, TResponseInputItem
from runa.exceptions import DuplicateToolCallError
from runa.lifecycle import RunHooks
from runa.run_internal.agent_runner_helpers import (
    _agent_tools,
    _find_tool,
    _gate_tool_call,
    _model_settings,
    _normalized_handoffs,
    _parse_arguments,
)
from runa.run_internal.guardrails import _run_tool_input_guardrails, _run_tool_output_guardrails
from runa.run_internal.spans import _close_span, _new_span
from runa.run_state import Interruption
from runa.tool import FunctionTool
from runa.tracing.traces import Trace


async def _run_tool_call(
    tool: FunctionTool,
    call: dict[str, Any],
    context_wrapper: RunContextWrapper,
    agent: Any,
    hooks: RunHooks[Any],
    trace: Trace,
    parent_id: str,
) -> TResponseInputItem:
    """Run one already-approved tool call end to end: guardrails, invocation, guardrails.

    Guards against executing the same `call_id` twice (e.g. a resumed/duplicated `RunState`)
    before anything else runs.
    """
    call_id = call["id"]
    if call_id in context_wrapper.executed_call_ids:
        raise DuplicateToolCallError(call_id, tool.name)
    context_wrapper.executed_call_ids.add(call_id)

    args_json = call["function"]["arguments"] or "{}"
    span_type = "delegate" if tool.is_delegate else "tool"
    span = _new_span(trace, parent_id, tool.name, span_type, input=args_json)
    await hooks.on_tool_start(context_wrapper, agent, tool)
    try:
        await _run_tool_input_guardrails(tool, args_json, call_id, context_wrapper, trace, span.id)
        try:
            result = await tool.on_invoke_tool(context_wrapper, args_json, call_id)
            error: str | None = None
        except Exception as exc:  # noqa: BLE001 -- a tool failing is data, not a run-ending error
            result = f"error: {exc}"
            error = str(exc)
        await _run_tool_output_guardrails(
            tool, args_json, call_id, result, context_wrapper, trace, span.id
        )
    except BaseException as exc:  # a tripwire, a cancelled sibling call, ...: say which
        detail = str(exc)
        _close_span(span, error=f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__)
        raise
    _close_span(span, error=error, output=result)
    await hooks.on_tool_end(context_wrapper, agent, tool, result)
    return {"role": "tool", "tool_call_id": call_id, "content": str(result)}


@dataclass
class _TurnOutcome:
    final_output: str | None
    generated: list[TResponseInputItem]
    interruptions: list[Interruption]
    ready_results: list[TResponseInputItem]
    current_agent: Any
    context_tokens: int = 0


async def _run_message_tool_calls(
    message: dict[str, Any],
    current_agent: Any,
    context_wrapper: RunContextWrapper,
    hooks: RunHooks[Any],
    trace: Trace,
    parent_id: str,
    approvals: dict[str, bool] | None,
    rejection_messages: dict[str, str] | None = None,
    ready_results: list[TResponseInputItem] | None = None,
) -> tuple[list[TResponseInputItem], list[Interruption], Any]:
    """Execute (or defer for approval) every tool call in `message`; returns results so far.

    Calls are gated one by one, in order, then the approved ones run concurrently, unless the
    agent's `model_settings.parallel_tool_calls` is `False`. Results keep the message's call
    order either way. `ready_results` are results a paused run already computed for calls in
    `message`: reused as is on resume, never executed a second time.
    """
    handoff_map = _normalized_handoffs(getattr(current_agent, "handoffs", []))
    tools = await _agent_tools(current_agent)
    results: list[TResponseInputItem] = []
    interruptions: list[Interruption] = []
    switched_agent: Any = None
    approvals = approvals or {}
    ready = {result["tool_call_id"]: result for result in ready_results or []}
    runs: dict[int, Callable[[], Awaitable[TResponseInputItem]]] = {}

    for call in message.get("tool_calls") or []:
        name = call["function"]["name"]
        call_id = call["id"]
        if name in handoff_map:
            handoff = handoff_map[name]
            switched_agent = handoff.agent
            span = _new_span(trace, parent_id, handoff.tool_name, "handoff")
            _close_span(span)
            await hooks.on_handoff(context_wrapper, current_agent, switched_agent)
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": f"Transferred to {switched_agent.name}.",
                }
            )
            continue

        if call_id in ready:
            results.append(ready[call_id])
            continue
        tool = _find_tool(tools, name)
        if tool is None:
            results.append(
                {"role": "tool", "tool_call_id": call_id, "content": f"error: unknown tool {name}"}
            )
            continue

        args_json = call["function"]["arguments"] or "{}"
        args = _parse_arguments(args_json)
        if isinstance(args, str):
            results.append({"role": "tool", "tool_call_id": call_id, "content": args})
            continue
        gate = await _gate_tool_call(
            tool, args, call_id, context_wrapper, approvals, rejection_messages
        )
        if gate.action == "interrupt":
            interruptions.append(
                Interruption(
                    name=name, arguments=args_json, call_id=call_id, tool=tool, agent=current_agent
                )
            )
            continue
        if gate.action == "reject":
            results.append({"role": "tool", "tool_call_id": call_id, "content": gate.message})
            continue

        runs[len(results)] = partial(
            _run_tool_call, tool, call, context_wrapper, current_agent, hooks, trace, parent_id
        )
        results.append({})  # filled in once the approved calls have run

    parallel = _model_settings(current_agent).parallel_tool_calls is not False
    for index, result in zip(runs, await _execute(list(runs.values()), parallel), strict=True):
        results[index] = result
    return results, interruptions, switched_agent


async def _execute(
    calls: list[Callable[[], Awaitable[TResponseInputItem]]], parallel: bool
) -> list[TResponseInputItem]:
    """Run `calls` concurrently (or one by one), cancelling the rest if one raises."""
    if not parallel:
        return [await call() for call in calls]
    tasks = [asyncio.ensure_future(call()) for call in calls]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)  # let their spans close
        raise


__all__ = ["_TurnOutcome", "_run_message_tool_calls", "_run_tool_call"]
