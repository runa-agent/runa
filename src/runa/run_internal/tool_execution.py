"""tool_execution.py: executing one message's tool calls, handoffs, approval gating, guardrails."""

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

from runa._types import RunContextWrapper, TResponseInputItem
from runa.exceptions import DuplicateToolCallError
from runa.handoff import DelegatePaused
from runa.lifecycle import _Dispatch
from runa.run_internal.agent_shape import AgentLike, _agent_tools, _find_tool, _normalized_handoffs
from runa.run_internal.guardrails import _run_tool_input_guardrails, _run_tool_output_guardrails
from runa.run_internal.spans import _close_span, _new_span
from runa.run_state import Interruption
from runa.tool import FunctionTool
from runa.tracing.traces import Trace


async def _run_tool_call(
    tool: FunctionTool,
    call: dict[str, Any],
    context_wrapper: RunContextWrapper,
    agent: AgentLike,
    hooks: _Dispatch[Any],
    trace: Trace,
    parent_id: str,
) -> TResponseInputItem | DelegatePaused:
    """Run one already-approved tool call end to end: guardrails, invocation, guardrails.

    Guards against executing the same `call_id` twice (e.g. a resumed/duplicated `RunState`)
    before anything else runs. A delegate that paused for approval comes back as its
    `DelegatePaused`, not a result: the call hasn't finished, so resuming may run it again.
    """
    call_id = call["id"]
    if call_id in context_wrapper.executed_call_ids:
        raise DuplicateToolCallError(call_id, tool.name)
    context_wrapper.executed_call_ids.add(call_id)

    args_json = call["function"]["arguments"] or "{}"
    span_type = "delegate" if tool.delegate is not None else "tool"
    span = _new_span(trace, parent_id, tool.name, span_type, input=args_json)
    await hooks.on_tool_start(context_wrapper, agent, tool)
    try:
        await _run_tool_input_guardrails(tool, args_json, call_id, context_wrapper, trace, span.id)
        try:
            result = await tool.on_invoke_tool(context_wrapper, args_json, call_id)
            error: str | None = None
        except DelegatePaused as paused:
            context_wrapper.executed_call_ids.discard(call_id)
            _close_span(span, output="paused for approval")
            return paused
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


def _parse_arguments(args_json: str) -> dict[str, Any] | str:
    """A tool call's arguments as a dict, or an error string to feed back to the model."""
    try:
        args = json.loads(args_json or "{}")
    except json.JSONDecodeError as exc:
        return f"error: invalid JSON arguments: {exc}"
    if not isinstance(args, dict):
        return "error: tool arguments must be a JSON object"
    return args


async def _needs_approval(
    tool: FunctionTool, context_wrapper: RunContextWrapper, args: dict[str, Any], call_id: str
) -> bool:
    if isinstance(tool.needs_approval, bool):
        return tool.needs_approval
    verdict = tool.needs_approval(context_wrapper, args, call_id)
    return bool(await verdict if inspect.isawaitable(verdict) else verdict)


@dataclass
class _ApprovalGate:
    """What `_gate_tool_call` decided for one tool call."""

    action: Literal["run", "reject", "interrupt"]
    message: str | None = None


async def _gate_tool_call(
    tool: FunctionTool,
    args: dict[str, Any],
    call_id: str,
    context_wrapper: RunContextWrapper,
    approvals: dict[str, bool] | None = None,
    rejection_messages: dict[str, str] | None = None,
) -> _ApprovalGate:
    """Decide whether a tool call should run, be rejected, or pause for approval.

    Consults `context_wrapper.approval_ledger` first -- the sticky "always approve"/"always
    reject" decisions set via `RunState.approve`/`.reject(..., always=True)` -- before falling
    back to `_needs_approval` and the per-call-id `approvals` dict.
    """
    sticky = context_wrapper.approval_ledger.get(tool.name)
    if sticky is True:
        return _ApprovalGate("run")
    if sticky is False:
        return _ApprovalGate(
            "reject",
            context_wrapper.approval_ledger_messages.get(tool.name, "rejected by the operator"),
        )
    if not await _needs_approval(tool, context_wrapper, args, call_id):
        return _ApprovalGate("run")
    verdict = (approvals or {}).get(call_id)
    if verdict is None:
        return _ApprovalGate("interrupt")
    if verdict is False:
        return _ApprovalGate(
            "reject", (rejection_messages or {}).get(call_id, "rejected by the operator")
        )
    return _ApprovalGate("run")


@dataclass
class _TurnOutcome:
    final_output: Any
    generated: list[TResponseInputItem]
    interruptions: list[Interruption]
    ready_results: list[TResponseInputItem]
    current_agent: Any
    context_tokens: int = 0


async def _run_message_tool_calls(
    message: dict[str, Any],
    current_agent: AgentLike,
    context_wrapper: RunContextWrapper,
    hooks: _Dispatch[Any],
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
    handoff_map = _normalized_handoffs(current_agent.handoffs)
    tools = await _agent_tools(current_agent)
    results: list[TResponseInputItem] = []
    interruptions: list[Interruption] = []
    switched_agent: Any = None
    approvals = approvals or {}
    ready = {result["tool_call_id"]: result for result in ready_results or []}
    runs: dict[int, Callable[[], Awaitable[TResponseInputItem | DelegatePaused]]] = {}

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

    parallel = current_agent.model_settings.parallel_tool_calls is not False
    paused: list[int] = []
    for index, result in zip(runs, await _execute(list(runs.values()), parallel), strict=True):
        if isinstance(result, DelegatePaused):
            interruptions.extend(result.interruptions)
            paused.append(index)
        else:
            results[index] = result
    results = [result for index, result in enumerate(results) if index not in paused]
    return results, interruptions, switched_agent


async def _execute[T](calls: list[Callable[[], Awaitable[T]]], parallel: bool) -> list[T]:
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
