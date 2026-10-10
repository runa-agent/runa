"""tool_execution.py: executing one message's tool calls, handoffs, approval gating, guardrails."""

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from runa._items import ConversationItem, parsed_arguments
from runa._types import RunContextWrapper
from runa.exceptions import DuplicateToolCallError
from runa.guardrail import Phase, ToolInputGuardrailData
from runa.handoff import DelegatePaused
from runa.run_internal.active_run import _Run
from runa.run_internal.guardrails import _run_guardrails
from runa.run_internal.spans import _close_span
from runa.run_state import Interruption, RunState
from runa.tool import FunctionTool, ToolCall


async def _run_tool_call(
    run: _Run, tool: FunctionTool, call: dict[str, Any]
) -> ConversationItem | DelegatePaused:
    """Run one already-approved tool call end to end: guardrails, invocation, guardrails.

    Guards against executing the same `call_id` twice (e.g. a resumed/duplicated `RunState`)
    before anything else runs. A delegate that paused for approval comes back as its
    `DelegatePaused`, not a result: the call hasn't finished, so resuming may run it again.
    """
    context_wrapper = run.context_wrapper
    call_id = call["id"]
    if not context_wrapper.approval_ledger.claim(call_id):
        raise DuplicateToolCallError(call_id, tool.name)

    args_json = call["function"]["arguments"] or "{}"
    span_type = "delegate" if tool.delegate is not None else "tool"
    span = run.span(tool.name, span_type, input=args_json)
    inside = run.spans.under(span)  # this call's guardrails belong under the call
    data = ToolInputGuardrailData.of(tool.name, args_json, call_id)
    await run.hooks.on_tool_start(context_wrapper, run.current_agent, tool)
    try:
        await _run_guardrails(run, Phase.TOOL_INPUT, data, tool=tool, spans=inside)
        try:
            result = await tool.on_invoke_tool(context_wrapper, args_json, call_id)
            error: str | None = None
        except DelegatePaused as paused:
            context_wrapper.approval_ledger.release(call_id)
            _close_span(span, output="paused for approval")
            return paused
        except Exception as exc:  # noqa: BLE001 -- a tool failing is data, not a run-ending error
            result = f"error: {exc}"
            error = str(exc)
        await _run_guardrails(
            run, Phase.TOOL_OUTPUT, data.returning(result), tool=tool, spans=inside
        )
    except BaseException as exc:  # a tripwire, a cancelled sibling call, ...: say which
        detail = str(exc)
        _close_span(span, error=f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__)
        raise
    _close_span(span, error=error, output=result)
    if tool.delegate is None:  # a delegate's own run records its own calls, under its own trace
        run.tool_calls.append(ToolCall(tool.name, args_json, str(result)))
    await run.hooks.on_tool_end(context_wrapper, run.current_agent, tool, result)
    return {"role": "tool", "tool_call_id": call_id, "content": str(result)}


def _arguments_or_error(args_json: str) -> dict[str, Any] | str:
    """A tool call's arguments as a dict, or an error string to feed back to the model.

    Arguments a model got wrong are data, not a failed run: the string goes back as that call's
    result and the model gets to try again. What counts as wrong is `_items`' to say.
    """
    try:
        return parsed_arguments(args_json)
    except ValueError as exc:
        return f"error: {exc}"


async def _needs_approval(
    tool: FunctionTool, context_wrapper: RunContextWrapper, args: dict[str, Any], call_id: str
) -> bool:
    """Ask the tool itself whether this call needs a human, `bool` or predicate alike."""
    if isinstance(tool.needs_approval, bool):
        return tool.needs_approval
    verdict = tool.needs_approval(context_wrapper, args, call_id)
    return bool(await verdict if inspect.isawaitable(verdict) else verdict)


@dataclass
class _TurnOutcome:
    """How the turn loop came out: an answer, or interruptions waiting on a human.

    Only what the loop couldn't put on its `_Run` as it went. Everything it did record there --
    the generated items, which agent ended up answering -- `_finish` reads off the run.
    """

    final_output: Any
    interruptions: list[Interruption]
    ready_results: list[ConversationItem]
    context_tokens: int = 0


async def _run_message_tool_calls(
    run: _Run, message: dict[str, Any], resume: RunState | None = None
) -> tuple[list[ConversationItem], list[Interruption], Any]:
    """Execute (or defer for approval) every tool call in `message`; returns results so far.

    Calls are gated one by one, in order, then the approved ones run concurrently, unless the
    agent's `model_settings.parallel_tool_calls` is `False`. Results keep the message's call
    order either way. `resume` is the paused state being continued, read for its decisions about
    the calls in `message` and the results it already computed for the rest of them: those are
    reused as is, never executed a second time.
    """
    shape = run.shape
    agent = shape.agent
    context_wrapper = run.context_wrapper
    handoff_map = shape.handoffs
    results: list[ConversationItem] = []
    interruptions: list[Interruption] = []
    switched_agent: Any = None
    approvals = resume.approvals if resume is not None else {}
    rejection_messages = resume.rejection_messages if resume is not None else None
    ready = {
        result["tool_call_id"]: result
        for result in (resume.ready_results if resume is not None else [])
    }
    runs: dict[int, Callable[[], Awaitable[ConversationItem | DelegatePaused]]] = {}

    for call in message.get("tool_calls") or []:
        name = call["function"]["name"]
        call_id = call["id"]
        if name in handoff_map:
            handoff = handoff_map[name]
            switched_agent = handoff.agent
            span = run.span(handoff.tool_name, "handoff")
            _close_span(span)
            await run.hooks.on_handoff(context_wrapper, agent, switched_agent)
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
        tool = shape.find_tool(name)
        if tool is None:
            results.append(
                {"role": "tool", "tool_call_id": call_id, "content": f"error: unknown tool {name}"}
            )
            continue

        args_json = call["function"]["arguments"] or "{}"
        args = _arguments_or_error(args_json)
        if isinstance(args, str):
            results.append({"role": "tool", "tool_call_id": call_id, "content": args})
            continue
        gate = await context_wrapper.approval_ledger.decide(
            tool.name,
            call_id,
            needs_approval=partial(_needs_approval, tool, context_wrapper, args, call_id),
            approvals=approvals,
            rejection_messages=rejection_messages,
        )
        if gate.action == "interrupt":
            interruptions.append(
                Interruption(name=name, arguments=args_json, call_id=call_id, agent=agent)
            )
            continue
        if gate.action == "reject":
            results.append({"role": "tool", "tool_call_id": call_id, "content": gate.message})
            continue

        runs[len(results)] = partial(_run_tool_call, run, tool, call)
        results.append({})  # filled in once the approved calls have run

    parallel = shape.model_settings.parallel_tool_calls is not False
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
