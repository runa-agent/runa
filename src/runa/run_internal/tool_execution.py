"""tool_execution.py: one message's tool calls, delegates and handoffs, approval gating, guardrails.

The run these steps are passed is the turn loop's, imported for its type only: a delegate call
reaches back into the loop through the run it is handed, never through an import, so the runtime
dependency still points from the loop to here.
"""

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from runa._items import (
    ConversationItem,
    answered_call_id,
    call_arguments,
    call_name,
    parsed_arguments,
    tool_calls,
    tool_result,
)
from runa._items import call_id as id_of  # `call_id` here is the id of the call at hand
from runa._types import RunContext
from runa.agent_like import AgentLike
from runa.exceptions import DuplicateToolCallError
from runa.guardrail import Phase
from runa.handoff import Delegate
from runa.run_internal.guardrails import _run_guardrails
from runa.run_state import Interruption, RunState
from runa.tool import FunctionTool, ToolCall

if TYPE_CHECKING:
    from runa.run_internal.run_loop import _Run


class DelegatePaused(Exception):  # noqa: N818 -- a signal, not an error
    """Raised by a delegate call whose nested run paused: carries its `interruptions` up."""

    def __init__(self, interruptions: list[Interruption]) -> None:
        """Hold the nested run's pending `Interruption`s for the caller's run to surface."""
        super().__init__(f"delegate paused on {len(interruptions)} approval(s)")
        self.interruptions = interruptions


def _claim(run: _Run, call_id: str, name: str) -> None:
    """Refuse to execute `call_id` a second time (a resumed or duplicated `RunState`)."""
    if not run.frame.governance.approval_ledger.claim(call_id):
        raise DuplicateToolCallError(call_id, name)


async def _run_tool_call(run: _Run, tool: FunctionTool, call: dict[str, Any]) -> ConversationItem:
    """Run one already-approved tool call end to end: guardrails, invocation, guardrails."""
    run_context = run.frame.run_context
    call_id = id_of(call)
    _claim(run, call_id, tool.name)

    args_json = call_arguments(call)
    with run.span(tool.name, "tool", input=args_json) as span:
        inside = run.spans.under(span)  # this call's guardrails belong under the call
        await run.frame.hooks.on_tool_start(run_context, run.current_agent, tool)
        await _run_guardrails(run, Phase.TOOL_INPUT, args_json, tool=tool, spans=inside)
        try:
            result = await tool.on_invoke_tool(run_context, args_json, call_id)
        except Exception as exc:  # noqa: BLE001 -- a tool failing is data, not a run-ending error
            result = f"error: {exc}"
            span.error = str(exc)
        await _run_guardrails(run, Phase.TOOL_OUTPUT, result, tool=tool, spans=inside)
        span.output = result
    run.transcript.record_call(ToolCall(tool.name, args_json, str(result)))
    await run.frame.hooks.on_tool_end(run_context, run.current_agent, tool, result)
    return tool_result(call_id, result)


async def _run_delegate_call(
    run: _Run, delegate: Delegate, call: dict[str, Any]
) -> ConversationItem | DelegatePaused:
    """Run one delegate call as a nested run of this one, under the call's `"delegate"` span.

    Recorded like a tool call, its `children` the calls the nested run made. Arguments the model
    got wrong, or a delegate that errored, answer `"error: ..."`, for the model to respond to;
    one that paused for approval comes back as its `DelegatePaused`, not a result: the call
    hasn't finished, so resuming may run it again.
    """
    call_id = id_of(call)
    args_json = call_arguments(call)
    args = _arguments_or_error(args_json)
    if isinstance(args, str):
        return tool_result(call_id, args)
    _claim(run, call_id, delegate.tool_name)

    with run.span(delegate.tool_name, "delegate", input=args_json) as span:
        try:
            nested = await run.delegate(delegate, str(args.get("input", "")), call_id, under=span)
        except DelegatePaused as paused:
            run.frame.governance.approval_ledger.release(call_id)
            span.output = "paused for approval"
            return paused
        if nested.status == "completed":
            result = nested.output
        else:
            result = f"error: {nested.error}"
            span.error = nested.error
        span.output = result
    run.transcript.record_call(
        ToolCall(delegate.tool_name, args_json, str(result), children=nested.tool_calls)
    )
    return tool_result(call_id, result)


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
    tool: FunctionTool, run_context: RunContext, args: dict[str, Any], call_id: str
) -> bool:
    """Ask the tool itself whether this call needs a human, `bool` or predicate alike."""
    if isinstance(tool.needs_approval, bool):
        return tool.needs_approval
    verdict = tool.needs_approval(run_context, args, call_id)
    return bool(await verdict if inspect.isawaitable(verdict) else verdict)


type _Thunk = Callable[[], Awaitable[ConversationItem | DelegatePaused]]


@dataclass
class _Slot:
    """One tool call's place in the turn's results, in the message's call order.

    A call that was handed off, rejected, unknown, malformed or reused from a paused state has
    its `item` from the start; an approved one is given one once it has run. A call waiting on a
    human never gets a slot, and a delegate that paused leaves its own empty: neither has
    finished, so neither has a result to send the model, and `_run_message_tool_calls` returns
    only the slots that were filled.
    """

    item: ConversationItem | None = None


@dataclass
class _TurnOutcome:
    """How the turn loop came out: an answer, or interruptions waiting on a human.

    Only what the loop couldn't record on the run as it went; the generated items and which
    agent ended up answering are read off the run itself.
    """

    final_output: Any
    interruptions: list[Interruption]
    ready_results: list[ConversationItem]
    context_tokens: int = 0


async def _run_message_tool_calls(
    run: _Run, message: dict[str, Any], resume: RunState | None = None
) -> tuple[list[ConversationItem], list[Interruption], AgentLike | None]:
    """Execute (or defer for approval) every tool call in `message`; returns results so far.

    Calls are gated one by one, in order, then the approved ones run concurrently, unless the
    agent's `model_settings.parallel_tool_calls` is `False`. Results keep the message's call
    order either way. A call that needs approval is decided by the run's approval ledger.
    `resume` is the paused state being continued, read for the results it already computed for
    calls in `message` that didn't need approval: those are reused as is, never executed a
    second time.
    """
    shape = run.shape
    agent = shape.agent
    run_context = run.frame.run_context
    handoff_map = shape.handoffs
    slots: list[_Slot] = []
    deferred: list[tuple[_Slot, _Thunk]] = []
    interruptions: list[Interruption] = []
    switched_agent: AgentLike | None = None
    ready = {
        answered_call_id(result): result
        for result in (resume.ready_results if resume is not None else [])
    }

    for call in tool_calls(message):
        name = call_name(call)
        call_id = id_of(call)
        if name in handoff_map:
            handoff = handoff_map[name]
            switched_agent = handoff.agent
            with run.span(handoff.tool_name, "handoff"):
                await run.frame.hooks.on_handoff(run_context, agent, switched_agent)
            slots.append(_Slot(tool_result(call_id, f"Transferred to {switched_agent.name}.")))
            continue

        if call_id in ready:
            slots.append(_Slot(ready[call_id]))
            continue
        if name in shape.delegates:  # needs no approval of its own: its tools ask for theirs
            slot = _Slot()  # filled in below, once the approved calls have run
            slots.append(slot)
            deferred.append((slot, partial(_run_delegate_call, run, shape.delegates[name], call)))
            continue
        tool = shape.find_tool(name)
        if tool is None:
            slots.append(_Slot(tool_result(call_id, f"error: unknown tool {name}")))
            continue

        args_json = call_arguments(call)
        args = _arguments_or_error(args_json)
        if isinstance(args, str):
            slots.append(_Slot(tool_result(call_id, args)))
            continue
        gate = await run.frame.governance.approval_ledger.decide(
            tool.name,
            call_id,
            needs_approval=partial(_needs_approval, tool, run_context, args, call_id),
        )
        if gate.action == "interrupt":
            interruptions.append(
                Interruption(name=name, arguments=args_json, call_id=call_id, agent=agent)
            )
            continue
        if gate.action == "reject":
            slots.append(_Slot(tool_result(call_id, gate.message)))
            continue

        slot = _Slot()  # filled in below, once the approved calls have run
        slots.append(slot)
        deferred.append((slot, partial(_run_tool_call, run, tool, call)))

    parallel = agent.model_settings.parallel_tool_calls is not False
    outcomes = await _execute([thunk for _, thunk in deferred], parallel)
    for (slot, _), outcome in zip(deferred, outcomes, strict=True):
        if isinstance(outcome, DelegatePaused):
            interruptions.extend(outcome.interruptions)  # unfinished: its slot stays empty
        else:
            slot.item = outcome
    return [slot.item for slot in slots if slot.item is not None], interruptions, switched_agent


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


__all__ = [
    "DelegatePaused",
    "_TurnOutcome",
    "_run_delegate_call",
    "_run_message_tool_calls",
    "_run_tool_call",
]
