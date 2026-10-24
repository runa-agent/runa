"""tool_execution.py: one message's tool calls, delegates and handoffs.

Part of the turn loop, kept in its own file for length. These steps are passed the loop's run,
imported for its type only, and use what it can do (claim a call, decide its approval, guard it,
fire a hook, run a delegate), not how it is laid out.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
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
from runa._log import logger
from runa.exceptions import ToolError
from runa.guardrail import Phase
from runa.handoff import Delegate
from runa.run_internal.record import Interruption
from runa.tool import FunctionTool, ToolCall
from runa.tracing.spans import Span

if TYPE_CHECKING:
    from runa.agent import Agent
    from runa.run_internal.run_loop import _Run


async def _run_call(
    run: _Run,
    tool: FunctionTool,
    call: dict[str, Any],
    args: dict[str, Any],
    delegate: Delegate | None = None,
) -> ConversationItem | list[Interruption]:
    """Run one approved call end to end. Every tool call and delegate call goes through this.

    In order, the steps are: claim the call, fire `on_tool_start`, guard its input, invoke it,
    guard its output, record it, then fire `on_tool_end`. Only the invoke step differs: a tool
    runs its function, and `delegate` runs a nested run of this one. A delegate that pauses for
    approval comes back as its interruptions, because the call hasn't finished. When that call
    resumes it continues where it stopped, so its start is not repeated and it ends once.

    `args` are the call's arguments as the gate already parsed them; the raw JSON is kept only
    for the span and the `ToolCall`, which record what the model wrote.
    """
    call_id = id_of(call)
    run.claim(call_id, tool.name)

    args_json = call_arguments(call)
    continuing = call_id in run.record.delegates
    with run.span(tool.name, "tool" if delegate is None else "delegate", input=args_json) as span:
        if not continuing:
            await run.hooks.on_tool_start(run.run_context, run.current_agent, tool)
            await run.guard(Phase.TOOL_INPUT, args, tool=tool, under=span)
        if delegate is None:
            outcome = await _invoked(run, tool, call_id, args, args_json, span)
        else:
            outcome = await _delegated(run, delegate, call_id, args, args_json, span)
        if isinstance(outcome, list):
            span.output = "paused for approval"
            return outcome
        result, recorded = outcome
        await run.guard(Phase.TOOL_OUTPUT, result, tool=tool, under=span)
        span.output = result
    item = _answer(run, call_id, recorded)
    await run.hooks.on_tool_end(run.run_context, run.current_agent, tool, result)
    return item


async def _invoked(
    run: _Run, tool: FunctionTool, call_id: str, args: dict[str, Any], args_json: str, span: Span
) -> tuple[Any, ToolCall]:
    """Invoke `tool`: its result, and the record of the call. A tool that raises fails the call."""
    try:
        result = await tool.on_invoke_tool(run.run_context, args, call_id)
    except Exception as exc:  # noqa: BLE001 -- a tool failing is data, not a run-ending error
        failed = _failed(tool.name, args_json, str(exc) or type(exc).__name__, exc, span)
        return failed.output, failed
    return result, ToolCall(tool.name, args_json, str(result))


async def _delegated(
    run: _Run, delegate: Delegate, call_id: str, args: dict[str, Any], args_json: str, span: Span
) -> tuple[Any, ToolCall] | list[Interruption]:
    """Invoke `delegate` as a nested run under `span`: its output and the record of the call.

    The record's `children` are the calls the nested run made. A delegate that errored fails the
    call, and one that paused for approval comes back as its interruptions.
    """
    nested = await run.delegate(delegate, str(args.get("input", "")), call_id, under=span)
    if isinstance(nested, list):
        return nested
    name = delegate.tool.name
    if nested.status == "completed":
        output = nested.output
        return output, ToolCall(name, args_json, str(output), children=nested.tool_calls)
    error = nested.error or "the delegate failed"
    failed = _failed(name, args_json, error, span=span, children=nested.tool_calls)
    return failed.output, failed


def _arguments_or_error(args_json: str) -> dict[str, Any] | str:
    """A tool call's arguments as a dict, or why they aren't one.

    Arguments a model got wrong are data, not a failed run: they fail the call, and the model
    gets to try again. What counts as wrong is `_items`' to say.
    """
    try:
        return parsed_arguments(args_json)
    except ValueError as exc:
        return str(exc)


def _failed(
    name: str,
    arguments: str,
    error: str,
    exc: Exception | None = None,
    span: Span | None = None,
    *,
    children: tuple[ToolCall, ...] = (),
) -> ToolCall:
    """The record of a call that failed with `error`; its `output` is what the model is answered.

    Every way a call can fail comes through here. The model gets `"error: ..."` to respond to,
    and everyone else gets the failure as data: the record's `error`, the span's, and a warning
    in the log, with `exc`'s traceback unless the tool raised `ToolError` to fail on purpose.
    """
    if span is not None:
        span.error = error
    traceback = None if isinstance(exc, ToolError) else exc
    logger.warning("tool call %s failed: %s", name, error, exc_info=traceback)
    return ToolCall(name, arguments, f"error: {error}", children=children, error=error)


def _answer(run: _Run, call_id: str, call: ToolCall) -> ConversationItem:
    """Record `call` on the run, and answer the model's call `call_id` with its output."""
    run.record.tool_calls.append(call)
    return tool_result(call_id, call.output)


type _Thunk = Callable[[], Awaitable[ConversationItem | list[Interruption]]]


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
    run: _Run, message: dict[str, Any], ready_results: Sequence[ConversationItem] = ()
) -> tuple[list[ConversationItem], list[Interruption], Agent | None]:
    """Execute (or defer for approval) every tool call in `message`; returns results so far.

    Calls are gated one by one, in order, then the approved ones run concurrently, unless the
    agent's `model_settings.parallel_tool_calls` is `False`. Results keep the message's call
    order either way. A tool's call and a delegate's are gated alike: a call that needs approval
    is decided by the run's approval ledger, then run by `_run_call`. A delegate call resuming
    after its delegate paused already passed the gate and is not decided again.

    A handoff is the one call that isn't gated, because it is a switch rather than a call. It
    fires `on_handoff`, not the tool hooks, and is not one of `run.tool_calls`.

    `ready_results` are a paused run's results already computed for calls in `message` that
    didn't need approval: those are reused as is, never executed a second time.
    """
    shape = run.shape
    handoff_map = shape.handoffs
    slots: list[_Slot] = []
    deferred: list[tuple[_Slot, _Thunk]] = []
    interruptions: list[Interruption] = []
    switched_agent: Agent | None = None
    ready = {answered_call_id(result): result for result in ready_results}

    for call in tool_calls(message):
        name = call_name(call)
        call_id = id_of(call)
        if name in handoff_map:
            handoff = handoff_map[name]
            switched_agent = handoff.agent
            with run.span(handoff.tool_name, "handoff"):
                await run.hooks.on_handoff(run.run_context, run.current_agent, switched_agent)
            slots.append(_Slot(tool_result(call_id, f"Transferred to {switched_agent.name}.")))
            continue

        if call_id in ready:
            slots.append(_Slot(ready[call_id]))
            continue
        args_json = call_arguments(call)
        tool = shape.callables.get(name)
        if tool is None:
            slots.append(
                _Slot(_answer(run, call_id, _failed(name, args_json, f"unknown tool {name}")))
            )
            continue

        args = _arguments_or_error(args_json)
        if isinstance(args, str):
            slots.append(_Slot(_answer(run, call_id, _failed(name, args_json, args))))
            continue
        if call_id not in run.record.delegates:
            gate = await run.decide(tool, call_id, args)
            if gate.action == "interrupt":
                interruptions.append(
                    Interruption(name=name, arguments=args_json, call_id=call_id, agent=shape.agent)
                )
                continue
            if gate.action == "reject":
                slots.append(_Slot(tool_result(call_id, gate.message)))
                continue

        slot = _Slot()  # filled in below, once the approved calls have run
        slots.append(slot)
        delegate = shape.delegates.get(name)
        deferred.append((slot, partial(_run_call, run, tool, call, args, delegate)))

    parallel = shape.model_settings.parallel_tool_calls is not False
    outcomes = await _execute([thunk for _, thunk in deferred], parallel)
    for (slot, _), outcome in zip(deferred, outcomes, strict=True):
        if isinstance(outcome, list):
            interruptions.extend(outcome)  # a paused delegate: its slot stays empty
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


__all__ = ["_TurnOutcome", "_run_call", "_run_message_tool_calls"]
