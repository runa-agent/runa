"""tool_execution.py: one message's tool calls, delegates and handoffs.

Part of the turn loop, kept in its own file for length. These steps are passed the loop's run,
imported for its type only, and use what it can do (claim a call, decide its approval, guard it,
fire a hook, run a delegate), not how it is laid out.
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

from runa._items import (
    ConversationItem,
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
from runa.run_internal.record import Interruption, RunRecord
from runa.tool import FunctionTool, ToolCall
from runa.tracing.spans import Span

if TYPE_CHECKING:
    from runa.agent import Agent
    from runa.run_internal.run_loop import _Run


async def _run_call(
    run: _Run,
    callee: FunctionTool | Delegate,
    call: dict[str, Any],
    args: dict[str, Any],
    paused: RunRecord | None = None,
) -> ConversationItem | list[Interruption]:
    """Run one approved call end to end. Every tool call and delegate call goes through this.

    In order, the steps are: claim the call, fire `on_tool_start`, guard its input, invoke it,
    guard its output, record it, then fire `on_tool_end`. Only the invoke step differs: a tool
    runs its function, and a `Delegate` runs a nested run of this one. A delegate that pauses for
    approval comes back as its interruptions, because the call hasn't finished. Its call is run
    again with the delegate's `paused` record and continues where it stopped, so its start is not
    repeated and it ends once.

    `args` are the call's arguments as the gate already parsed them; the raw JSON is kept only
    for the span and the `ToolCall`, which record what the model wrote.
    """
    tool = _gated(callee)
    call_id = id_of(call)
    run.claim(call_id, tool.name)

    args_json = call_arguments(call)
    with run.span(
        tool.name, "delegate" if isinstance(callee, Delegate) else "tool", input=args_json
    ) as span:
        if paused is None:
            await run.hooks.on_tool_start(run.run_context, run.current_agent, tool)
            await run.guard(Phase.TOOL_INPUT, args, tool=tool, under=span)
        if isinstance(callee, Delegate):
            outcome = await _delegated(run, callee, call_id, args, args_json, span, paused)
        else:
            outcome = await _invoked(run, tool, call_id, args, args_json, span)
        if isinstance(outcome, list):
            span.output = "paused for approval"
            return outcome
        result, recorded = outcome
        await run.guard(Phase.TOOL_OUTPUT, result, tool=tool, under=span)
        span.output = result
    item = _answer(run, call_id, recorded)
    await run.hooks.on_tool_end(run.run_context, run.current_agent, tool, result)
    return item


def _gated(callee: FunctionTool | Delegate) -> FunctionTool:
    """What a call to `callee` is approved, guarded and handed to hooks as: a delegate's `tool`."""
    return callee.tool if isinstance(callee, Delegate) else callee


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
    run: _Run,
    delegate: Delegate,
    call_id: str,
    args: dict[str, Any],
    args_json: str,
    span: Span,
    paused: RunRecord | None,
) -> tuple[Any, ToolCall] | list[Interruption]:
    """Invoke `delegate` as a nested run under `span`: its output and the record of the call.

    The record's `children` are the calls the nested run made. A delegate that errored fails the
    call, and one that paused for approval comes back as its interruptions.
    """
    input = str(args.get("input", ""))
    nested = await run.delegate(delegate, input, call_id, under=span, paused=paused)
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
    """One tool call's place in the message's answers, in the message's call order.

    A call that was handed off, rejected, unknown, malformed or reused from a pause has its
    `item` from the start; an approved one is given one once it has run. A call waiting on a
    human never gets a slot, and a delegate that paused leaves its own empty: neither has
    finished, so neither has a result to send the model, and only the filled slots are answers.
    """

    item: ConversationItem | None = None


type _Answered = tuple[list[ConversationItem], list[Interruption], Agent | None]
"""One message's answers so far, the calls still waiting on a human, and the agent handed off to."""


@dataclass
class _Answers:
    """One message's tool calls on their way to answers: each one admitted, then all settled.

    Calls are admitted one by one, in order, then the approved ones run concurrently, unless the
    agent's `model_settings.parallel_tool_calls` is `False`. Answers keep the message's call
    order either way.
    """

    run: _Run
    slots: list[_Slot] = field(default_factory=list)
    deferred: list[tuple[_Slot, _Thunk]] = field(default_factory=list)
    interruptions: list[Interruption] = field(default_factory=list)
    switched: Agent | None = None

    def answer(self, item: ConversationItem) -> None:
        """Answer the next call with `item` as it is."""
        self.slots.append(_Slot(item))

    def defer(self, thunk: _Thunk) -> None:
        """Answer the next call by running `thunk` once every call is admitted."""
        slot = _Slot()
        self.slots.append(slot)
        self.deferred.append((slot, thunk))

    async def admit(self, call: dict[str, Any]) -> None:
        """Gate `call`, and answer it, defer it, or leave it waiting on a human.

        A tool's call and a delegate's are gated alike: one no tool answers, or whose arguments
        don't parse, fails; any other is decided by the run's approval ledger, then run by
        `_run_call`, rejected, or left waiting. A handoff is the one call that isn't gated,
        because it is a switch rather than a call. It fires `on_handoff`, not the tool hooks, and
        is not one of `run.tool_calls`.
        """
        run, shape = self.run, self.run.shape
        name, call_id = call_name(call), id_of(call)
        handoff = shape.handoffs.get(name)
        if handoff is not None:
            self.switched = handoff.agent
            with run.span(handoff.tool_name, "handoff"):
                await run.hooks.on_handoff(run.run_context, run.current_agent, handoff.agent)
            self.answer(tool_result(call_id, f"Transferred to {handoff.agent.name}."))
            return

        args_json = call_arguments(call)
        callee = shape.callables.get(name)
        if callee is None:
            self.answer(_answer(run, call_id, _failed(name, args_json, f"unknown tool {name}")))
            return
        args = _arguments_or_error(args_json)
        if isinstance(args, str):
            self.answer(_answer(run, call_id, _failed(name, args_json, args)))
            return

        gate = await run.decide(_gated(callee), call_id, args)
        if gate.action == "interrupt":
            self.interruptions.append(
                Interruption(name=name, arguments=args_json, call_id=call_id, agent=shape.agent)
            )
        elif gate.action == "reject":
            self.answer(tool_result(call_id, gate.message))
        else:
            self.defer(partial(_run_call, run, callee, call, args))

    def go_on(self, call: dict[str, Any], paused: RunRecord) -> None:
        """Defer `call`, a delegate's that paused with `paused` as its record, to continue it."""
        delegate = self.run.shape.delegates[call_name(call)]
        args = parsed_arguments(call_arguments(call))
        self.defer(partial(_run_call, self.run, delegate, call, args, paused))

    async def settled(self) -> _Answered:
        """Run the deferred calls, and return the message's answers."""
        parallel = self.run.shape.model_settings.parallel_tool_calls is not False
        outcomes = await _execute([thunk for _, thunk in self.deferred], parallel)
        for (slot, _), outcome in zip(self.deferred, outcomes, strict=True):
            if isinstance(outcome, list):
                self.interruptions.extend(outcome)  # a paused delegate: its slot stays empty
            else:
                slot.item = outcome
        answered = [slot.item for slot in self.slots if slot.item is not None]
        return answered, self.interruptions, self.switched


async def _run_message_tool_calls(run: _Run, message: dict[str, Any]) -> _Answered:
    """Execute (or defer for approval) every tool call in `message`, as `_Answers` admits them."""
    answers = _Answers(run)
    for call in tool_calls(message):
        await answers.admit(call)
    return await answers.settled()


async def _continue_message_tool_calls(
    run: _Run,
    message: dict[str, Any],
    ready: Mapping[str, ConversationItem],
    delegates: Mapping[str, RunRecord],
) -> _Answered:
    """Settle the calls of `message`, the one the run paused on, from where the pause left them.

    A call answered before the pause (`ready`, by call id) is reused, never run again. A
    delegate call whose delegate paused (`delegates`, its record by call id) continues that
    delegate's run: it was admitted and started before the pause, so neither happens again.
    Every other call is admitted again, and one that waited on a human finds its decision on the
    ledger. So is a handoff, answered before the pause but never made, because the run paused
    before switching.
    """
    answers = _Answers(run)
    for call in tool_calls(message):
        call_id = id_of(call)
        if call_id in delegates:
            answers.go_on(call, delegates[call_id])
        elif call_id in ready and call_name(call) not in run.shape.handoffs:
            answers.answer(ready[call_id])
        else:
            await answers.admit(call)
    return await answers.settled()


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
    "_TurnOutcome",
    "_continue_message_tool_calls",
    "_run_call",
    "_run_message_tool_calls",
]
