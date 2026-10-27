"""tool_execution.py: one message's tool calls, delegates and handoffs, and where they stand.

What happens to each call of a model message lives here, a pause included: which calls were
answered and which wait on a human, written to the run's record as each one settles, so a pause
on the message finds it recorded and resuming answers that message again from there. These
steps are passed the loop's run, imported for its type only, and use what it can do (claim a
call, decide its approval, guard it, fire a hook, run a delegate), not how it is laid out.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
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
from runa.handoff import Delegate, Handoff
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
    suspended: RunRecord | None = None,
) -> ConversationItem | None:
    """Run one approved call end to end. Every tool call and delegate call goes through this.

    In order, the steps are: claim the call, fire `on_tool_start`, guard its input, invoke it,
    guard its output, record it, then fire `on_tool_end`. Only the invoke step differs: a tool
    runs its function, and a `Delegate` runs a nested run of this one. A delegate that pauses for
    approval leaves the call unanswered (`None`), like a call waiting on a human. Resuming runs
    the call again with its `suspended` record and it continues where it stopped: claimed and
    started before the pause, it isn't claimed or started again, and it ends once.

    `args` are the call's arguments as the gate already parsed them; the raw JSON is kept only
    for the span and the `ToolCall`, which record what the model wrote.
    """
    tool = _gated(callee)
    call_id = id_of(call)
    started = suspended is not None
    if not started:
        run.claim(call_id, tool.name)

    args_json = call_arguments(call)
    with run.span(
        tool.name, "delegate" if isinstance(callee, Delegate) else "tool", input=args_json
    ) as span:
        if not started:
            await run.hooks.on_tool_start(run.run_context, run.current_agent, tool)
            await run.guard(Phase.TOOL_INPUT, args, tool=tool, under=span)
        if isinstance(callee, Delegate):
            outcome = await _delegated(run, callee, call_id, args, args_json, span, suspended)
        else:
            outcome = await _invoked(run, tool, call_id, args, args_json, span)
        if outcome is None:
            span.output = "paused for approval"
            return None
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
    suspended: RunRecord | None,
) -> tuple[Any, ToolCall] | None:
    """Invoke `delegate` as a nested run under `span`: its output and the record of the call.

    The record's `children` are the calls the nested run made. A delegate that errored fails the
    call, and one that paused for approval has no output yet (`None`).
    """
    input = str(args.get("input", ""))
    nested = await run.delegate(delegate, input, call_id, under=span, suspended=suspended)
    if nested is None:
        return None
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


type _Thunk = Callable[[], Awaitable[ConversationItem | None]]


@dataclass
class _Slot:
    """One tool call's place in the message's answers, in the message's call order.

    A call that was rejected, unknown, malformed or reused from a pause has its `item` from the
    start; an approved one is given one once it has run, unless it is a delegate that paused, and
    a handoff once the message settles with nothing left waiting. A call waiting on a human never
    gets a slot.
    """

    call_id: str
    item: ConversationItem | None = None


type _Answered = tuple[list[ConversationItem], Agent | None]
"""One message's answers, in its call order, and the agent handed off to."""


@dataclass
class _Answers:
    """One message's tool calls on their way to answers: each one admitted, then all settled.

    Calls are admitted one by one, in order, then the approved ones run concurrently, unless the
    agent's `model_settings.parallel_tool_calls` is `False`. Answers keep the message's call
    order either way.

    Where the calls stand is the record's own, written as each one settles (see
    `RunRecord.standings`): an answer or a wait on a human here, a paused delegate's run by the
    delegate itself. A message the run pauses on is so already recorded as it stands, and the
    message a resumed run paused on starts from there.
    """

    run: _Run
    slots: list[_Slot] = field(default_factory=list)
    deferred: list[tuple[_Slot, _Thunk]] = field(default_factory=list)
    handoffs: list[tuple[_Slot, Handoff]] = field(default_factory=list)

    def answer(self, item: ConversationItem) -> None:
        """Answer the next call with `item` as it is."""
        slot = _Slot(answered_call_id(item))
        self.slots.append(slot)
        self._give(slot, item)

    def _give(self, slot: _Slot, item: ConversationItem) -> None:
        """Answer `slot`'s call with `item`, written to the record as that call's standing."""
        slot.item = item
        self.run.record.answer(slot.call_id, item)

    def defer(self, call_id: str, thunk: _Thunk) -> None:
        """Answer the next call, `call_id`, by running `thunk` once every call is admitted."""
        slot = _Slot(call_id)
        self.slots.append(slot)
        self.deferred.append((slot, thunk))

    async def admit(self, call: dict[str, Any]) -> None:
        """Gate `call`, and answer it, defer it, or leave it waiting on a human.

        A call a pause settled isn't gated again: its result is reused, never run again, and a
        delegate's that paused continues that delegate's run, admitted and started before the
        pause. A tool's call and a delegate's are otherwise gated alike: one no tool answers, or
        whose arguments don't parse, fails; any other is decided by the run's approval ledger
        (where one that waited on a human finds its decision), then run by `_run_call`, rejected,
        or left waiting. A handoff is the one call that isn't gated, because it is a switch rather
        than a call: it is only made once the message settles (see `settled`).
        """
        run, shape = self.run, self.run.shape
        name, call_id = call_name(call), id_of(call)
        suspended = run.record.suspended(call_id)
        if suspended is not None:
            args = parsed_arguments(call_arguments(call))
            delegate = shape.delegates[name]
            self.defer(call_id, partial(_run_call, run, delegate, call, args, suspended))
            return
        earlier = run.record.answered(call_id)
        if earlier is not None:
            self.answer(earlier)
            return

        handoff = shape.handoffs.get(name)
        if handoff is not None:
            slot = _Slot(call_id)
            self.slots.append(slot)
            self.handoffs.append((slot, handoff))
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
            run.record.wait(
                Interruption(name=name, arguments=args_json, call_id=call_id, agent=shape.agent)
            )
        elif gate.action == "reject":
            self.answer(tool_result(call_id, gate.message))
        else:
            self.defer(call_id, partial(_run_call, run, callee, call, args))

    async def settled(self) -> _Answered | None:
        """Run the deferred calls, then the handoffs: the answers and the agent switched to.

        `None` if a call still waits on a human, a paused delegate's included, after the calls
        the message waits on directly: the run pauses on the message, its record already showing
        where every call stands. A handoff is made only past that point, after the message's other
        calls ran, so a pause leaves it unmade, and resuming makes it once. Making it fires
        `on_handoff`, not the tool hooks, and it is not one of `run.tool_calls`.
        """
        run, record = self.run, self.run.record
        parallel = run.shape.model_settings.parallel_tool_calls is not False
        outcomes = await _execute([thunk for _, thunk in self.deferred], parallel)
        for (slot, _), item in zip(self.deferred, outcomes, strict=True):
            if item is not None:
                self._give(slot, item)
        if record.pending:
            return None
        switched = None
        for slot, handoff in self.handoffs:
            with run.span(handoff.tool_name, "handoff"):
                await run.hooks.on_handoff(run.run_context, run.current_agent, handoff.agent)
            slot.item = tool_result(slot.call_id, f"Transferred to {handoff.agent.name}.")
            switched = handoff.agent
        record.settle_message()
        return [slot.item for slot in self.slots if slot.item is not None], switched


async def _run_message_tool_calls(run: _Run, message: dict[str, Any]) -> _Answered | None:
    """Answer every tool call in `message` as `_Answers` admits them; `None` if the run pauses.

    The message a resumed run paused on goes through here too, its calls continuing from where
    the pause left them.
    """
    answers = _Answers(run)
    for call in tool_calls(message):
        await answers.admit(call)
    return await answers.settled()


@dataclass
class _TurnOutcome:
    """How the turn loop came out: an answer, or a pause on calls waiting on a human.

    Only what the loop couldn't record on the run as it went; the generated items, which agent
    ended up answering and what a pause waits on are read off the run itself.
    """

    final_output: Any
    paused: bool = False
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
    "_run_call",
    "_run_message_tool_calls",
]
