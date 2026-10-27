"""tool_execution.py: answering one model message, its tool calls, delegates and handoff.

What happens to each call of a model message lives here, a pause included: which calls were
answered, which wait on a human and which delegates paused, written to the run's record as each
one settles, so a pause on the message finds it recorded and resuming answers that message again
from there. This is the only module that writes a call's standing. Once the message is
answered, its answers enter the run's transcript and its handoff, if any, is made, so which
agent runs next is decided here too, as is each call's claim and approval, against the run
tree's ledger, and the record a delegate's run starts from. These steps are passed the loop's
run, imported for its type only, and use what it and its record can do (guard a call, fire a
hook, run a record nested in it, list its tools; record a call, an answer, a wait or a
suspension). The one field assigned is the record's `agent`, by a handoff.
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
from runa._types import RunContext
from runa.approval import ApprovalDecision
from runa.exceptions import DuplicateToolCallError, ToolError
from runa.guardrail import Phase
from runa.handoff import Delegate, Handoff
from runa.run_internal.record import Interruption, RunRecord
from runa.stream_events import AgentChanged, ToolOutput
from runa.tool import FunctionTool, Tool, ToolCall
from runa.tracing.spans import Span

if TYPE_CHECKING:
    from runa.agent import Agent
    from runa.run import Run
    from runa.run_internal.agent_shape import AgentShape
    from runa.run_internal.run_loop import _Run


type _Outcome = tuple[Any, ToolCall] | None
"""What invoking a call came to: its result and the record of it, or `None` while it's paused."""


@dataclass(frozen=True)
class _Callee:
    """What a call's name resolves to, so the gate runs every callee alike without asking which.

    `tool` is what the call is approved, guarded and handed to hooks as, `kind` its span's type,
    and `invoke` the gate's one varying step: given the run, the call, its parsed arguments and
    its span, what invoking it came to.
    """

    tool: Tool
    kind: str
    invoke: Callable[[_Run, dict[str, Any], dict[str, Any], Span], Awaitable[_Outcome]]


def _callee(shape: AgentShape, name: str) -> _Callee | None:
    """The tool or delegate the model calls `name`, or `None` when neither answers to it."""
    delegate = shape.delegates.get(name)
    if delegate is not None:
        return _Callee(delegate.tool, "delegate", partial(_delegate_invoked, delegate))
    tool = shape.tools_by_name.get(name)
    return None if tool is None else _Callee(tool, "tool", partial(_invoked, tool))


async def _run_call(
    run: _Run, callee: _Callee, call: dict[str, Any], args: dict[str, Any]
) -> ConversationItem | None:
    """Run one approved call through the gate. Every tool call and delegate call goes through this.

    In order, the steps are: claim the call, fire `on_tool_start`, guard its input, invoke it,
    guard its output, record it, then fire `on_tool_end`. This function opens the call, up to
    the invoke step, and `_closed` takes it from there. Only the invoke step differs: a tool runs
    its function, and a `Delegate` runs a nested run of this one. A delegate that pauses for
    approval leaves the call unanswered (`None`), like a call waiting on a human, and is
    continued by `_resume_call`.

    `args` are the call's arguments as the gate already parsed them; the raw JSON is kept only
    for the span and the `ToolCall`, which record what the model wrote.
    """
    tool = callee.tool
    _claim(run, id_of(call), tool.name)

    async def invoked(span: Span) -> _Outcome:
        await run.hooks.on_tool_start(run.run_context, run.current_agent, tool)
        await run.guard(Phase.TOOL_INPUT, args, tool=tool, under=span)
        return await callee.invoke(run, call, args, span)

    return await _closed(run, tool, callee.kind, call, invoked)


async def _resume_call(
    run: _Run, delegate: Delegate, call: dict[str, Any], suspended: RunRecord
) -> ConversationItem | None:
    """Continue delegate call `call`, whose run paused as `suspended`, from its invoke step.

    `_run_call` opened it before the pause, so it isn't claimed, started or guarded again; it
    closes once, through `_closed`, like any call.
    """

    async def continued(span: Span) -> _Outcome:
        return await _delegate_run(run, delegate, suspended.resumed(), None, call, span)

    return await _closed(run, delegate.tool, "delegate", call, continued)


async def _closed(
    run: _Run,
    tool: Tool,
    kind: str,
    call: dict[str, Any],
    invoke: Callable[[Span], Awaitable[_Outcome]],
) -> ConversationItem | None:
    """`invoke` the call in its span, then close it: guard its output, record it, `on_tool_end`.

    `tool` and `kind` are what `_Callee` says they are. The second half of the gate `_run_call`
    states. A call `invoke` leaves paused (`None`) isn't closed, and is left unanswered.
    """
    with run.span(tool.name, kind, input=call_arguments(call)) as span:
        outcome = await invoke(span)
        if outcome is None:
            span.output = "paused for approval"
            return None
        result, recorded = outcome
        await run.guard(Phase.TOOL_OUTPUT, result, tool=tool, under=span)
        span.output = result
    item = _answer(run, id_of(call), recorded)
    await run.hooks.on_tool_end(run.run_context, run.current_agent, tool, result)
    return item


def _claim(run: _Run, call_id: str, name: str) -> None:
    """Mark `call_id` as executing, raising `DuplicateToolCallError` if it already ran."""
    if not run.tree.approvals.claim(call_id):
        raise DuplicateToolCallError(call_id, name)


async def _decide(run: _Run, tool: Tool, call_id: str, args: dict[str, Any]) -> ApprovalDecision:
    """Whether this call to `tool` runs, is rejected, or waits on a human."""
    needs_approval = partial(tool.needs_approval, run.run_context, args, call_id)
    return await run.tree.approvals.decide(tool.name, call_id, needs_approval=needs_approval)


async def _invoked(
    tool: FunctionTool, run: _Run, call: dict[str, Any], args: dict[str, Any], span: Span
) -> _Outcome:
    """Invoke `tool`: its result, and the record of the call. A tool that raises fails the call."""
    call_id, args_json = id_of(call), call_arguments(call)
    try:
        result = await tool.on_invoke_tool(run.run_context, args, call_id)
    except Exception as exc:  # noqa: BLE001 -- a tool failing is data, not a run-ending error
        failed = _failed(tool.name, args_json, str(exc) or type(exc).__name__, exc, span)
        return failed.output, failed
    return result, ToolCall(tool.name, args_json, str(result))


async def _delegate_invoked(
    delegate: Delegate, run: _Run, call: dict[str, Any], args: dict[str, Any], span: Span
) -> _Outcome:
    """Invoke `delegate`: a nested run of this one, given the call's `input`.

    The delegate's run has a record of its own, started on its agent, over the same `context`
    and `user_id` as this run.
    """
    context = run.run_context
    record = RunRecord(
        start=delegate.agent,
        agent=delegate.agent,
        run_context=RunContext(context=context.context, user_id=context.user_id),
    )
    given = str(args.get("input", ""))
    return await _delegate_run(run, delegate, record, given, call, span)


async def _delegate_run(
    run: _Run,
    delegate: Delegate,
    record: RunRecord,
    given: str | None,
    call: dict[str, Any],
    span: Span,
) -> _Outcome:
    """Run `delegate`'s `record` nested in `run` (see its `nested`) as `call`: what it came to.

    A delegate's run that pauses for approval is `None`, its record suspended on `run`'s as the
    call's standing (see `RunRecord.suspend`), for `run` to pause on and `_resume_call` to
    continue. Otherwise its output and the record of the call, whose `children` are the calls
    the nested run made; a delegate that errored fails the call.
    """
    nested = await run.nested(record, given, under=span)
    if nested.status == "paused":
        run.record.suspend(id_of(call), record)
        return None
    return _delegated(delegate, nested, call_arguments(call), span)


def _delegated(delegate: Delegate, nested: Run, args_json: str, span: Span) -> _Outcome:
    """What `delegate`'s completed or errored `nested` run came to (see `_delegate_run`)."""
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
    run.record.record_call(call)
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


@dataclass
class _Answers:
    """One message's tool calls on their way to answers: each one admitted, then all settled.

    Calls are admitted one by one, in order, then the approved ones run concurrently, unless the
    agent's `model_settings.parallel_tool_calls` is `False`. Answers keep the message's call
    order either way.

    Where the calls stand is the record's own, written as each one settles (see
    `RunRecord.standings`): an answer, a wait on a human or a paused delegate's run. A message
    the run pauses on is so already recorded as it stands, and the message a resumed run paused
    on starts from there.
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
            delegate = shape.delegates[name]
            self.defer(call_id, partial(_resume_call, run, delegate, call, suspended))
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
        callee = _callee(shape, name)
        if callee is None:
            self.answer(_answer(run, call_id, _failed(name, args_json, f"unknown tool {name}")))
            return
        args = _arguments_or_error(args_json)
        if isinstance(args, str):
            self.answer(_answer(run, call_id, _failed(name, args_json, args)))
            return

        gate = await _decide(run, callee.tool, call_id, args)
        if gate.action == "interrupt":
            run.record.wait(
                Interruption(name=name, arguments=args_json, call_id=call_id, agent=shape.agent)
            )
        elif gate.action == "reject":
            self.answer(tool_result(call_id, gate.message))
        else:
            self.defer(call_id, partial(_run_call, run, callee, call, args))

    async def settled(self) -> bool:
        """Run the deferred calls, then make the handoff, and fold the answers into the run.

        `False` if a call still waits on a human, a paused delegate's included, after the calls
        the message waits on directly: the run pauses on the message, its record already showing
        where every call stands, and its transcript without the message's answers. Otherwise the
        answers go into the transcript in call order, each one emitted, and the run switches to
        the agent handed off to.

        A handoff is made only past the pause, after the message's other calls ran, so a pause
        leaves it unmade, and resuming makes it once. A message hands off once, to its first
        handoff: a later one is answered with an error instead, and never made. Making it fires
        `on_handoff`, not the tool hooks, and it is not one of `run.tool_calls`.
        """
        run, record = self.run, self.run.record
        parallel = run.shape.model_settings.parallel_tool_calls is not False
        outcomes = await _execute([thunk for _, thunk in self.deferred], parallel)
        for (slot, _), item in zip(self.deferred, outcomes, strict=True):
            if item is not None:
                self._give(slot, item)
        if record.pending:
            return False
        switched: Agent | None = None
        for slot, handoff in self.handoffs:
            if switched is not None:
                error = f"error: this message already handed off to {switched.name}"
                slot.item = tool_result(slot.call_id, error)
                continue
            with run.span(handoff.tool_name, "handoff"):
                await run.hooks.on_handoff(run.run_context, run.current_agent, handoff.agent)
            slot.item = tool_result(slot.call_id, f"Transferred to {handoff.agent.name}.")
            switched = handoff.agent
        record.settle_message()

        results = [slot.item for slot in self.slots if slot.item is not None]
        run.transcript.record(*results)
        for result in results:
            run.notify(ToolOutput(result))
        if switched is not None:
            record.agent = switched
            await run.list_tools()
            run.notify(AgentChanged(switched))
            await run.hooks.on_agent_start(run.run_context, run.current_agent)
        return True


async def _answer_message(run: _Run, message: dict[str, Any]) -> bool:
    """Answer every tool call in `message` and fold the answers into the run (see `_Answers`).

    The one way tool results enter a run, resumed or fresh: the message a resumed run paused on
    goes through here too, its calls continuing from where the pause left them. `True` once the
    message is answered, `False` if the run pauses on it.
    """
    answers = _Answers(run)
    for call in tool_calls(message):
        await answers.admit(call)
    return await answers.settled()


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
    "_answer_message",
    "_run_call",
]
