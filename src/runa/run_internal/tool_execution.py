"""tool_execution.py: answering one model message, its tool calls, delegates and handoff.

What happens to each call of a model message lives here, a pause included: which calls were
answered, which wait on a human and which delegates paused, written to the run's record as each
one settles, so a pause on the message finds it recorded and resuming answers that message again
from there. This is the only module that writes a call's standing. Once the message is
answered, its answers enter the run's transcript, and which agent it hands off to, if any, is
returned for the loop to switch to; each call's claim and approval, against the run tree's
ledger, and the record a delegate's run starts from are decided here too. These steps are
passed the loop's run, imported for its type only, and use only what a call's gate needs from
it and its record (guard a call, open a span, fire a hook, run a record nested in it; record a
call, an answer, a wait or a suspension). They assign none of its fields.
"""

import asyncio
from collections.abc import Awaitable, Callable
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
from runa._types import RunContext
from runa.approval import ApprovalDecision
from runa.exceptions import DuplicateToolCallError, ToolError
from runa.guardrail import Phase
from runa.handoff import Delegate, Handoff
from runa.run_internal.record import Interruption, RunRecord
from runa.stream_events import ToolOutput
from runa.tool import FunctionTool, Tool, ToolCall
from runa.tracing.spans import Span

if TYPE_CHECKING:
    from runa.agent import Agent
    from runa.run import Run
    from runa.run_internal.run_loop import _Run


type _Outcome = tuple[Any, ToolCall] | None
"""What invoking a call came to: its result and the record of it, or `None` while it's paused."""


@dataclass(frozen=True)
class Answered:
    """A model message whose calls are all answered, and the agent it handed off to, if any."""

    handoff: Agent | None = None


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


def _callee(target: FunctionTool | Delegate) -> _Callee:
    """`target`, a tool or a delegate, as the gate runs it."""
    if isinstance(target, Delegate):
        return _Callee(target.tool, "delegate", partial(_delegate_invoked, target))
    return _Callee(target, "tool", partial(_invoked, target))


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
class _Answers:
    """One message's tool calls on their way to answers: each one admitted, then all settled.

    Calls are admitted one by one, in order, then the approved ones run concurrently, unless the
    agent's `model_settings.parallel_tool_calls` is `False`. Answers keep the message's call
    order either way.

    Where the calls stand is the record's own, written as each one settles (see
    `RunRecord.standings`): an answer, a wait on a human or a paused delegate's run. The record
    is the one store of the message's answers; only a handoff's, which never has a standing, is
    held here until the message settles. A message the run pauses on is so already recorded as
    it stands, and the message a resumed run paused on starts from there.
    """

    run: _Run
    message: dict[str, Any]
    deferred: list[tuple[str, _Thunk]] = field(default_factory=list)
    handoffs: list[tuple[str, Handoff]] = field(default_factory=list)

    def answer(self, call_id: str, item: ConversationItem) -> None:
        """Answer call `call_id` with `item`, written to the record as its standing."""
        self.run.record.answer(call_id, item)

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
        target = shape.callees.get(name)
        suspended = run.record.suspended(call_id)
        if suspended is not None and isinstance(target, Delegate):
            self.deferred.append((call_id, partial(_resume_call, run, target, call, suspended)))
            return
        if run.record.answered(call_id) is not None:
            return

        if isinstance(target, Handoff):
            self.handoffs.append((call_id, target))
            return

        args_json = call_arguments(call)
        if target is None:
            failed = _failed(name, args_json, f"unknown tool {name}")
            self.answer(call_id, _answer(run, call_id, failed))
            return
        args = _arguments_or_error(args_json)
        if isinstance(args, str):
            self.answer(call_id, _answer(run, call_id, _failed(name, args_json, args)))
            return

        callee = _callee(target)
        gate = await _decide(run, callee.tool, call_id, args)
        if gate.action == "interrupt":
            run.record.wait(
                Interruption(name=name, arguments=args_json, call_id=call_id, agent=shape.agent)
            )
        elif gate.action == "reject":
            self.answer(call_id, tool_result(call_id, gate.message))
        else:
            self.deferred.append((call_id, partial(_run_call, run, callee, call, args)))

    async def settled(self) -> Answered | None:
        """Run the deferred calls, then pick the handoff, and fold the answers into the run.

        `None` if a call still waits on a human, a paused delegate's included, after the calls
        the message waits on directly: the run pauses on the message, its record already showing
        where every call stands, and its transcript without the message's answers. Otherwise the
        answers go into the transcript in call order, each one emitted, and the agent handed off
        to, if any, is returned for the run to switch to.

        A handoff is picked only past the pause, after the message's other calls ran, so a pause
        leaves it unpicked, and resuming picks it once. A message hands off once, to its first
        handoff: a later one is answered with an error instead. Picking it fires `on_handoff`,
        not the tool hooks, and it is not one of `run.tool_calls`.
        """
        run, record = self.run, self.run.record
        parallel = run.shape.model_settings.parallel_tool_calls is not False
        outcomes = await _execute([thunk for _, thunk in self.deferred], parallel)
        for (call_id, _), item in zip(self.deferred, outcomes, strict=True):
            if item is not None:
                self.answer(call_id, item)
        if record.pending:
            return None
        switched: Agent | None = None
        handed: dict[str, ConversationItem] = {}
        for call_id, handoff in self.handoffs:
            if switched is not None:
                error = f"error: this message already handed off to {switched.name}"
                handed[call_id] = tool_result(call_id, error)
                continue
            with run.span(handoff.tool_name, "handoff"):
                await run.hooks.on_handoff(run.run_context, run.current_agent, handoff.agent)
            handed[call_id] = tool_result(call_id, f"Transferred to {handoff.agent.name}.")
            switched = handoff.agent

        results = [
            item
            for call_id in map(id_of, tool_calls(self.message))
            if (item := handed.get(call_id) or record.answered(call_id)) is not None
        ]
        record.settle_message()
        run.transcript.record(*results)
        for result in results:
            run.notify(ToolOutput(result))
        return Answered(switched)


async def _answer_message(run: _Run, message: dict[str, Any]) -> Answered | None:
    """Answer every tool call in `message` and fold the answers into the run (see `_Answers`).

    The one way tool results enter a run, resumed or fresh: the message a resumed run paused on
    goes through here too, its calls continuing from where the pause left them. `Answered` once
    the message is, `None` if the run pauses on it.
    """
    answers = _Answers(run, message)
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
    "Answered",
    "_answer_message",
    "_run_call",
]
