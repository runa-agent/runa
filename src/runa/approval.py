"""approval.py: the approval protocol -- `@approval`, and the ledger that decides each call.

Approval is the most intricate thing the runtime does: a per-call decision, a sticky per-tool one
that outlives it, a custom rejection message, a replay guard so an executed call never runs twice,
and all of it durable across a process restart. The rules live here, in one object, rather than
spread across the turn loop and `RunState`:

* sticky beats the tool's own `needs_approval` predicate, which beats the per-call decision;
* a rejection with no message of its own gets `ApprovalLedger.DEFAULT_REJECTION`;
* a call id is claimed before it runs, and a claimed id never runs again.

`RunContextWrapper` holds one `ApprovalLedger`, shared by reference with every delegate it
forks (`RunContextWrapper.fork`), which is what makes a caller's "always" answer cover the
delegate's matching tool. `run_internal.tool_execution` asks it `decide`/`claim`; `RunState`
tells it `record` when an operator answers "always". Where a *paused delegate* is stashed, which
of a run's interruptions are really its, and how both survive a restart belong to
`runa.paused_delegates.PausedDelegates`, held on the context the same way this is.

This module imports nothing from `runa` at runtime, on purpose: `_types` imports it to put a
ledger on the context, and `runa.exceptions` imports `_types`. `runa.paused_delegates` is the
other half of pausing a run, kept out of here for the same reason: a delegate's pause is about
which *run* resolves a call, not which decision applies to it.
"""

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from runa._types import RunContextWrapper

_NeedsApproval = Callable[["RunContextWrapper[Any]", dict[str, Any], str], Awaitable[bool]]


def approval(func: Callable[..., bool | Awaitable[bool]]) -> _NeedsApproval:
    """Turn a predicate into a `@tool(needs_approval=...)` callable.

    Write the predicate like the tool it guards: parameters are looked up by name from the
    tool call's parsed arguments. Name a parameter `ctx` or `call_id` to receive the run
    context or call id instead of a tool argument. Works on `def`, `async def`, and `lambda`.
    """
    names = list(inspect.signature(func).parameters)

    async def wrapper(ctx: RunContextWrapper[Any], params: dict[str, Any], call_id: str) -> bool:
        reserved = {"ctx": ctx, "call_id": call_id}
        bound = {name: reserved[name] if name in reserved else params[name] for name in names}
        result = func(**bound)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)

    return wrapper


@dataclass
class ApprovalDecision:
    """What the ledger decided for one tool call: run it, reject it, or pause the run.

    `message` is what the model is told in place of the tool's result, and is set only for
    `"reject"` -- never `None` there, since a rejection with no message of its own falls back to
    `ApprovalLedger.DEFAULT_REJECTION`.
    """

    action: Literal["run", "reject", "interrupt"]
    message: str | None = None


@dataclass
class ApprovalLedger:
    """Every approval decision a run has recorded, and the rules for reading them back.

    `sticky`/`sticky_messages` are the "always approve"/"always reject" decisions, keyed by tool
    *name*: once recorded they cover every later call to that tool, in this run or a delegate's.
    `executed` is the set of call ids that have already run, so resuming a stale `RunState`
    cannot run a tool a second time.

    Per-*call* decisions aren't held here: they belong to the `RunState` the operator resolved
    and are handed to `decide` as `approvals`/`rejection_messages`. Sticky decisions outlive any
    one pause; per-call ones don't.
    """

    DEFAULT_REJECTION: ClassVar[str] = "rejected by the operator"
    """What the model is told when a call was refused and nobody said why."""

    sticky: dict[str, bool] = field(default_factory=dict)
    sticky_messages: dict[str, str] = field(default_factory=dict)
    executed: set[str] = field(default_factory=set)

    async def decide(
        self,
        tool_name: str,
        call_id: str,
        *,
        needs_approval: Callable[[], Awaitable[bool]],
        approvals: Mapping[str, bool] | None = None,
        rejection_messages: Mapping[str, str] | None = None,
    ) -> ApprovalDecision:
        """Decide whether this call runs, is rejected, or pauses the run for a human.

        In precedence order: a sticky decision for `tool_name` settles it outright; otherwise the
        tool's own `needs_approval` predicate is consulted, and a call that doesn't need approval
        runs; otherwise the per-call decision for `call_id` applies, and its absence is the pause.

        `needs_approval` is a callable rather than the tool itself so this stays the whole rule
        without the ledger having to know what a `FunctionTool` is.
        """
        sticky = self.sticky.get(tool_name)
        if sticky is True:
            return ApprovalDecision("run")
        if sticky is False:
            return self._rejection(self.sticky_messages.get(tool_name))
        if not await needs_approval():
            return ApprovalDecision("run")
        verdict = (approvals or {}).get(call_id)
        if verdict is None:
            return ApprovalDecision("interrupt")
        if verdict is False:
            return self._rejection((rejection_messages or {}).get(call_id))
        return ApprovalDecision("run")

    def record(self, tool_name: str, *, approved: bool, message: str | None = None) -> None:
        """Record a sticky decision for `tool_name`, as `RunState.approve`/`.reject(always=True)`.

        The decision owns its message slot: `message` replaces whatever was there, and `None`
        clears it, so a tool stickied twice can't answer with the earlier decision's text.
        """
        self.sticky[tool_name] = approved
        if message is None:
            self.sticky_messages.pop(tool_name, None)
        else:
            self.sticky_messages[tool_name] = message

    def claim(self, call_id: str) -> bool:
        """Claim `call_id` for execution; `False` if it already ran and must not run again.

        The caller turns that into a `DuplicateToolCallError` -- the ledger owns the rule, the
        runtime owns its vocabulary of errors (see this module's docstring for why).
        """
        if call_id in self.executed:
            return False
        self.executed.add(call_id)
        return True

    def release(self, call_id: str) -> None:
        """Give a claim back: the call didn't finish, so resuming may legitimately run it again."""
        self.executed.discard(call_id)

    def _rejection(self, message: str | None) -> ApprovalDecision:
        """A `"reject"` decision carrying `message`, or the default text when there isn't one."""
        return ApprovalDecision("reject", message or self.DEFAULT_REJECTION)


__all__ = ["ApprovalDecision", "ApprovalLedger", "approval"]
