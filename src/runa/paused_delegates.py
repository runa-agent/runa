"""paused_delegates.py: where a paused delegate waits, and whose decision resolves it.

A `.delegate` subagent runs as a tool (`runa.handoff.agent_as_tool`), so when its nested run
pauses for approval there is nobody to ask halfway down a call stack: the nested `RunState` is
set aside here, the caller's run pauses on the *delegate's* interruptions, and resuming the
caller re-enters the delegate where it stopped instead of starting it over. The rules live here,
in one object, rather than spread across `handoff`, the turn loop and `RunState`:

* a paused delegate waits under the id of the delegate tool call that started it;
* every interruption it brought up is owned by that delegate's state, recorded as it is stashed;
* an owned interruption's decision belongs to its owner alone -- the state whose run will
  actually execute that call -- so the caller records only the calls it will run itself;
* taking a delegate back out drops its ownership with it, so a second pause records afresh.

Ownership is recorded at stash time because that is the one moment it's known for free. A
delegate of a delegate shares this stash (a delegate runs on its caller's governance, so
every depth lands in the same place), and the innermost run is the first to finish, so it is
also the first to stash: `stash` keeps the earliest claim on a call id, which is the deepest
one. That's the call stack unwinding, not a convention two modules have to agree on separately.

`runa.handoff` stashes and takes back; `RunState` asks `owner_of` whose decision a resolved
interruption is, and carries the whole stash through `to_json`/`from_json` via `restore`.

This module imports nothing from `runa` at runtime, on purpose, for the same reason
`runa.approval` doesn't: `_governance` imports it to put one beside every run's context.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runa.run_state import Interruption, RunState


@dataclass
class PausedDelegates:
    """The delegate runs a run paused on, and which pending call each of them owns.

    `waiting` maps a delegate tool call's id to the nested `RunState` that call paused on.
    `owners` maps a *pending* call's id to the `waiting` key that owns it -- the delegate whose
    own run will execute that call once it's approved. Both are plain data, as
    `ApprovalLedger`'s are: the rules for reading them back are the methods below.
    """

    waiting: dict[str, RunState] = field(default_factory=dict)
    owners: dict[str, str] = field(default_factory=dict)

    def stash(self, call_id: str, state: RunState) -> None:
        """Set a paused delegate aside under `call_id`, claiming the interruptions it brought up.

        An interruption already claimed by a deeper delegate stays with it (see this module's
        docstring on why the deepest is the first to get here): a nested pause surfaces the same
        `Interruption` object at every level, and only the run that will execute the call owns it.
        """
        self.waiting[call_id] = state
        for interruption in state.pending:
            self.owners.setdefault(interruption.call_id, call_id)

    def take(self, call_id: str) -> RunState | None:
        """Take the delegate stashed under `call_id` back out to resume it, `None` if there's none.

        Its claims go with it: the delegate is about to run again, and a second pause stashes a
        fresh state with fresh ownership rather than inheriting the last one's.
        """
        state = self.waiting.pop(call_id, None)
        if state is not None:
            self.owners = {
                pending_id: owner for pending_id, owner in self.owners.items() if owner != call_id
            }
        return state

    def owner_of(self, interruption: Interruption) -> RunState | None:
        """The paused delegate whose decision `interruption` is, or `None` if it's the caller's own.

        `RunState.approve`/`.reject` record on this state instead of themselves, which is what
        makes resuming the caller resume the delegate with its approvals already in hand.
        """
        owner = self.owners.get(interruption.call_id)
        return self.waiting.get(owner) if owner is not None else None

    def restore(self, waiting: Mapping[str, RunState], owners: Mapping[str, str]) -> None:
        """Refill a stash deserialized by `RunState.from_json`, ownership included.

        The states can't be passed to `PausedDelegates(...)` the way `ApprovalLedger`'s fields
        are: each one's context is a fork of the context holding *this* stash, so the stash has
        to exist before there is anything to put in it.
        """
        self.waiting.update(waiting)
        self.owners.update(owners)


__all__ = ["PausedDelegates"]
