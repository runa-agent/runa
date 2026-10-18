"""paused_delegates.py: where a paused delegate waits, and whose decision resolves it.

A `.delegate` subagent that pauses for approval is set aside here under the id of the tool call
that started it; its caller pauses on the delegate's interruptions, and resuming the caller
re-enters the delegate where it stopped. Each interruption is owned by the state whose run will
execute that call, and its decision is recorded there.

Imports nothing from `runa` at runtime: `runa._governance` imports it.
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
    `owners` maps a pending call's id to the `waiting` key whose run will execute it.
    """

    waiting: dict[str, RunState] = field(default_factory=dict)
    owners: dict[str, str] = field(default_factory=dict)

    def stash(self, call_id: str, state: RunState) -> None:
        """Set a paused delegate aside under `call_id`, claiming the interruptions it brought up.

        A nested pause surfaces the same `Interruption` at every depth, and the innermost run
        stashes first, so the earliest claim on a call id (the deepest delegate's) is kept.
        """
        self.waiting[call_id] = state
        for interruption in state.pending:
            self.owners.setdefault(interruption.call_id, call_id)

    def take(self, call_id: str) -> RunState | None:
        """Take the delegate stashed under `call_id` back out to resume it, `None` if there's none.

        Its claims go with it, so a second pause records ownership afresh.
        """
        state = self.waiting.pop(call_id, None)
        if state is not None:
            self.owners = {
                pending_id: owner for pending_id, owner in self.owners.items() if owner != call_id
            }
        return state

    def owner_of(self, interruption: Interruption) -> RunState | None:
        """The paused delegate whose decision `interruption` is, or `None` if it's the caller's own.

        `RunState.approve`/`.reject` record the decision on that state rather than their own.
        """
        owner = self.owners.get(interruption.call_id)
        return self.waiting.get(owner) if owner is not None else None

    def restore(self, waiting: Mapping[str, RunState], owners: Mapping[str, str]) -> None:
        """Refill a stash deserialized by `RunState.from_json`, ownership included.

        Each state's context forks the one holding this stash, so the stash exists before them.
        """
        self.waiting.update(waiting)
        self.owners.update(owners)


__all__ = ["PausedDelegates"]
