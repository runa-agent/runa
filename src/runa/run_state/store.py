"""run_state/store.py: `PausedRunStore`, where a server keeps the runs it paused.

A paused `RunState` carries its approval ledger, and the ledger is what stops a call id from
running twice and records who approved what. Handed to a client, that record is the client's to
replay or edit. Kept here, a client holds only a `run_id`, and resuming one is a `claim`: an
atomic delete, so of two requests resuming the same pause exactly one gets it.

One interface, two adapters: `run_state/sqlite.py` and `run_state/postgres.py`. Which one a
caller gets is `runa.db.paused_runs()`'s decision. Nothing in this module touches a database: it
holds the table both adapters create and the contract they answer.

A row is written once and never updated: a resumed run that pauses again is a new row with a new
id. A pause nobody resumes stays until an operator prunes it by `created_at`.
"""

from typing import Any, Protocol

from runa.db.schema import Column, Table

PAUSED_RUNS = Table(
    "paused_runs",
    columns=(
        Column("run_id", "text", primary_key=True),
        Column("agent", "text"),
        Column("state", "text"),
        Column("created_at", "timestamp", default_now=True),
    ),
)


class PausedRunStore(Protocol):
    """What a server needs to hold a paused run for a client: save it, read it, claim it.

    Every method is scoped to the agent the run started with, so a pause is only ever found again
    through that agent's route. `state` is `RunState.to_json()`'s blob, stored as is.
    """

    async def save(self, agent: str, state: dict[str, Any]) -> str:
        """Keep `state`, paused on `agent`, and return the new `run_id` it is found by."""
        ...

    async def load(self, agent: str, run_id: str) -> dict[str, Any] | None:
        """The state saved under `run_id` for `agent`; `None` if there is none or it was claimed."""
        ...

    async def claim(self, agent: str, run_id: str) -> bool:
        """Take `run_id` off the store to resume it; `False` if it was already gone.

        Atomic: of any number of concurrent claims on one id, exactly one returns `True`.
        """
        ...


__all__ = ["PAUSED_RUNS", "PausedRunStore"]
