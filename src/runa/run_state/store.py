"""run_state/store.py: `PausedRunStore`, where every paused run is kept until it resumes.

A paused `RunState` carries its approval ledger, the record of who approved what. Kept here, a
caller holds only a `run_id`, and resuming one is a `claim`: an atomic delete, so of two
resumes of the same pause exactly one gets it. `RunState` is the only caller.

One interface, one implementation over `db/handle.py`'s `Database`: `run_state/database.py`.
Which database it queries is `runa.db.paused_runs()`'s decision. Nothing in this module touches a
database: it holds the table and the contract.

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
    """What a paused run needs kept until it resumes: save it, read it, claim it.

    Every method is scoped to the agent the run started with, so a pause is only ever found again
    through that agent. `state` is `RunState`'s own blob, stored as is.
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
