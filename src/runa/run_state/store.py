"""run_state/store.py: `PausedRunStore`, where every paused run is kept until it resumes.

A paused `RunState` carries its approval ledger, the record of who approved what. Kept here, a
caller holds only a `run_id`, and resuming one is a `claim`: an atomic delete, so of two
resumes of the same pause exactly one gets it. `RunState` is the only caller.

The `paused_runs` table in whichever database the deployment has, queried through `db/handle.py`'s
`Database`, so on a shared one a run paused on one replica resumes on any other. Which database it
is is `runa.db.paused_runs()`'s decision.

A row is written once and never updated: a resumed run that pauses again is a new row with a new
id. A pause nobody resumes stays until an operator prunes it by `created_at`.
"""

import json
import uuid
from typing import Any

from runa.db.handle import Database
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


class PausedRunStore:
    """What a paused run needs kept until it resumes: save it, read it, claim it.

    Every method is scoped to the agent the run started with, so a pause is only ever found again
    through that agent. `state` is `RunState`'s own blob, stored as is.
    """

    def __init__(self, database: Database) -> None:
        """Store the database this table lives in; the table is created on the first `save`."""
        self.database = database

    async def save(self, agent: str, state: dict[str, Any]) -> str:
        """Keep `state`, paused on `agent`, and return the new `run_id` it is found by."""
        run_id = uuid.uuid4().hex
        await self.database.execute(
            f"INSERT INTO {PAUSED_RUNS.name} (run_id, agent, state) VALUES ($1, $2, $3)",
            run_id,
            agent,
            json.dumps(state),
        )
        return run_id

    async def load(self, agent: str, run_id: str) -> dict[str, Any] | None:
        """The state saved under `run_id` for `agent`; `None` if there is none or it was claimed."""
        state = await self.database.fetchval(
            f"SELECT state FROM {PAUSED_RUNS.name} WHERE run_id = $1 AND agent = $2", run_id, agent
        )
        return json.loads(state) if state is not None else None

    async def claim(self, agent: str, run_id: str) -> bool:
        """Take `run_id` off the store to resume it; `False` if it was already gone.

        Atomic: of any number of concurrent claims on one id, exactly one returns `True`.
        """
        deleted = await self.database.execute(
            f"DELETE FROM {PAUSED_RUNS.name} WHERE run_id = $1 AND agent = $2", run_id, agent
        )
        return deleted == 1


__all__ = ["PAUSED_RUNS", "PausedRunStore"]
