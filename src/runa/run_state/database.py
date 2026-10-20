"""run_state/database.py: `DatabasePausedRunStore`, the `PausedRunStore` over a `Database`.

The `paused_runs` table in whichever database the deployment has, so on a shared one a run paused
on one replica resumes on any other. `runa.db.paused_runs()` returns this.
"""

import json
import uuid
from typing import Any

from runa.db.handle import Database
from runa.run_state.store import PAUSED_RUNS


class DatabasePausedRunStore:
    """Persistent `PausedRunStore`: the `paused_runs` table of a `Database`."""

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
        """Take `run_id` off the store to resume it; `False` if it was already gone."""
        deleted = await self.database.execute(
            f"DELETE FROM {PAUSED_RUNS.name} WHERE run_id = $1 AND agent = $2", run_id, agent
        )
        return deleted == 1


__all__ = ["DatabasePausedRunStore"]
