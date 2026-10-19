"""run_state/postgres.py: `PostgresPausedRunStore`, the shared `PausedRunStore`.

The `runa-ai[postgres]` extra, not a core dependency. The same `paused_runs` table as
`run_state/sqlite.py`, in the database every replica already shares, so a run paused on one
replica resumes on any other.
"""

import json
import uuid
from typing import Any

from runa.db.pool import Shared
from runa.run_state.store import PAUSED_RUNS


class PostgresPausedRunStore(Shared):
    """Persistent `PausedRunStore`: the `paused_runs` table in this deployment's shared Postgres."""

    def __init__(self, url: str) -> None:
        """Store which Postgres database this table lives in; connected lazily."""
        super().__init__(url, PAUSED_RUNS)

    async def save(self, agent: str, state: dict[str, Any]) -> str:
        """Keep `state`, paused on `agent`, and return the new `run_id` it is found by."""
        run_id = uuid.uuid4().hex
        pool = await self._pool()
        await pool.execute(
            f"INSERT INTO {PAUSED_RUNS.name} (run_id, agent, state) VALUES ($1, $2, $3)",
            run_id,
            agent,
            json.dumps(state),
        )
        return run_id

    async def load(self, agent: str, run_id: str) -> dict[str, Any] | None:
        """The state saved under `run_id` for `agent`; `None` if there is none or it was claimed."""
        pool = await self._pool()
        state = await pool.fetchval(
            f"SELECT state FROM {PAUSED_RUNS.name} WHERE run_id = $1 AND agent = $2", run_id, agent
        )
        return json.loads(state) if state is not None else None

    async def claim(self, agent: str, run_id: str) -> bool:
        """Take `run_id` off the store to resume it; `False` if it was already gone."""
        pool = await self._pool()
        status = await pool.execute(
            f"DELETE FROM {PAUSED_RUNS.name} WHERE run_id = $1 AND agent = $2", run_id, agent
        )
        return bool(status == "DELETE 1")


__all__ = ["PostgresPausedRunStore"]
