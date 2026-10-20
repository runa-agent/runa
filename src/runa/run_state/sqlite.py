"""run_state/sqlite.py: `SQLitePausedRunStore`, the local `PausedRunStore`.

`db/runa.db`'s `paused_runs` table, in the same connect-and-create-if-missing file every other
local adapter writes to (`db/sqlite.py`). `runa.db.paused_runs()` returns this whenever the
deployment's state is local.
"""

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from runa.db import sqlite_database
from runa.db.sqlite import run, run_read
from runa.run_state.store import PAUSED_RUNS


class SQLitePausedRunStore:
    """Persistent `PausedRunStore`: the `paused_runs` table inside `db/runa.db`."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Store where this table lives, defaulting to `db.sqlite_database()`.

        The table is created on the first `save`.
        """
        self.db_path = Path(db_path) if db_path is not None else sqlite_database()

    async def save(self, agent: str, state: dict[str, Any]) -> str:
        """Keep `state`, paused on `agent`, and return the new `run_id` it is found by."""
        run_id = uuid.uuid4().hex
        encoded = json.dumps(state)

        def body(conn: sqlite3.Connection) -> None:
            conn.execute(
                f"INSERT INTO {PAUSED_RUNS.name} (run_id, agent, state) VALUES (?, ?, ?)",
                (run_id, agent, encoded),
            )
            conn.commit()

        await run(self.db_path, (PAUSED_RUNS,), body)
        return run_id

    async def load(self, agent: str, run_id: str) -> dict[str, Any] | None:
        """The state saved under `run_id` for `agent`; `None` if there is none or it was claimed."""

        def body(conn: sqlite3.Connection | None) -> str | None:
            if conn is None:
                return None
            row = conn.execute(
                f"SELECT state FROM {PAUSED_RUNS.name} WHERE run_id = ? AND agent = ?",
                (run_id, agent),
            ).fetchone()
            return row["state"] if row is not None else None

        state = await run_read(self.db_path, (PAUSED_RUNS,), body)
        return json.loads(state) if state is not None else None

    async def claim(self, agent: str, run_id: str) -> bool:
        """Take `run_id` off the store to resume it; `False` if it was already gone."""

        def body(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                f"DELETE FROM {PAUSED_RUNS.name} WHERE run_id = ? AND agent = ?", (run_id, agent)
            )
            conn.commit()
            return cursor.rowcount == 1

        return await run(self.db_path, (PAUSED_RUNS,), body)


__all__ = ["SQLitePausedRunStore"]
