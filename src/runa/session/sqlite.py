"""session/sqlite.py: `SQLiteSession`, the local `SessionABC`.

`db/runa.db`'s `agent_sessions`/`agent_messages` tables, in the same
connect-and-create-if-missing file every other local adapter writes to (`db/sqlite.py`).

A thin synchronous wrapper: the local backend is one process by definition, so this skips the
thread-local connections, WAL mode, and cross-process file locking a multi-process-safe store
would need. That store is `session/postgres.py`, and `runa.db` picks it when the environment
says to.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from runa._types import TResponseInputItem
from runa.db import DEFAULT_DB_PATH
from runa.db.sqlite import connect as _connect_db
from runa.session import SessionABC

SESSIONS_TABLE = "agent_sessions"
MESSAGES_TABLE = "agent_messages"

DDL = f"""
CREATE TABLE IF NOT EXISTS {SESSIONS_TABLE} (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS {MESSAGES_TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES {SESSIONS_TABLE}(session_id) ON DELETE CASCADE,
    message_data TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_{MESSAGES_TABLE}_session_id ON {MESSAGES_TABLE} (session_id, id);
"""


class SQLiteSession(SessionABC):
    """Conversation history for one `session_id`, persisted to the local `db/runa.db`."""

    def __init__(
        self,
        session_id: str,
        db_path: str | Path = DEFAULT_DB_PATH,
        *,
        user_id: str | None = None,
    ) -> None:
        """Store `session_id` and where in `runa.db` its history lives.

        `user_id` scopes this session's automatic memory, if its agent has any; see `SessionABC`.
        """
        self.session_id = session_id
        self.db_path = Path(db_path)
        self.user_id = user_id

    def _connect(self) -> sqlite3.Connection:
        return _connect_db(self.db_path, DDL)

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        """Return this session's items, oldest first, capped at the latest `limit` if given."""
        with closing(self._connect()) as conn:
            if limit is None:
                rows = conn.execute(
                    f"SELECT message_data FROM {MESSAGES_TABLE} WHERE session_id = ? ORDER BY id",
                    (self.session_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""
                    SELECT message_data FROM {MESSAGES_TABLE} WHERE session_id = ?
                    ORDER BY id DESC LIMIT ?
                    """,
                    (self.session_id, limit),
                ).fetchall()
                rows.reverse()
        return [json.loads(row[0]) for row in rows]

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """Append `items`, creating the session row on first write."""
        if not items:
            return
        with closing(self._connect()) as conn:
            conn.execute(
                f"INSERT OR IGNORE INTO {SESSIONS_TABLE} (session_id) VALUES (?)",
                (self.session_id,),
            )
            conn.executemany(
                f"INSERT INTO {MESSAGES_TABLE} (session_id, message_data) VALUES (?, ?)",
                [(self.session_id, json.dumps(item)) for item in items],
            )
            conn.execute(
                f"UPDATE {SESSIONS_TABLE} SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (self.session_id,),
            )
            conn.commit()

    async def set_items(self, items: list[TResponseInputItem]) -> None:
        """Replace this session's entire history with `items`, in one transaction."""
        with closing(self._connect()) as conn:
            conn.execute(f"DELETE FROM {MESSAGES_TABLE} WHERE session_id = ?", (self.session_id,))
            if items:
                conn.execute(
                    f"INSERT OR IGNORE INTO {SESSIONS_TABLE} (session_id) VALUES (?)",
                    (self.session_id,),
                )
                conn.executemany(
                    f"INSERT INTO {MESSAGES_TABLE} (session_id, message_data) VALUES (?, ?)",
                    [(self.session_id, json.dumps(item)) for item in items],
                )
            conn.execute(
                f"UPDATE {SESSIONS_TABLE} SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (self.session_id,),
            )
            conn.commit()

    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return this session's most recent item, or `None` if it has none."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"""
                DELETE FROM {MESSAGES_TABLE}
                WHERE id = (
                    SELECT id FROM {MESSAGES_TABLE} WHERE session_id = ? ORDER BY id DESC LIMIT 1
                )
                RETURNING message_data
                """,
                (self.session_id,),
            ).fetchone()
            conn.commit()
        return json.loads(row[0]) if row else None

    async def clear_session(self) -> None:
        """Delete this session and all of its items."""
        with closing(self._connect()) as conn:
            conn.execute(f"DELETE FROM {MESSAGES_TABLE} WHERE session_id = ?", (self.session_id,))
            conn.execute(f"DELETE FROM {SESSIONS_TABLE} WHERE session_id = ?", (self.session_id,))
            conn.commit()


__all__ = ["DDL", "MESSAGES_TABLE", "SESSIONS_TABLE", "SQLiteSession"]
