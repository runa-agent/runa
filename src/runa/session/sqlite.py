"""session/sqlite.py: the local session backend, both sides of it.

`db/runa.db`'s `agent_sessions`/`agent_messages` tables, in the same
connect-and-create-if-missing file every other local adapter writes to (`db/sqlite.py`).
`SQLiteSession` is the write side a run appends to, `SQLiteSessionStore` the read side
`runa sessions` and `runa ui` query. Same file, same two tables, so they live together.

`SQLiteSession` is a thin synchronous wrapper: the local backend is one process by definition, so
it skips the thread-local connections, WAL mode, and cross-process file locking a
multi-process-safe store would need. That store is `session/postgres.py`, and `runa.db` picks it
when the environment says to.

The two tables are declared once in `session/store.py` and rendered here by `db/schema.py`;
this module's own code is the SQLite query shapes, nothing about the schema.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from runa._types import TResponseInputItem
from runa.db import sqlite_path
from runa.db.schema import SQLITE, ddl
from runa.db.sqlite import connect as _connect_db
from runa.session import SessionABC
from runa.session.store import (
    MESSAGES,
    SESSIONS,
    SessionMessage,
    SessionNotFound,
    SessionSummary,
    agent_filter,
    as_timestamp,
    to_message,
)

DDL = ddl(SQLITE, SESSIONS, MESSAGES)


class SQLiteSession(SessionABC):
    """Conversation history for one `session_id`, persisted to the local `db/runa.db`."""

    def __init__(
        self,
        session_id: str,
        db_path: str | Path | None = None,
        *,
        user_id: str | None = None,
    ) -> None:
        """Store `session_id` and where in `runa.db` its history lives.

        `db_path` defaults to `runa.db.sqlite_path()`, so this history lands in the same local
        file as every other concern's -- including when `RUNA_DATABASE_URL` relocates it. Pass
        one to point at some other file.

        `user_id` scopes this session's automatic memory, if its agent has any; see `SessionABC`.
        """
        self.session_id = session_id
        self.db_path = Path(db_path) if db_path is not None else sqlite_path()
        self.user_id = user_id

    def _connect(self) -> sqlite3.Connection:
        return _connect_db(self.db_path, DDL)

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        """Return this session's items, oldest first, capped at the latest `limit` if given."""
        with closing(self._connect()) as conn:
            if limit is None:
                rows = conn.execute(
                    f"SELECT message_data FROM {MESSAGES.name} WHERE session_id = ? ORDER BY id",
                    (self.session_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""
                    SELECT message_data FROM {MESSAGES.name} WHERE session_id = ?
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
                f"INSERT OR IGNORE INTO {SESSIONS.name} (session_id) VALUES (?)",
                (self.session_id,),
            )
            conn.executemany(
                f"INSERT INTO {MESSAGES.name} (session_id, message_data) VALUES (?, ?)",
                [(self.session_id, json.dumps(item)) for item in items],
            )
            conn.execute(
                f"UPDATE {SESSIONS.name} SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (self.session_id,),
            )
            conn.commit()

    async def set_items(self, items: list[TResponseInputItem]) -> None:
        """Replace this session's entire history with `items`, in one transaction."""
        with closing(self._connect()) as conn:
            conn.execute(f"DELETE FROM {MESSAGES.name} WHERE session_id = ?", (self.session_id,))
            if items:
                conn.execute(
                    f"INSERT OR IGNORE INTO {SESSIONS.name} (session_id) VALUES (?)",
                    (self.session_id,),
                )
                conn.executemany(
                    f"INSERT INTO {MESSAGES.name} (session_id, message_data) VALUES (?, ?)",
                    [(self.session_id, json.dumps(item)) for item in items],
                )
            conn.execute(
                f"UPDATE {SESSIONS.name} SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (self.session_id,),
            )
            conn.commit()

    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return this session's most recent item, or `None` if it has none."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"""
                DELETE FROM {MESSAGES.name}
                WHERE id = (
                    SELECT id FROM {MESSAGES.name} WHERE session_id = ? ORDER BY id DESC LIMIT 1
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
            conn.execute(f"DELETE FROM {MESSAGES.name} WHERE session_id = ?", (self.session_id,))
            conn.execute(f"DELETE FROM {SESSIONS.name} WHERE session_id = ?", (self.session_id,))
            conn.commit()


class SQLiteSessionStore:
    """The local `SessionStore`: the read side of `db/runa.db`'s session tables.

    Opens the file read-only in spirit: a listing on a project that has never run an agent finds
    no tables rather than creating them, so `runa chat --list` doesn't leave a `db/runa.db`
    behind as a side effect of answering "no sessions found".
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Store which SQLite file this history lives in, defaulting to `db.sqlite_path()`."""
        self.db_path = Path(db_path) if db_path is not None else sqlite_path()

    def listing(self, *, agent: str | None = None) -> list[SessionSummary]:
        """Return this file's sessions, most recently updated first."""
        where, params = agent_filter(agent, SQLITE)
        rows = self._query(
            f"SELECT session_id, updated_at FROM {SESSIONS.name} {where}"
            "ORDER BY updated_at DESC, session_id DESC",
            params,
        )
        return [
            SessionSummary(id=row["session_id"], updated_at=as_timestamp(row["updated_at"]))
            for row in rows
        ]

    def messages(self, session_id: str) -> list[SessionMessage]:
        """Return `session_id`'s messages, oldest first."""
        if not self._query(f"SELECT 1 FROM {SESSIONS.name} WHERE session_id = ?", (session_id,)):
            raise SessionNotFound(f"no session found with id {session_id!r}")
        rows = self._query(
            f"SELECT created_at, message_data FROM {MESSAGES.name} WHERE session_id = ? "
            "ORDER BY id",
            (session_id,),
        )
        return [to_message(row["created_at"], row["message_data"]) for row in rows]

    def _query(self, sql: str, params: tuple[object, ...]) -> list[sqlite3.Row]:
        """Run `sql`, treating a file or table that isn't there yet as no rows.

        A deployment's first read can legitimately come before its first write, and "this app has
        no history" is the answer then, not an error about a missing table.
        """
        if not self.db_path.exists():
            return []
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            try:
                return conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                return []


__all__ = [
    "DDL",
    "SQLiteSession",
    "SQLiteSessionStore",
]
