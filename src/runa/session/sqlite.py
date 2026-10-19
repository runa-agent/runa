"""session/sqlite.py: the local session backend, both sides of it.

`db/runa.db`'s `agent_sessions`/`agent_messages` tables, in the same
connect-and-create-if-missing file every other local adapter writes to (`db/sqlite.py`).
`SQLiteSession` is the write side a run appends to, `SQLiteSessionStore` the read side
`runa sessions` and `runa ui` query. Same file, same two tables, so they live together.

`SQLiteSession` is a thin wrapper over `sqlite3`: the local backend is one process by definition,
so it skips the thread-local connections, WAL mode, and cross-process file locking a
multi-process-safe store would need. That store is `session/postgres.py`, and `runa.db` picks it
when the environment says to.

Thin, but not on the event loop: `sqlite3` is blocking, and `Session`'s methods are `async def`
because a run awaits them. `_off_loop` below is where the two meet.

The two tables are declared once in `session/store.py` and rendered here by `db/schema.py`;
this module's own code is the SQLite query shapes, nothing about the schema.
"""

import asyncio
import functools
import json
import sqlite3
from collections.abc import Callable, Coroutine
from contextlib import closing
from pathlib import Path
from typing import Any

from runa._items import ConversationItem
from runa.db import sqlite_database
from runa.db.schema import SQLITE
from runa.db.sqlite import connect as _connect_db
from runa.db.sqlite import read as _read_db
from runa.session import Session
from runa.session.store import (
    MESSAGES,
    SESSIONS,
    SessionMessage,
    SessionNotFound,
    SessionSummary,
    as_timestamp,
    listing_clause,
    to_message,
)


def _off_loop[**P, T](method: Callable[P, T]) -> Callable[P, Coroutine[Any, Any, T]]:
    """Turn one synchronous `sqlite3` method into the `async def` `Session` declares.

    Off the event loop, like a sync `@tool` and a sync `@guardrail`: these bodies are file I/O, a
    session-backed turn runs three of them (a read, then a read and a write to save the turn), and
    `runa serve`'s default deployment is this backend on the loop that is also serving every other
    request. Awaiting something that never yields is how one turn's `fsync` becomes every concurrent
    turn's latency.

    A thread hop costs a few hundred microseconds against a local file's millisecond, and it is
    what makes the `await` honest. What it does not buy is parallel writes: SQLite has one writer
    per file, so concurrent turns now wait on its lock (`sqlite3`'s five-second busy timeout)
    instead of on the loop -- the waiting moves off the thread that has other work to do, which is
    the whole point. A deployment that needs genuine write concurrency is on `session/postgres.py`.
    """

    @functools.wraps(method)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return await asyncio.to_thread(method, *args, **kwargs)

    return wrapper


class SQLiteSession(Session):
    """Conversation history for one `session_id`, persisted to the local `db/runa.db`.

    Every method here is `_off_loop`: a plain synchronous body, handed to a worker thread to
    satisfy `Session`'s `async def`. Each opens its own short-lived connection, so no `sqlite3`
    object is ever touched from two threads.
    """

    def __init__(
        self,
        session_id: str,
        db_path: str | Path | None = None,
        *,
        user_id: str | None = None,
    ) -> None:
        """Store `session_id` and where in `runa.db` its history lives.

        `db_path` defaults to `runa.db.sqlite_database()`, so this history lands in the same local
        database as every other concern's -- including when `RUNA_DATABASE_URL` relocates it or
        keeps it in memory. Pass one to point at some other file.

        `user_id` scopes the memory of a run on this session, if its agent has any; see `Session`.
        """
        self.session_id = session_id
        self.db_path = Path(db_path) if db_path is not None else sqlite_database()
        self.user_id = user_id

    def _connect(self) -> sqlite3.Connection:
        return _connect_db(self.db_path, SESSIONS, MESSAGES)

    @_off_loop
    def get_items(self) -> list[ConversationItem]:
        """Return this session's items, oldest first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT message_data FROM {MESSAGES.name} WHERE session_id = ? ORDER BY id",
                (self.session_id,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    @_off_loop
    def add_items(self, items: list[ConversationItem]) -> None:
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

    @_off_loop
    def set_items(self, items: list[ConversationItem]) -> None:
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

    @_off_loop
    def clear_session(self) -> None:
        """Delete this session's history and its `agent_sessions` row, in one transaction."""
        with closing(self._connect()) as conn:
            conn.execute(f"DELETE FROM {MESSAGES.name} WHERE session_id = ?", (self.session_id,))
            conn.execute(f"DELETE FROM {SESSIONS.name} WHERE session_id = ?", (self.session_id,))
            conn.commit()


class SQLiteSessionStore:
    """The local `SessionStore`: the read side of `db/runa.db`'s session tables.

    Queries through `db/sqlite.py`'s `read`, like every other local read side, so a listing on a
    project that has never run an agent finds no tables rather than creating them: `runa chat
    --list` doesn't leave a `db/runa.db` behind as a side effect of answering "no sessions found".
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Store which SQLite database this history lives in: `db.sqlite_database()` by default."""
        self.db_path = Path(db_path) if db_path is not None else sqlite_database()

    def listing(self, *, agent: str | None = None) -> list[SessionSummary]:
        """Return this file's sessions, most recently updated first."""
        clause, params = listing_clause(agent, SQLITE)
        rows = self._query(f"SELECT session_id, updated_at FROM {SESSIONS.name} {clause}", params)
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
        no history" is the answer then, not an error about a missing table. That absence is the
        only thing answered with no rows, and `read` tests it; every other way a query can fail
        raises, here as in the three read sides next door.
        """
        with _read_db(self.db_path, SESSIONS, MESSAGES) as conn:
            return conn.execute(sql, params).fetchall() if conn is not None else []


__all__ = [
    "SQLiteSession",
    "SQLiteSessionStore",
]
