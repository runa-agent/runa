"""session/postgres.py: the shared session backend, both sides of it.

The `runa-ai[postgres]` extra, not a core dependency. Same tables and query shapes as
`session/sqlite.py`, minus the single-process assumption: every process on the same URL sees the
same history, since Postgres (unlike a bare `sqlite3.connect`) tolerates concurrent writers
without corrupting the file. `PostgresSession` is the write side a run appends to,
`PostgresSessionStore` the read side `runa sessions` and `runa ui` query.

`runa.db.session(...)`/`runa.db.sessions(...)` build these whenever `RUNA_DATABASE_URL` is a
`postgresql://` one, so no call site has to name them. Constructing one by hand is the escape
hatch for a database that is not this deployment's shared one.
"""

import json

import asyncpg

from runa._types import TResponseInputItem
from runa.db.pool import connect as _connect
from runa.db.pool import run_sync
from runa.session import SessionABC
from runa.session.store import (
    SessionMessage,
    SessionNotFound,
    SessionSummary,
    agent_pattern,
    as_timestamp,
    to_message,
)

SESSIONS_TABLE = "agent_sessions"
MESSAGES_TABLE = "agent_messages"

DDL = f"""
CREATE TABLE IF NOT EXISTS {SESSIONS_TABLE} (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_{SESSIONS_TABLE}_updated_at
    ON {SESSIONS_TABLE} (updated_at DESC, session_id DESC);
CREATE TABLE IF NOT EXISTS {MESSAGES_TABLE} (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES {SESSIONS_TABLE}(session_id) ON DELETE CASCADE,
    message_data TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_{MESSAGES_TABLE}_session_id ON {MESSAGES_TABLE} (session_id, id);
"""


class PostgresSession(SessionABC):
    """`SessionABC` backed by Postgres, for a deployment sharing history across processes."""

    def __init__(self, session_id: str, url: str, *, user_id: str | None = None) -> None:
        """Store `session_id` and which Postgres database its history lives in."""
        self.session_id = session_id
        self.url = url
        self.user_id = user_id

    async def _pool(self) -> asyncpg.Pool:
        return await _connect(self.url, DDL)

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        """Return this session's items, oldest first, capped at the latest `limit` if given."""
        pool = await self._pool()
        if limit is None:
            rows = await pool.fetch(
                f"SELECT message_data FROM {MESSAGES_TABLE} WHERE session_id = $1 ORDER BY id",
                self.session_id,
            )
        else:
            rows = await pool.fetch(
                f"""
                SELECT message_data FROM {MESSAGES_TABLE} WHERE session_id = $1
                ORDER BY id DESC LIMIT $2
                """,
                self.session_id,
                limit,
            )
            rows.reverse()
        return [json.loads(row["message_data"]) for row in rows]

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """Append `items`, creating the session row on first write."""
        if not items:
            return
        pool = await self._pool()
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(
                f"INSERT INTO {SESSIONS_TABLE} (session_id) VALUES ($1) "
                "ON CONFLICT (session_id) DO NOTHING",
                self.session_id,
            )
            await conn.executemany(
                f"INSERT INTO {MESSAGES_TABLE} (session_id, message_data) VALUES ($1, $2)",
                [(self.session_id, json.dumps(item)) for item in items],
            )
            await conn.execute(
                f"UPDATE {SESSIONS_TABLE} SET updated_at = now() WHERE session_id = $1",
                self.session_id,
            )

    async def set_items(self, items: list[TResponseInputItem]) -> None:
        """Replace this session's entire history with `items`, in one transaction."""
        pool = await self._pool()
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(
                f"DELETE FROM {MESSAGES_TABLE} WHERE session_id = $1", self.session_id
            )
            if items:
                await conn.execute(
                    f"INSERT INTO {SESSIONS_TABLE} (session_id) VALUES ($1) "
                    "ON CONFLICT (session_id) DO NOTHING",
                    self.session_id,
                )
                await conn.executemany(
                    f"INSERT INTO {MESSAGES_TABLE} (session_id, message_data) VALUES ($1, $2)",
                    [(self.session_id, json.dumps(item)) for item in items],
                )
            await conn.execute(
                f"UPDATE {SESSIONS_TABLE} SET updated_at = now() WHERE session_id = $1",
                self.session_id,
            )

    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return this session's most recent item, or `None` if it has none."""
        pool = await self._pool()
        row = await pool.fetchrow(
            f"""
            DELETE FROM {MESSAGES_TABLE}
            WHERE id = (
                SELECT id FROM {MESSAGES_TABLE} WHERE session_id = $1 ORDER BY id DESC LIMIT 1
            )
            RETURNING message_data
            """,
            self.session_id,
        )
        return json.loads(row["message_data"]) if row else None

    async def clear_session(self) -> None:
        """Delete this session and all of its items."""
        pool = await self._pool()
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(
                f"DELETE FROM {MESSAGES_TABLE} WHERE session_id = $1", self.session_id
            )
            await conn.execute(
                f"DELETE FROM {SESSIONS_TABLE} WHERE session_id = $1", self.session_id
            )


class PostgresSessionStore:
    """The shared `SessionStore`: the read side of this deployment's session tables.

    Same ordering and the same timestamp rendering as `session/sqlite.py`, both of which come
    from `session/store.py` rather than from here. They used to be this module's own: it broke
    ties on `session_id` where SQLite broke them on `rowid`, and rendered a `TIMESTAMPTZ` with its
    offset where SQLite's text had none, so one deployment's Sessions page and another's listed
    the same two sessions in a different order and printed their timestamps differently.
    """

    def __init__(self, url: str) -> None:
        """Store which Postgres database this history lives in; connected lazily."""
        self.url = url

    def listing(self, *, agent: str | None = None) -> list[SessionSummary]:
        """Return this database's sessions, most recently updated first."""
        return run_sync(self._listing(agent))

    def messages(self, session_id: str) -> list[SessionMessage]:
        """Return `session_id`'s messages, oldest first."""
        return run_sync(self._messages(session_id))

    async def _listing(self, agent: str | None) -> list[SessionSummary]:
        pool = await _connect(self.url, DDL)
        where = ""
        params: tuple[object, ...] = ()
        if agent is not None:
            where = "WHERE session_id = $1 OR session_id LIKE $2 "
            params = (agent, agent_pattern(agent))
        rows = await pool.fetch(
            f"SELECT session_id, updated_at FROM {SESSIONS_TABLE} {where}"
            "ORDER BY updated_at DESC, session_id DESC",
            *params,
        )
        return [
            SessionSummary(id=row["session_id"], updated_at=as_timestamp(row["updated_at"]))
            for row in rows
        ]

    async def _messages(self, session_id: str) -> list[SessionMessage]:
        pool = await _connect(self.url, DDL)
        exists = await pool.fetchval(
            f"SELECT 1 FROM {SESSIONS_TABLE} WHERE session_id = $1", session_id
        )
        if exists is None:
            raise SessionNotFound(f"no session found with id {session_id!r}")
        rows = await pool.fetch(
            f"SELECT created_at, message_data FROM {MESSAGES_TABLE} WHERE session_id = $1 "
            "ORDER BY id",
            session_id,
        )
        return [to_message(row["created_at"], row["message_data"]) for row in rows]


__all__ = [
    "DDL",
    "MESSAGES_TABLE",
    "SESSIONS_TABLE",
    "PostgresSession",
    "PostgresSessionStore",
]
