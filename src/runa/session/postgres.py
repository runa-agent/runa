"""session/postgres.py: `PostgresSession`, the shared `SessionABC`.

The `runa-ai[postgres]` extra, not a core dependency. Same tables and query shapes as
`session/sqlite.py`, minus the single-process assumption: every process on the same URL sees the
same history, since Postgres (unlike a bare `sqlite3.connect`) tolerates concurrent writers
without corrupting the file.

`runa.db.session(...)` builds this whenever `RUNA_DATABASE_URL` is a `postgresql://` one, so no
call site has to name it. Constructing it by hand is the escape hatch for a database that is not
this deployment's shared one.
"""

import json
from typing import Any

import asyncpg

from runa._types import TResponseInputItem
from runa.db.pool import connect as _connect
from runa.db.pool import run_sync
from runa.session import SessionABC

SESSIONS_TABLE = "agent_sessions"
MESSAGES_TABLE = "agent_messages"

DDL = f"""
CREATE TABLE IF NOT EXISTS {SESSIONS_TABLE} (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
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


async def _rows(url: str, where: str, *params: Any) -> list[tuple[str, str]]:
    """`(session_id, updated_at)` rows matching `where`, timestamps as ISO text.

    `session/storage.py` compares and renders these as strings, the way SQLite already hands them
    over, so a `TIMESTAMPTZ` is formatted here rather than leaking a `datetime` into one backend's
    results and not the other's.
    """
    pool = await _connect(url, DDL)
    rows = await pool.fetch(f"SELECT session_id, updated_at FROM {SESSIONS_TABLE} {where}", *params)
    return [
        (row["session_id"], row["updated_at"].isoformat(sep=" ", timespec="seconds"))
        for row in rows
    ]


def session_rows(url: str) -> list[tuple[str, str]]:
    """Return `(session_id, updated_at)` for every session in `url`, oldest first."""
    return run_sync(_rows(url, "ORDER BY updated_at"))


def sessions_for_agent(url: str, agent_name: str, pattern: str) -> list[tuple[str, str]]:
    """Return `(session_id, updated_at)` for `agent_name`'s past sessions, newest first."""
    return run_sync(
        _rows(
            url,
            "WHERE session_id = $1 OR session_id LIKE $2 ORDER BY updated_at DESC, session_id DESC",
            agent_name,
            pattern,
        )
    )


async def _messages(url: str, session_id: str) -> list[tuple[str, str]]:
    pool = await _connect(url, DDL)
    exists = await pool.fetchval(
        f"SELECT 1 FROM {SESSIONS_TABLE} WHERE session_id = $1", session_id
    )
    if exists is None:
        from runa.session.storage import SessionNotFound

        raise SessionNotFound(f"no session found with id {session_id!r}")
    rows = await pool.fetch(
        f"SELECT created_at, message_data FROM {MESSAGES_TABLE} WHERE session_id = $1 ORDER BY id",
        session_id,
    )
    return [
        (row["created_at"].isoformat(sep=" ", timespec="seconds"), row["message_data"])
        for row in rows
    ]


def session_messages(url: str, session_id: str) -> list[tuple[str, str]]:
    """Return `session_id`'s `(created_at, message_data)` rows from `url`, oldest first."""
    return run_sync(_messages(url, session_id))


__all__ = [
    "DDL",
    "MESSAGES_TABLE",
    "SESSIONS_TABLE",
    "PostgresSession",
    "session_messages",
    "session_rows",
    "sessions_for_agent",
]
