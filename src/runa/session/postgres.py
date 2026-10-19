"""session/postgres.py: the shared session backend, both sides of it.

The `runa-ai[postgres]` extra, not a core dependency. The same two tables `session/store.py`
declares and the same query shapes as `session/sqlite.py`, minus the single-process assumption:
every process on the same URL sees the same history, since Postgres (unlike a bare
`sqlite3.connect`) tolerates concurrent writers without corrupting the file. `PostgresSession`
is the write side a run appends to, `PostgresSessionStore` the read side `runa sessions` and
`runa ui` query.

`runa.db.session(...)`/`runa.db.sessions(...)` build these whenever `RUNA_DATABASE_URL` is a
`postgresql://` one, so no call site has to name them. Constructing one by hand is the escape
hatch for a database that is not this deployment's shared one.
"""

import json

from runa._items import ConversationItem
from runa.db.pool import Shared, sync
from runa.db.schema import POSTGRES
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


class PostgresSession(Shared, Session):
    """`Session` backed by Postgres, for a deployment sharing history across processes.

    `Shared` comes first so `super().__init__` reaches the one that takes `(url, *tables)`; the
    identity that matters to a caller is still `Session`, which this is the write side of.
    """

    def __init__(self, session_id: str, url: str) -> None:
        """Store `session_id` and which Postgres database its history lives in."""
        self.session_id = session_id
        super().__init__(url, SESSIONS, MESSAGES)

    async def get_items(self) -> list[ConversationItem]:
        """Return this session's items, oldest first."""
        pool = await self._pool()
        rows = await pool.fetch(
            f"SELECT message_data FROM {MESSAGES.name} WHERE session_id = $1 ORDER BY id",
            self.session_id,
        )
        return [json.loads(row["message_data"]) for row in rows]

    async def add_items(self, items: list[ConversationItem]) -> None:
        """Append `items`, creating the session row on first write."""
        if not items:
            return
        pool = await self._pool()
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(
                f"INSERT INTO {SESSIONS.name} (session_id) VALUES ($1) "
                "ON CONFLICT (session_id) DO NOTHING",
                self.session_id,
            )
            await conn.executemany(
                f"INSERT INTO {MESSAGES.name} (session_id, message_data) VALUES ($1, $2)",
                [(self.session_id, json.dumps(item)) for item in items],
            )
            await conn.execute(
                f"UPDATE {SESSIONS.name} SET updated_at = now() WHERE session_id = $1",
                self.session_id,
            )

    async def set_items(self, items: list[ConversationItem]) -> None:
        """Replace this session's entire history with `items`, in one transaction."""
        pool = await self._pool()
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(
                f"DELETE FROM {MESSAGES.name} WHERE session_id = $1", self.session_id
            )
            if items:
                await conn.execute(
                    f"INSERT INTO {SESSIONS.name} (session_id) VALUES ($1) "
                    "ON CONFLICT (session_id) DO NOTHING",
                    self.session_id,
                )
                await conn.executemany(
                    f"INSERT INTO {MESSAGES.name} (session_id, message_data) VALUES ($1, $2)",
                    [(self.session_id, json.dumps(item)) for item in items],
                )
            await conn.execute(
                f"UPDATE {SESSIONS.name} SET updated_at = now() WHERE session_id = $1",
                self.session_id,
            )

    async def clear_session(self) -> None:
        """Delete this session's history and its `agent_sessions` row, in one transaction."""
        pool = await self._pool()
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(
                f"DELETE FROM {MESSAGES.name} WHERE session_id = $1", self.session_id
            )
            await conn.execute(
                f"DELETE FROM {SESSIONS.name} WHERE session_id = $1", self.session_id
            )


class PostgresSessionStore(Shared):
    """The shared `SessionStore`: the read side of this deployment's session tables.

    Same ordering and the same timestamp rendering as `session/sqlite.py`, both of which come
    from `session/store.py` rather than from here, so a local deployment and a shared one list
    the same sessions in the same order with the same timestamps.
    """

    def __init__(self, url: str) -> None:
        """Store which Postgres database this history lives in; connected lazily."""
        super().__init__(url, SESSIONS, MESSAGES)

    @sync
    async def listing(self, *, agent: str | None = None) -> list[SessionSummary]:
        """Return this database's sessions, most recently updated first."""
        pool = await self._pool()
        clause, params = listing_clause(agent, POSTGRES)
        rows = await pool.fetch(
            f"SELECT session_id, updated_at FROM {SESSIONS.name} {clause}", *params
        )
        return [
            SessionSummary(id=row["session_id"], updated_at=as_timestamp(row["updated_at"]))
            for row in rows
        ]

    @sync
    async def messages(self, session_id: str) -> list[SessionMessage]:
        """Return `session_id`'s messages, oldest first."""
        pool = await self._pool()
        exists = await pool.fetchval(
            f"SELECT 1 FROM {SESSIONS.name} WHERE session_id = $1", session_id
        )
        if exists is None:
            raise SessionNotFound(f"no session found with id {session_id!r}")
        rows = await pool.fetch(
            f"SELECT created_at, message_data FROM {MESSAGES.name} WHERE session_id = $1 "
            "ORDER BY id",
            session_id,
        )
        return [to_message(row["created_at"], row["message_data"]) for row in rows]


__all__ = [
    "PostgresSession",
    "PostgresSessionStore",
]
