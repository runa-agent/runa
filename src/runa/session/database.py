"""session/database.py: the session stores, both sides of them, over a `Database`.

`DatabaseSession` is the write side a run appends to, `DatabaseSessionStore` the read side `runa
sessions` and `runa ui` query. The SQL is written once, against `db/handle.py`'s `Database`, so a
local file and a shared Postgres answer the same queries; `runa.db.session(...)` and
`runa.db.sessions()` pick which one it is.

The two tables are declared once in `session/store.py` and rendered per dialect by
`db/schema.py`; this module's own code is the query shapes, nothing about the schema.
"""

import json

from runa._items import ConversationItem
from runa.db.handle import Database, Queries
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


class DatabaseSession(Session):
    """Conversation history for one `session_id`, persisted to a `Database`."""

    def __init__(self, session_id: str, database: Database, agent: str | None = None) -> None:
        """Store `session_id`, the database its history lives in and the agent it belongs to.

        The first write that carries an `agent` records it, so `SessionStore.listing` can find
        the session by agent whatever its id.
        """
        self.session_id = session_id
        self.agent_name = agent
        self.database = database

    async def get_items(self) -> list[ConversationItem]:
        """Return this session's items, oldest first."""
        rows = await self.database.fetch(
            f"SELECT message_data FROM {MESSAGES.name} WHERE session_id = $1 ORDER BY id",
            self.session_id,
        )
        return [json.loads(row["message_data"]) for row in rows]

    async def add_items(self, items: list[ConversationItem]) -> None:
        """Append `items`, creating the session row on first write."""
        if not items:
            return
        async with self.database.transaction() as tx:
            await self._append(tx, items)
            await self._touch(tx)

    async def set_items(self, items: list[ConversationItem]) -> None:
        """Replace this session's entire history with `items`, in one transaction."""
        async with self.database.transaction() as tx:
            await tx.execute(f"DELETE FROM {MESSAGES.name} WHERE session_id = $1", self.session_id)
            if items:
                await self._append(tx, items)
            await self._touch(tx)

    async def clear_session(self) -> None:
        """Delete this session's history and its `agent_sessions` row, in one transaction."""
        async with self.database.transaction() as tx:
            await tx.execute(f"DELETE FROM {MESSAGES.name} WHERE session_id = $1", self.session_id)
            await tx.execute(f"DELETE FROM {SESSIONS.name} WHERE session_id = $1", self.session_id)

    async def _append(self, tx: Queries, items: list[ConversationItem]) -> None:
        await tx.execute(
            f"INSERT INTO {SESSIONS.name} (session_id) VALUES ($1) "
            "ON CONFLICT (session_id) DO NOTHING",
            self.session_id,
        )
        await tx.executemany(
            f"INSERT INTO {MESSAGES.name} (session_id, message_data) VALUES ($1, $2)",
            [(self.session_id, json.dumps(item)) for item in items],
        )

    async def _touch(self, tx: Queries) -> None:
        await tx.execute(
            f"UPDATE {SESSIONS.name} SET updated_at = CURRENT_TIMESTAMP, "
            "agent_name = COALESCE(agent_name, $2) WHERE session_id = $1",
            self.session_id,
            self.agent_name,
        )


class DatabaseSessionStore:
    """The `SessionStore` over a `Database`: the read side of its session tables.

    Same ordering and the same timestamp rendering whichever database answers, both of which come
    from `session/store.py`, so a local deployment and a shared one list the same sessions in the
    same order. A listing on a project that has never run an agent finds no tables rather than
    creating them: `runa chat --list` doesn't leave a `db/runa.db` behind as a side effect of
    answering "no sessions found".
    """

    def __init__(self, database: Database) -> None:
        """Store the database this history lives in."""
        self.database = database

    async def listing(self, *, agent: str | None = None) -> list[SessionSummary]:
        """Return this database's sessions, most recently updated first."""
        clause, params = listing_clause(agent)
        rows = await self.database.fetch(
            f"SELECT session_id, updated_at FROM {SESSIONS.name} {clause}", *params
        )
        return [
            SessionSummary(id=row["session_id"], updated_at=as_timestamp(row["updated_at"]))
            for row in rows
        ]

    async def messages(self, session_id: str) -> list[SessionMessage]:
        """Return `session_id`'s messages, oldest first."""
        exists = await self.database.fetchval(
            f"SELECT 1 FROM {SESSIONS.name} WHERE session_id = $1", session_id
        )
        if exists is None:
            raise SessionNotFound(f"no session found with id {session_id!r}")
        rows = await self.database.fetch(
            f"SELECT created_at, message_data FROM {MESSAGES.name} WHERE session_id = $1 "
            "ORDER BY id",
            session_id,
        )
        return [to_message(row["created_at"], row["message_data"]) for row in rows]


__all__ = ["DatabaseSession", "DatabaseSessionStore"]
