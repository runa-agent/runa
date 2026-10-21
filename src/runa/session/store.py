"""session/store.py: the `agent_sessions`/`agent_messages` tables, and both sides of them.

`DatabaseSession` is the write side a run appends to; `SessionStore` is the read side `runa
sessions`, `runa chat --list/--continue/--resume` and `runa ui`'s Sessions pages query, a
transcript view over whatever the first wrote. Both query through `db/handle.py`'s `Database`, so a
local file and a shared Postgres answer the same SQL; which one it is is `runa.db.session(...)`'s
and `runa.db.sessions()`'s decision, so no reader here or in `web/` branches on a backend or names
a file.

The two tables are declared here (`db/schema.py` renders them per dialect), beside the queries
that name their columns, the two objects a row becomes, and the rules a listing follows: which
agent a session belongs to and how it is ordered (`listing_clause`), and how a timestamp is
rendered. What a stored message's `role` and `text` are is `runa._items`'s rule, applied here.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime

from runa._items import ConversationItem, role_and_text
from runa.db.handle import Database, Queries
from runa.db.schema import Column, Index, Table
from runa.exceptions import OperatorError
from runa.session import Session

SESSIONS = Table(
    "agent_sessions",
    columns=(
        Column("session_id", "text", primary_key=True),
        Column("created_at", "timestamp", default_now=True),
        Column("updated_at", "timestamp", default_now=True),
        Column("agent_name", "text", null=True),
    ),
    indexes=(Index("updated_at", "updated_at DESC, session_id DESC"),),
)
"""One row per session. `agent_name` is the agent whose run first saved a turn to it.

Null until then: a session seeded with `add_items` before its first run belongs to no agent yet.
The first write that names an agent fills it and no later one changes it, so a session one agent
started stays that agent's after a handoff or a reuse of its id.
"""

MESSAGES = Table(
    "agent_messages",
    columns=(
        Column("id", "serial"),
        Column("session_id", "text", references=f"{SESSIONS.name}(session_id)"),
        Column("message_data", "text"),
        Column("created_at", "timestamp", default_now=True),
    ),
    indexes=(Index("session_id", "session_id, id"),),
)


class SessionNotFound(OperatorError):
    """Raised when a caller names a session id this deployment has no history for."""


@dataclass(frozen=True)
class SessionSummary:
    """One session in a listing: its id and when it was last written to."""

    id: str
    updated_at: str


@dataclass(frozen=True)
class SessionMessage:
    """One stored message, flattened to the `(role, text)` a transcript shows."""

    created_at: str
    role: str
    text: str


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


class SessionStore:
    """The read side of one deployment's conversation history.

    Two questions, which is every question `runa sessions`, `runa chat` and the Sessions pages
    ask: which sessions are there, and what was said in one of them. A listing on a project that
    has never run an agent finds no tables rather than creating them: `runa chat --list` doesn't
    leave a `db/runa.db` behind as a side effect of answering "no sessions found".
    """

    def __init__(self, database: Database) -> None:
        """Store the database this history lives in."""
        self.database = database

    async def listing(self, *, agent: str | None = None) -> list[SessionSummary]:
        """Return this deployment's sessions, most recently updated first.

        `agent` keeps only the sessions whose `agent_name` is `agent`, the agent whose run first
        saved a turn to it. A session id is the application's and says nothing about its agent.

        Ties on `updated_at` break by session id descending, whichever database answers, so `runa
        chat --list` and the Sessions page agree between a local deployment and a shared one.
        """
        clause, params = listing_clause(agent)
        rows = await self.database.fetch(
            f"SELECT session_id, updated_at FROM {SESSIONS.name} {clause}", *params
        )
        return [
            SessionSummary(id=row["session_id"], updated_at=_as_timestamp(row["updated_at"]))
            for row in rows
        ]

    async def messages(self, session_id: str) -> list[SessionMessage]:
        """Return `session_id`'s messages, oldest first.

        Raises `SessionNotFound` if this deployment has no session row for `session_id`, rather
        than returning an empty list: "no such session" and "a session with nothing in it" are
        different answers, and `runa chat --show` prints different things for them.
        """
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
        return [_to_message(row["created_at"], row["message_data"]) for row in rows]


_ORDER = "ORDER BY updated_at DESC, session_id DESC"


def listing_clause(agent: str | None) -> tuple[str, tuple[str, ...]]:
    """Everything after `SELECT ... FROM agent_sessions` for `listing`, and the values it binds.

    One sentence of policy: keep the sessions whose `agent_name` is `agent`, if one was named,
    most recently updated first, ties broken on the session id. `agent=None` filters nothing, so a
    listing composes the same string either way. The whole tail rather than the filter alone, so
    the tie-break is one rule rather than one per caller.
    """
    if agent is None:
        return _ORDER, ()
    return f"WHERE agent_name = $1 {_ORDER}", (agent,)


def _as_timestamp(value: str | datetime) -> str:
    """One rendering of a stored timestamp, whichever engine produced it.

    SQLite hands back `CURRENT_TIMESTAMP`'s `"YYYY-MM-DD HH:MM:SS"` text and Postgres a
    `TIMESTAMPTZ` -- naive on the first, offset-bearing on the second. Both go through here: one
    format, on one clock (UTC).
    """
    if isinstance(value, str):
        return value
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _to_message(created_at: str | datetime, message_data: str) -> SessionMessage:
    """One stored `(created_at, message_data)` row as a `SessionMessage`.

    What `role` and `text` mean for a stored item is `runa._items.role_and_text`'s to say, not
    this module's: a transcript reads an item, it doesn't define one.
    """
    role, text = role_and_text(json.loads(message_data))
    return SessionMessage(created_at=_as_timestamp(created_at), role=role, text=text)


__all__ = [
    "DatabaseSession",
    "SessionMessage",
    "SessionNotFound",
    "SessionStore",
    "SessionSummary",
    "listing_clause",
]
