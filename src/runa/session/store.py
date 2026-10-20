"""session/store.py: `SessionStore`, the read side of `agent_sessions`/`agent_messages`.

What `runa sessions`, `runa chat --list/--continue/--resume` and `runa ui`'s Sessions pages read.
The write side is `Session` itself; this is the transcript view over whatever it wrote.

One interface, one implementation over `db/handle.py`'s `Database`: `session/database.py`.
Which database it queries is `runa.db.sessions(...)`'s decision, asked once, so no reader here or
in `web/` branches on a backend or names a file. Nothing in this module touches a database: it
holds the two tables (`db/schema.py` renders them per dialect), the contract, the two objects a
row becomes, and the rules a listing follows -- which agent a session belongs to, how it is
ordered, and how a timestamp is rendered. What a
stored message's `role` and `text` are is `runa._items`'s rule, applied here by `to_message`.

The first two are `listing_clause`, which is the whole reason it is here rather than in the store
that calls it.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from runa._items import role_and_text
from runa.db.schema import Column, Index, Table
from runa.exceptions import OperatorError

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


class SessionStore(Protocol):
    """The read side of one deployment's conversation history.

    Two questions, which is every question `runa sessions`, `runa chat` and the Sessions pages
    ask: which sessions are there, and what was said in one of them. No inheritance required,
    every adapter satisfies this by matching shape, the same escape hatch as `Cache`'s.
    """

    def listing(self, *, agent: str | None = None) -> list[SessionSummary]:
        """Return this deployment's sessions, most recently updated first.

        `agent` keeps only the sessions whose `agent_name` is `agent`, the agent whose run first
        saved a turn to it. A session id is the application's and says nothing about its agent.

        Ties on `updated_at` break by session id descending, in every adapter, so `runa chat
        --list` and the Sessions page agree between a local deployment and a shared one.
        """
        ...

    def messages(self, session_id: str) -> list[SessionMessage]:
        """Return `session_id`'s messages, oldest first.

        Raises `SessionNotFound` if this deployment has no session row for `session_id`, rather
        than returning an empty list: "no such session" and "a session with nothing in it" are
        different answers, and `runa chat --show` prints different things for them.
        """
        ...


def as_timestamp(value: str | datetime) -> str:
    """One rendering of a stored timestamp, whichever backend produced it.

    SQLite hands back `CURRENT_TIMESTAMP`'s `"YYYY-MM-DD HH:MM:SS"` text and Postgres a
    `TIMESTAMPTZ` -- naive on the first, offset-bearing on the second. Both go through here: one
    format, on one clock (UTC).
    """
    if isinstance(value, str):
        return value
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


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


def to_message(created_at: str | datetime, message_data: str) -> SessionMessage:
    """One stored `(created_at, message_data)` row as a `SessionMessage`.

    The row-to-object mapping, written once: every adapter stores a message as the same JSON
    blob, so none of them should be deciding for itself what `role` or `text` means. What those
    two mean for a stored item is `runa._items.role_and_text`'s to say, not this module's: a
    transcript reads an item, it doesn't define one.
    """
    role, text = role_and_text(json.loads(message_data))
    return SessionMessage(created_at=as_timestamp(created_at), role=role, text=text)


__all__ = [
    "SessionMessage",
    "SessionNotFound",
    "SessionStore",
    "SessionSummary",
    "as_timestamp",
    "listing_clause",
    "to_message",
]
