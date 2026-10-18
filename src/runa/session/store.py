"""session/store.py: `SessionStore`, the read side of `agent_sessions`/`agent_messages`.

What `runa sessions`, `runa chat --list/--continue/--resume` and `runa ui`'s Sessions pages read.
The write side is `Session` itself; this is the transcript view over whatever it wrote.

One interface, three adapters: `session/sqlite.py`, `session/postgres.py`, `session/ephemeral.py`.
Which one a caller gets is `runa.db.sessions(...)`'s decision, asked once, so no reader here or
in `web/` branches on a backend or names a file. Nothing in this module touches a database: it
holds the two tables both SQL adapters create (`db/schema.py` renders them per dialect), the
contract, the two objects a row becomes, and the rules the adapters have to agree on -- how a
session id matches an agent, how a listing is ordered, and how a timestamp is rendered. What a
stored message's `role` and `text` are is `runa._items`'s rule, applied here by `to_message`.

The first two are one rule in two languages: `listing_clause` is it in SQL, for the two adapters
that speak it, and `listing_rows` is it in Python, for the one that doesn't. They sit side by side
below, which is the whole reason they are here rather than in the adapters that call them.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from runa._items import role_and_text
from runa.db.schema import Column, Dialect, Index, Table
from runa.exceptions import OperatorError

SESSIONS = Table(
    "agent_sessions",
    columns=(
        Column("session_id", "text", primary_key=True),
        Column("created_at", "timestamp", default_now=True),
        Column("updated_at", "timestamp", default_now=True),
    ),
    indexes=(Index("updated_at", "updated_at DESC, session_id DESC"),),
)

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

        `agent` keeps only that agent's sessions, matching a session id that is either exactly
        `agent` (the old, pre-session-per-chat default) or starts with `f"{agent}-"`
        (`cli/chat.py`'s current scheme), so `--continue`/`--resume` find history under either.

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


def listing_clause(agent: str | None, dialect: Dialect) -> tuple[str, tuple[str, ...]]:
    r"""Everything after `SELECT ... FROM agent_sessions` for `listing`, and the values it binds.

    One sentence of policy: keep the sessions belonging to `agent`, if one was named, most
    recently updated first, ties broken on the session id. Belonging means a session id that is
    either exactly `agent` or starts with `f"{agent}-"`, with wildcards in `agent` escaped, so an
    agent named `a_b` can't match a session of `axb` in one backend and not the other.
    `agent=None` filters nothing, so a listing composes the same string either way. Placeholders
    are the `dialect`'s, like `Table.placeholders`.

    The whole tail rather than the `LIKE` pattern alone, because escaping a wildcard only works
    paired with the `ESCAPE` clause naming the escape character, and an adapter handed the pattern
    plus a docstring asking for the clause can take half the rule. The `ORDER BY` is here for the
    same reason: one tie-break, not one per adapter.
    """
    if agent is None:
        return _ORDER, ()
    exact, prefix = dialect.placeholder(1), dialect.placeholder(2)
    escaped = agent.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return (
        rf"WHERE session_id = {exact} OR session_id LIKE {prefix} ESCAPE '\' {_ORDER}",
        (agent, f"{escaped}-%"),
    )


def listing_rows(
    summaries: Iterable[SessionSummary], agent: str | None = None
) -> list[SessionSummary]:
    """`listing_clause` said in Python, for the adapter that keeps its sessions in memory.

    Same filters and order, down to the tie-break, over the `SessionSummary` rows
    `session/ephemeral.py` keeps. Change one rendering and the other with it: `runa chat --list`
    must list sessions the same way on every backend.
    """
    matching = [
        summary
        for summary in summaries
        if agent is None or summary.id == agent or summary.id.startswith(f"{agent}-")
    ]
    matching.sort(key=lambda summary: (summary.updated_at, summary.id), reverse=True)
    return matching


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
    "listing_rows",
    "to_message",
]
