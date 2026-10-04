"""session/store.py: `SessionStore`, the read side of `agent_sessions`/`agent_messages`.

What `runa sessions`, `runa chat --list/--continue/--resume` and `runa ui`'s Sessions pages read.
The write side is `SessionABC` itself; this is the transcript view over whatever it wrote.

One interface, three adapters: `session/sqlite.py`, `session/postgres.py`, `session/ephemeral.py`.
Which one a caller gets is `runa.db.sessions(...)`'s decision, asked once, so no reader here or
in `web/` branches on a backend or names a file. Nothing in this module touches a database: it
holds the contract, the two objects a row becomes, and the three rules the adapters have to agree
on -- how a session id matches an agent, how a timestamp is rendered, and how a stored message is
flattened to text.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol


class SessionNotFound(Exception):
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

        Ties on `updated_at` break by session id descending, in every adapter. Two sessions
        written in the same second used to list in a different order depending on the backend,
        which made `runa chat --list` and the Sessions page disagree between a local deployment
        and a shared one.
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
    `TIMESTAMPTZ`, so the same session listed from a local deployment used to read differently
    from one listed from a shared one -- naive on the first, offset-bearing on the second. Both
    go through here instead: one format, on one clock (UTC).
    """
    if isinstance(value, str):
        return value
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def agent_pattern(agent: str) -> str:
    r"""The `LIKE` pattern matching `agent`'s session ids, wildcards in `agent` escaped.

    Shared by both SQL adapters so an agent named `a_b` can't match a session of `axb` in one
    backend and not the other. Pair it with `ESCAPE '\\'`.
    """
    escaped = agent.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}-%"


def message_text(item: dict[str, Any]) -> tuple[str, str]:
    """One stored message's `(role, text)`, flattening whichever content shape it was saved in."""
    role = item.get("role") or item.get("type", "item")
    content = item.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = [part["text"] for part in content if isinstance(part, dict) and "text" in part]
        text = "".join(parts) if parts else str(content)
    else:
        text = str(item)
    return role, text


def to_message(created_at: str | datetime, message_data: str) -> SessionMessage:
    """One stored `(created_at, message_data)` row as a `SessionMessage`.

    The row-to-object mapping, written once: every adapter stores a message as the same JSON
    blob, so none of them should be deciding for itself what `role` or `text` means.
    """
    role, text = message_text(json.loads(message_data))
    return SessionMessage(created_at=as_timestamp(created_at), role=role, text=text)


__all__ = [
    "SessionMessage",
    "SessionNotFound",
    "SessionStore",
    "SessionSummary",
    "agent_pattern",
    "as_timestamp",
    "message_text",
    "to_message",
]
