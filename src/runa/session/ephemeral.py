"""session/ephemeral.py: the in-process session backend, both sides of it.

What `runa.db.session()`/`runa.db.sessions()` resolve to under `RUNA_DATABASE_URL=memory://`:
conversation history that never leaves the process. A test that wants an agent to remember the
last turn, or wants to assert what `runa chat --show` prints, sets one environment variable
instead of reaching for a temporary directory.

`EphemeralSession` and `EphemeralSessionStore` are two views on the same module-level tables,
exactly as the SQLite pair are two views on one file. That shared state is what makes this a
backend and not a stub: a run writes through the first and the Sessions page reads it back
through the second, which is the path the other two backends are also asked to get right.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

from runa._items import ConversationItem
from runa.session import SessionABC
from runa.session.store import (
    SessionMessage,
    SessionNotFound,
    SessionSummary,
    as_timestamp,
    to_message,
)


@dataclass
class _Row:
    """One session: when it was last written to, and its messages in insertion order."""

    updated_at: str
    messages: list[tuple[str, str]] = field(default_factory=list)


_sessions: dict[str, _Row] = {}


def reset() -> None:
    """Drop every stored session. `runa.db.reset_ephemeral()` is how a test reaches this."""
    _sessions.clear()


def _now() -> str:
    """The timestamp both SQL backends would have written, in `session/store.py`'s format."""
    return as_timestamp(datetime.now(UTC))


def _touch(session_id: str) -> _Row:
    """The row for `session_id`, created on first write, with `updated_at` moved to now."""
    row = _sessions.setdefault(session_id, _Row(updated_at=_now()))
    row.updated_at = _now()
    return row


class EphemeralSession(SessionABC):
    """Conversation history for one `session_id`, held in this process only."""

    def __init__(self, session_id: str, *, user_id: str | None = None) -> None:
        """Store `session_id`; its history is this module's, not this object's.

        `user_id` scopes this session's automatic memory, if its agent has any; see `SessionABC`.
        """
        self.session_id = session_id
        self.user_id = user_id

    async def get_items(self, limit: int | None = None) -> list[ConversationItem]:
        """Return this session's items, oldest first, capped at the latest `limit` if given."""
        row = _sessions.get(self.session_id)
        if row is None:
            return []
        messages = row.messages if limit is None else row.messages[-limit:]
        return [json.loads(message_data) for _, message_data in messages]

    async def add_items(self, items: list[ConversationItem]) -> None:
        """Append `items`, creating the session row on first write."""
        if not items:
            return
        row = _touch(self.session_id)
        row.messages.extend((_now(), json.dumps(item)) for item in items)

    async def set_items(self, items: list[ConversationItem]) -> None:
        """Replace this session's entire history with `items`."""
        row = _touch(self.session_id)
        row.messages = [(_now(), json.dumps(item)) for item in items]

    async def pop_item(self) -> ConversationItem | None:
        """Remove and return this session's most recent item, or `None` if it has none."""
        row = _sessions.get(self.session_id)
        if row is None or not row.messages:
            return None
        _, message_data = row.messages.pop()
        return json.loads(message_data)

    async def clear_session(self) -> None:
        """Delete this session and all of its items."""
        _sessions.pop(self.session_id, None)


class EphemeralSessionStore:
    """The in-process `SessionStore`: the read side of this process's session history."""

    def listing(self, *, agent: str | None = None) -> list[SessionSummary]:
        """Return this process's sessions, most recently updated first."""
        matches = [
            SessionSummary(id=session_id, updated_at=row.updated_at)
            for session_id, row in _sessions.items()
            if agent is None or session_id == agent or session_id.startswith(f"{agent}-")
        ]
        matches.sort(key=lambda summary: (summary.updated_at, summary.id), reverse=True)
        return matches

    def messages(self, session_id: str) -> list[SessionMessage]:
        """Return `session_id`'s messages, oldest first."""
        row = _sessions.get(session_id)
        if row is None:
            raise SessionNotFound(f"no session found with id {session_id!r}")
        return [to_message(created_at, message_data) for created_at, message_data in row.messages]


__all__ = ["EphemeralSession", "EphemeralSessionStore", "reset"]
