"""session/storage.py: the read side of `agent_sessions`/`agent_messages`.

What `runa sessions`, `runa chat --list/--continue/--resume` and `runa ui`'s Sessions pages read.
The write side is the `SessionABC` implementations themselves; this is the transcript view over
whatever they wrote.

Every function here first asks `runa.db.shared_url()` whether this deployment has a database to
share, and hands off to `session/postgres.py` if it does, the same way `tracing/storage.py`
dispatches. Without it the Sessions page would be the one page in `runa ui` still reading a local
file while the Traces and Evaluations pages beside it read Postgres: an empty session list on a
deployment whose history is fine, which is worse than no page at all.

`db_path` is where the *local* file lives, for a CLI invoked against another project's `--root`.
A shared deployment has one database and no such choice to make, so that argument goes unread.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from runa.db import DEFAULT_DB_PATH, shared_url
from runa.session.sqlite import MESSAGES_TABLE, SESSIONS_TABLE


class SessionNotFound(Exception):
    """Raised when a caller names a session id this deployment has no history for."""


def _parse_item(item: dict[str, Any]) -> tuple[str, str]:
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


def session_rows(*, db_path: Path = DEFAULT_DB_PATH) -> list[tuple[str, str]]:
    """Return `(session_id, updated_at)` for every session, oldest first."""
    if (url := shared_url()) is not None:
        from runa.session import postgres

        return postgres.session_rows(url)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                f"SELECT session_id, updated_at FROM {SESSIONS_TABLE} ORDER BY updated_at"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
    return [(row["session_id"], row["updated_at"]) for row in rows]


def sessions_for_agent(
    agent_name: str, *, db_path: Path = DEFAULT_DB_PATH
) -> list[tuple[str, str]]:
    """Return `(session_id, updated_at)` for `agent_name`'s past sessions, newest first.

    Matches a session id that's either exactly `agent_name` (the old, pre-session-per-chat
    default) or starts with `f"{agent_name}-"` (`cli/chat.py`'s current scheme), so
    `--continue`/`--resume` find history under either.
    """
    pattern = f"{_escape_like(agent_name)}-%"
    if (url := shared_url()) is not None:
        from runa.session import postgres

        return postgres.sessions_for_agent(url, agent_name, pattern)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                f"SELECT session_id, updated_at FROM {SESSIONS_TABLE} "
                "WHERE session_id = ? OR session_id LIKE ? ESCAPE '\\' "
                "ORDER BY updated_at DESC, rowid DESC",
                (agent_name, pattern),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
    return [(row["session_id"], row["updated_at"]) for row in rows]


def session_messages(session_id: str, *, db_path: Path = DEFAULT_DB_PATH) -> list[dict[str, str]]:
    """Return `session_id`'s messages as `{"created_at", "role", "text"}` dicts, oldest first.

    Raises `SessionNotFound` if this deployment has no session row for `session_id`.
    """
    if (url := shared_url()) is not None:
        from runa.session import postgres

        raw = postgres.session_messages(url, session_id)
    else:
        raw = _sqlite_messages(session_id, db_path)
    messages = []
    for created_at, message_data in raw:
        role, text = _parse_item(json.loads(message_data))
        messages.append({"created_at": created_at, "role": role, "text": text})
    return messages


def _escape_like(value: str) -> str:
    """`value` with `LIKE`'s wildcards escaped, so an agent named `a_b` doesn't match `axb`."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _sqlite_messages(session_id: str, db_path: Path) -> list[tuple[str, str]]:
    """`session_id`'s `(created_at, message_data)` rows from the local file, oldest first."""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        try:
            exists = conn.execute(
                f"SELECT 1 FROM {SESSIONS_TABLE} WHERE session_id = ?", (session_id,)
            ).fetchone()
        except sqlite3.OperationalError:
            exists = None
        if exists is None:
            raise SessionNotFound(f"no session found with id {session_id!r}")
        rows = conn.execute(
            f"SELECT created_at, message_data FROM {MESSAGES_TABLE} WHERE session_id = ? "
            "ORDER BY id",
            (session_id,),
        ).fetchall()
    return [(row["created_at"], row["message_data"]) for row in rows]


__all__ = ["SessionNotFound", "session_messages", "session_rows", "sessions_for_agent"]
