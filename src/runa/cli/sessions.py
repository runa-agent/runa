"""cli/sessions.py: `runa chat --list`/`--show` over db/runa.db.

Session history (`agent_sessions`/`agent_messages`, written by `SQLiteSession`, see
`cli/chat.py`) lives in `db/runa.db`; this module only reads it.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from runa.cli._project import resolve_db_path


class SessionNotFound(Exception):
    """Raised when `show_session` names a session id `db/runa.db` has no history for."""


def _parse_item(item: dict[str, Any]) -> tuple[str, str]:
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


def session_rows(*, root: Path) -> list[tuple[str, str]]:
    """Return `(session_id, updated_at)` for every session in `db/runa.db`, oldest first."""
    db_path = resolve_db_path(root)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT session_id, updated_at FROM agent_sessions ORDER BY updated_at"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
    return [(row["session_id"], row["updated_at"]) for row in rows]


def list_sessions(*, root: Path) -> str:
    """List every session id `db/runa.db` has conversation history for."""
    rows = session_rows(root=root)
    if not rows:
        return "no sessions found"
    return "\n".join(f"{session_id}  {updated_at}" for session_id, updated_at in rows)


def list_sessions_for_agent(agent_name: str, *, root: Path) -> list[tuple[str, str]]:
    """Return `(session_id, updated_at)` for `agent_name`'s past sessions, newest first.

    Matches a session id that's either exactly `agent_name` (the old, pre-session-per-chat
    default) or starts with `f"{agent_name}-"` (`cli/chat.py`'s current scheme), so
    `--continue`/`--resume` find history under either.
    """
    db_path = resolve_db_path(root)
    escaped = agent_name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT session_id, updated_at FROM agent_sessions "
                "WHERE session_id = ? OR session_id LIKE ? ESCAPE '\\' "
                "ORDER BY updated_at DESC, rowid DESC",
                (agent_name, f"{escaped}-%"),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
    return [(row["session_id"], row["updated_at"]) for row in rows]


def session_messages(session_id: str, *, root: Path) -> list[dict[str, str]]:
    """Return `session_id`'s messages as `{"created_at", "role", "text"}` dicts, oldest first.

    Raises `SessionNotFound` if `db/runa.db` has no `agent_sessions` row for `session_id`.
    """
    db_path = resolve_db_path(root)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        try:
            exists = conn.execute(
                "SELECT 1 FROM agent_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        except sqlite3.OperationalError:
            exists = None
        if exists is None:
            raise SessionNotFound(f"no session found with id {session_id!r}")
        rows = conn.execute(
            "SELECT message_data, created_at FROM agent_messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()

    messages = []
    for row in rows:
        role, text = _parse_item(json.loads(row["message_data"]))
        messages.append({"created_at": row["created_at"], "role": role, "text": text})
    return messages


def show_session(session_id: str, *, root: Path) -> str:
    """Render a session's message history."""
    messages = session_messages(session_id, root=root)
    lines = [f"session {session_id}", ""]
    for message in messages:
        lines.append(f"{message['created_at']}  {message['role']}: {message['text']}")
    return "\n".join(lines)
