"""cli/sessions.py: `runa chat --list`/`--show`.

Thin formatting over `runa.session.storage`, the same way `cli/traces.py` only formats what
`runa.tracing.storage` already exposes. Which store the history comes from is `runa.db`'s
decision, so these commands read a shared Postgres and a local `db/runa.db` with the same code.
"""

from pathlib import Path

from runa.cli._project import resolve_db_path
from runa.session.storage import SessionNotFound, session_messages, session_rows

__all__ = ["SessionNotFound", "list_sessions", "show_session"]


def list_sessions(*, root: Path) -> str:
    """List every session id this deployment has conversation history for."""
    rows = session_rows(db_path=resolve_db_path(root))
    if not rows:
        return "no sessions found"
    return "\n".join(f"{session_id}  {updated_at}" for session_id, updated_at in rows)


def show_session(session_id: str, *, root: Path) -> str:
    """Render a session's message history."""
    messages = session_messages(session_id, db_path=resolve_db_path(root))
    lines = [f"session {session_id}", ""]
    for message in messages:
        lines.append(f"{message['created_at']}  {message['role']}: {message['text']}")
    return "\n".join(lines)
