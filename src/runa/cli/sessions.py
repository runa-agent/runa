"""cli/sessions.py: `runa chat --list`/`--show`.

Thin formatting over the `SessionStore` `runa.db.sessions(root)` hands back, the same way
`cli/traces.py` only formats what a `TraceStore` already exposes. Which store the history comes
from is `runa.db`'s decision, so these commands read a shared Postgres and a local `db/runa.db`
with the same code -- and now in the same order, which was not true while each backend broke ties
on `updated_at` its own way.
"""

from pathlib import Path

from runa import db
from runa.session.store import SessionNotFound

__all__ = ["SessionNotFound", "list_sessions", "show_session"]


def list_sessions(*, root: Path) -> str:
    """List every session id this deployment has conversation history for, newest first."""
    sessions = db.sessions(root).listing()
    if not sessions:
        return "no sessions found"
    return "\n".join(f"{session.id}  {session.updated_at}" for session in sessions)


def show_session(session_id: str, *, root: Path) -> str:
    """Render a session's message history."""
    messages = db.sessions(root).messages(session_id)
    lines = [f"session {session_id}", ""]
    for message in messages:
        lines.append(f"{message.created_at}  {message.role}: {message.text}")
    return "\n".join(lines)
