"""session/postgres.py: `PostgresSession`, the session store pointed at one specific Postgres.

The `runa-ai[postgres]` extra, not a core dependency. `runa.db.session(...)` builds the same store
whenever `RUNA_DATABASE_URL` is a `postgresql://` one, so no call site has to name this;
constructing one by hand is the escape hatch for a database that is not this deployment's shared
one. The queries are `session/database.py`'s.
"""

from runa.db.pool import PostgresDatabase
from runa.session.database import DatabaseSession
from runa.session.store import MESSAGES, SESSIONS


class PostgresSession(DatabaseSession):
    """Conversation history for one `session_id`, persisted to a Postgres database."""

    def __init__(self, session_id: str, url: str) -> None:
        """Store `session_id` and which Postgres database its history lives in."""
        super().__init__(session_id, PostgresDatabase(url, SESSIONS, MESSAGES))


__all__ = ["PostgresSession"]
