"""session/sqlite.py: `SQLiteSession`, the session store pointed at one specific local file.

`runa.db.session(...)` already answers every ordinary question about where a session lives; this
is for the run that needs a particular `db/runa.db`-shaped file, which is what `db_path` names.
The queries are `session/database.py`'s.
"""

from pathlib import Path

from runa.db import sqlite_database
from runa.db.sqlite import SQLiteDatabase
from runa.session.database import DatabaseSession
from runa.session.store import MESSAGES, SESSIONS


class SQLiteSession(DatabaseSession):
    """Conversation history for one `session_id`, persisted to a local SQLite file."""

    def __init__(self, session_id: str, db_path: str | Path | None = None) -> None:
        """Store `session_id` and which SQLite database its history lives in.

        `db_path` defaults to `runa.db.sqlite_database()`, so this history lands in the same local
        database as every other concern's -- including when `RUNA_DATABASE_URL` relocates it or
        keeps it in memory. Pass one to point at some other file.
        """
        location = Path(db_path) if db_path is not None else sqlite_database()
        super().__init__(session_id, SQLiteDatabase(location, SESSIONS, MESSAGES))


__all__ = ["SQLiteSession"]
