"""cache/sqlite.py: `SQLiteCache`, the persistent `Cache` pointed at one specific local file.

`runa.db.cache()` already answers where a deployment's cache lives; this is for a cache that needs
its own file, which is what `db_path` names. The queries are `cache/database.py`'s.
"""

from pathlib import Path

from runa.cache import ENTRIES
from runa.cache.database import DatabaseCache
from runa.db import sqlite_database
from runa.db.sqlite import SQLiteDatabase


class SQLiteCache(DatabaseCache):
    """Persistent `Cache`: the `cache_entries` table inside a local SQLite file."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Store where this cache's table lives, defaulting to `db.sqlite_database()`.

        The table is created on first use.
        """
        location = Path(db_path) if db_path is not None else sqlite_database()
        super().__init__(SQLiteDatabase(location, ENTRIES))


__all__ = ["SQLiteCache"]
