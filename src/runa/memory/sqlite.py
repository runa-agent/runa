"""memory/sqlite.py: `SQLiteMemoryStore`, the local `MemoryStore`.

`db/runa.db`'s `memory_items`/`memory_vectors` tables, in the same connect-and-create-if-missing
file every other local adapter writes to (`db/sqlite.py`). Both the `vec0` pairing and the mapping
onto it are shared (`db/vectors/sqlite.py`, `memory/vector.py`), so what belongs here is only the
pairing of the two: this concern's spec, and where its file lives.
"""

from pathlib import Path

from runa.db import sqlite_path
from runa.db.vectors.sqlite import SQLiteVectorStore
from runa.memory.vector import VectorMemoryStore, spec


class SQLiteMemoryStore(VectorMemoryStore):
    """The local `MemoryStore`: `db/runa.db`'s `memory_items`/`memory_vectors` tables.

    `user_id` is a `vec0` partition key (not just a `WHERE` filter), so nearest-neighbor search
    stays correct per user instead of a global top-k that another user's items could crowd out.
    """

    def __init__(self, db_path: str | Path | None = None, *, dimensions: int) -> None:
        """Store where this store's items/vectors live and the embedding size its table expects.

        `db_path` defaults to `runa.db.sqlite_path()`.
        """
        self.db_path = Path(db_path) if db_path is not None else sqlite_path()
        self.dimensions = dimensions
        super().__init__(SQLiteVectorStore(spec(dimensions), self.db_path))


__all__ = ["SQLiteMemoryStore"]
