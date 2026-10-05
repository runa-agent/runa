"""knowledge/sqlite.py: `SQLiteKnowledgeStore`, the local `KnowledgeStore`.

`db/runa.db`'s `knowledge_items`/`knowledge_vectors` tables, in the same
connect-and-create-if-missing file every other local adapter writes to (`db/sqlite.py`). Both the
`vec0` pairing and the mapping onto it are shared (`db/vectors/sqlite.py`,
`knowledge/vector.py`), so what belongs here is only the pairing of the two: this concern's spec,
and where its file lives. Unpartitioned, because `Knowledge` is application-scoped, not per user.
"""

from pathlib import Path

from runa.db import DEFAULT_DB_PATH
from runa.db.vectors.sqlite import SQLiteVectorStore
from runa.knowledge.vector import VectorKnowledgeStore, spec


class SQLiteKnowledgeStore(VectorKnowledgeStore):
    """The local `KnowledgeStore`: `db/runa.db`'s `knowledge_items`/`knowledge_vectors` tables."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, *, dimensions: int) -> None:
        """Store where this store's chunks/vectors live and the embedding size its table expects."""
        self.db_path = Path(db_path)
        self.dimensions = dimensions
        super().__init__(SQLiteVectorStore(spec(dimensions), self.db_path))


__all__ = ["SQLiteKnowledgeStore"]
