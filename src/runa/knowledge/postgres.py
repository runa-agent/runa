"""knowledge/postgres.py: `PostgresKnowledgeStore`, the shared `KnowledgeStore`.

The `runa-ai[postgres]` extra, not a core dependency. The same `knowledge_items` table as the
local adapter, with a `vector` column (the `pgvector` extension) in place of `sqlite-vec`'s `vec0`
virtual table, so every replica searches one ingested corpus instead of each ingesting its own.

Both the `pgvector` storage and the mapping onto it are shared (`db/vectors/postgres.py`,
`knowledge/vector.py`). Distance is `<->` (Euclidean/L2), matching `sqlite-vec`'s default.
"""

from runa.db.vectors.postgres import PostgresVectorStore
from runa.knowledge.vector import VectorKnowledgeStore, spec


class PostgresKnowledgeStore(VectorKnowledgeStore):
    """`KnowledgeStore` backed by Postgres and `pgvector`, shared across processes."""

    def __init__(self, url: str, *, dimensions: int) -> None:
        """Store which Postgres database this store's chunks/vectors live in, and vector size."""
        self.url = url
        self.dimensions = dimensions
        super().__init__(PostgresVectorStore(spec(dimensions), url))


__all__ = ["PostgresKnowledgeStore"]
