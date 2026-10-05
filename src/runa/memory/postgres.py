"""memory/postgres.py: `PostgresMemoryStore`, the shared `MemoryStore`.

The `runa-ai[postgres]` extra, not a core dependency. The same `memory_items` table as the local
adapter, with a `vector` column (the `pgvector` extension) in place of `sqlite-vec`'s `vec0`
virtual table, for a deployment where every replica searches one set of memories.

Both the `pgvector` storage and the mapping onto it are shared (`db/vectors/postgres.py`,
`memory/vector.py`). `user_id` is a filter column there rather than a partition key, since
pgvector has no partition-key primitive, and it is indexed because this spec declares it as the
partition. Distance is `<->` (Euclidean/L2), matching `sqlite-vec`'s default and `Memory`'s
`_DUPLICATE_DISTANCE` assumption.
"""

from runa.db.vectors.postgres import PostgresVectorStore
from runa.memory.vector import VectorMemoryStore, spec


class PostgresMemoryStore(VectorMemoryStore):
    """`MemoryStore` backed by Postgres and `pgvector`, shared across processes."""

    def __init__(self, url: str, *, dimensions: int) -> None:
        """Store which Postgres database this store's items/vectors live in, and vector size."""
        self.url = url
        self.dimensions = dimensions
        super().__init__(PostgresVectorStore(spec(dimensions), url))


__all__ = ["PostgresMemoryStore"]
