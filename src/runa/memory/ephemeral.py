"""memory/ephemeral.py: `EphemeralMemoryStore`, remembered facts that die with the process.

What `runa.db.memory_store()` resolves to under `RUNA_DATABASE_URL=memory://`. The nearest
neighbors are found by brute-force L2 distance over every stored vector (`db/vectors/ephemeral.py`),
which is the right algorithm for the only size this store ever reaches and keeps the whole backend
dependency-free: no `sqlite-vec` extension, no `pgvector`, no server.

Scoped by `user_id` the way `memory/sqlite.py`'s `vec0` partition key is, because it is the same
spec (`memory/vector.py`) that says so.
"""

from runa.db.vectors.ephemeral import EphemeralVectorStore
from runa.memory.vector import VectorMemoryStore, spec


class EphemeralMemoryStore(VectorMemoryStore):
    """The in-process `MemoryStore`: this process's remembered facts, gone when it exits."""

    def __init__(self, *, dimensions: int) -> None:
        """Store the embedding size this store expects, for parity with the other adapters."""
        self.dimensions = dimensions
        super().__init__(EphemeralVectorStore(spec(dimensions)))


__all__ = ["EphemeralMemoryStore"]
