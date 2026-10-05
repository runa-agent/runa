"""knowledge/ephemeral.py: `EphemeralKnowledgeStore`, ingested chunks that die with the process.

What `runa.db.knowledge_store()` resolves to under `RUNA_DATABASE_URL=memory://`. Brute-force L2
distance over every stored vector (`db/vectors/ephemeral.py`), which `memory/ephemeral.py` also
resolves to: the two stores differ in what they hold, not in how close two embeddings are.

Application-scoped rather than `user_id`-scoped, like every other `KnowledgeStore`, because that
is what this concern's spec declares (`knowledge/vector.py`).
"""

from runa.db.vectors.ephemeral import EphemeralVectorStore
from runa.knowledge.vector import VectorKnowledgeStore, spec


class EphemeralKnowledgeStore(VectorKnowledgeStore):
    """The in-process `KnowledgeStore`: this process's ingested chunks, gone when it exits."""

    def __init__(self, *, dimensions: int) -> None:
        """Store the embedding size this store expects, for parity with the other adapters."""
        self.dimensions = dimensions
        super().__init__(EphemeralVectorStore(spec(dimensions)))


__all__ = ["EphemeralKnowledgeStore"]
