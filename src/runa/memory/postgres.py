"""memory/postgres.py: `PostgresMemoryStore`, the shared `MemoryStore`.

The `runa-ai[postgres]` extra, not a core dependency. Same tables and query shapes as
`memory/sqlite.py`, with a `vector` column (the `pgvector` extension) in place of `sqlite-vec`'s
`vec0` virtual table, for a deployment where every replica searches one set of memories.

`user_id` is a plain filter column here, indexed so per-user nearest-neighbor search stays fast:
pgvector has no partition-key primitive the way `vec0` does. Distance is `<->` (Euclidean/L2),
matching `sqlite-vec`'s default and `Memory`'s `_DUPLICATE_DISTANCE` assumption.
"""

import json
from typing import Any

import asyncpg

from runa.db.pool import connect as _connect
from runa.memory import MemoryMatch

ITEMS_TABLE = "memory_items"


def ddl(dimensions: int) -> str:
    """The `memory_items` DDL for vectors of `dimensions` floats."""
    return f"""
    CREATE TABLE IF NOT EXISTS {ITEMS_TABLE} (
        id BIGSERIAL PRIMARY KEY,
        user_id TEXT,
        text TEXT NOT NULL,
        metadata TEXT,
        embedding vector({dimensions}) NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_{ITEMS_TABLE}_user_id ON {ITEMS_TABLE} (user_id);
    """


class PostgresMemoryStore:
    """`MemoryStore` backed by Postgres and `pgvector`, shared across processes."""

    def __init__(self, url: str, *, dimensions: int) -> None:
        """Store which Postgres database this store's items/vectors live in, and vector size."""
        self.url = url
        self.dimensions = dimensions

    async def _pool(self) -> asyncpg.Pool:
        return await _connect(self.url, ddl(self.dimensions))

    async def add(
        self,
        *,
        user_id: str | None,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None,
    ) -> int:
        """Store one already-embedded item for `user_id`, returning its new id."""
        pool = await self._pool()
        row = await pool.fetchrow(
            f"""
            INSERT INTO {ITEMS_TABLE} (user_id, text, metadata, embedding)
            VALUES ($1, $2, $3, $4)
            RETURNING id
            """,
            user_id,
            text,
            json.dumps(metadata) if metadata is not None else None,
            embedding,
        )
        assert row is not None
        return row["id"]

    async def search(
        self, *, user_id: str | None, embedding: list[float], k: int
    ) -> list[MemoryMatch]:
        """Return `user_id`'s `k` items closest to `embedding`, nearest first."""
        pool = await self._pool()
        rows = await pool.fetch(
            f"""
            SELECT id, text, metadata, embedding <-> $1 AS distance
            FROM {ITEMS_TABLE}
            WHERE user_id IS NOT DISTINCT FROM $2
            ORDER BY embedding <-> $1
            LIMIT $3
            """,
            embedding,
            user_id,
            k,
        )
        return [
            MemoryMatch(
                id=row["id"],
                text=row["text"],
                metadata=json.loads(row["metadata"]) if row["metadata"] is not None else None,
                distance=row["distance"],
            )
            for row in rows
        ]

    async def delete(self, *, user_id: str | None, memory_id: int) -> None:
        """Delete `user_id`'s item `memory_id`, if it exists."""
        pool = await self._pool()
        await pool.execute(
            f"DELETE FROM {ITEMS_TABLE} WHERE id = $1 AND user_id IS NOT DISTINCT FROM $2",
            memory_id,
            user_id,
        )


__all__ = ["ITEMS_TABLE", "PostgresMemoryStore", "ddl"]
