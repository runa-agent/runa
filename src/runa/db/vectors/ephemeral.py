"""db/vectors/ephemeral.py: the in-process `VectorStore`, gone when the process exits.

What `runa.db.vector_store()` resolves to under `RUNA_DATABASE_URL=memory://`. Nearest neighbors
are found by brute-force L2 distance over every stored row, which is the right algorithm for the
only size this store ever reaches and keeps the backend dependency-free: no `sqlite-vec`, no
`pgvector`, no server.

Rows live at module level, keyed by `spec.name`, the way a database keeps its tables on disk: two
`runa.db.vector_store()` calls have to see each other's writes, so per-instance state would make
the whole backend a no-op. `runa.db.reset_ephemeral()` is how a test empties them.

Payloads are encoded and decoded exactly as the two persistent adapters do, rather than kept as
live objects. That costs nothing here and is what makes `memory://` a usable stand-in: a value
that would not survive a round trip in SQLite does not survive one here either, so a test cannot
pass against this adapter and fail against the one a deployment uses.
"""

import math
from dataclasses import dataclass, field
from typing import Any

from runa.db.vectors import VectorMatch, VectorSpec


@dataclass
class _Table:
    """One store's rows, the next id to hand out, and the version `reset` stamped."""

    rows: list[tuple[int, dict[str, Any], list[float]]] = field(default_factory=list)
    next_id: int = 1
    version: str | None = None


_tables: dict[str, _Table] = {}


def reset() -> None:
    """Drop every stored row. `runa.db.reset_ephemeral()` is how a test reaches this."""
    _tables.clear()


def distance(left: list[float], right: list[float]) -> float:
    """The L2 distance between two vectors, which is what `vec0` and `pgvector` also report."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


class EphemeralVectorStore:
    """The in-process `VectorStore`: this process's rows, searched by brute force."""

    def __init__(self, spec: VectorSpec) -> None:
        """Name this store's table, from `spec.name`, in this process's shared set of them."""
        self.spec = spec

    @property
    def _table(self) -> _Table:
        """This store's rows, created empty on first use the way a `CREATE IF NOT EXISTS` is."""
        return _tables.setdefault(self.spec.name, _Table())

    def _in_scope(self, payload: dict[str, Any], partition: str | None) -> bool:
        """Whether `payload` belongs to `partition`, which `None` is a scope of its own in."""
        if self.spec.partition_by is None:
            return True
        return payload[self.spec.partition_by] == partition

    async def add(self, *, payload: dict[str, Any], embedding: list[float]) -> int:
        """Store one payload row and its embedding, returning the new row's id.

        Ids come from a counter rather than the row count, so one is never reused after a delete.
        """
        table = self._table
        item_id = table.next_id
        table.next_id += 1
        table.rows.append((item_id, self.spec.encode(payload), list(embedding)))
        return item_id

    async def nearest(
        self, *, embedding: list[float], k: int, partition: str | None = None
    ) -> list[VectorMatch]:
        """Return the `k` rows closest to `embedding` in `partition`, nearest first."""
        matches = [
            VectorMatch(
                id=item_id,
                payload=self.spec.decode(payload),
                distance=distance(embedding, stored),
            )
            for item_id, payload, stored in self._table.rows
            if self._in_scope(payload, partition)
        ]
        matches.sort(key=lambda match: match.distance)
        return matches[:k]

    async def delete(self, *, item_id: int, partition: str | None = None) -> None:
        """Delete one row and its embedding, if `item_id` exists in `partition`'s scope."""
        table = self._table
        table.rows = [
            row
            for row in table.rows
            if not (row[0] == item_id and self._in_scope(row[1], partition))
        ]

    async def reset(self, *, version: str | None = None) -> None:
        """Delete every row and embedding, recording `version`, and leave the id counter alone."""
        table = self._table
        table.rows.clear()
        table.version = version

    async def version(self) -> str | None:
        """What the last `reset` recorded, or `None` if nothing has been reset or stamped."""
        return self._table.version


__all__ = ["EphemeralVectorStore", "distance", "reset"]
