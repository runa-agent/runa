"""db/vectors: the vector storage both `Memory` and `Knowledge` are built on.

`Memory` keeps a user's facts and the metadata they came with. `Knowledge` keeps an application's
document chunks and the file each was read from. They keep them the same way: an embedding, a
payload row beside it, and a nearest-neighbor search that returns the closest payloads with their
distances. Only the payload differs, which is why the three backends below are shared and the two
concerns map onto them in `memory/vector.py` and `knowledge/vector.py`.

Three adapters, one per answer `runa.db` gives about where state lives: `sqlite.py` (a `vec0`
virtual table), `postgres.py` (a `pgvector` column) and `ephemeral.py` (brute force, in process).
`runa.db.vector_store` is how a caller picks one, so nothing above it names a backend.

This seam is plumbing, not a primitive. An application never holds a `VectorStore`: it holds a
`Memory` or a `Knowledge`, and `Memory(store=...)` still takes a `MemoryStore`. A fourth backend
implements the five methods below and is held to `tests/contracts/vector.py`, the same contract
the three in-tree adapters answer.

`reset`/`version` is the one pair only `Knowledge` uses, and it lives at this layer rather than
above it because a version has to be written in the same transaction as the rows it describes,
which only an adapter can do. `Memory` resets with no version and never reads one.

Distance is L2 in every adapter: `vec0`'s default, `pgvector`'s `<->`, and `math.sqrt` of the
summed squares in process. `Memory._DUPLICATE_DISTANCE` compares against it, so the three have to
agree on more than the ordering.
"""

import json
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Column:
    """One payload column: whether its value is JSON, and whether a row must have one.

    `json=True` is the declaration that made the three adapters disagree before they shared this
    module: a `dict` has to be encoded to be stored, and an adapter that skipped the encoding
    (the in-process one did) round-tripped values the other two could not. Each adapter now
    encodes a JSON column in whatever way its storage wants, and every adapter returns the
    decoded value.
    """

    json: bool = False
    required: bool = False


@dataclass(frozen=True)
class VectorSpec:
    """What one concern's vector storage is called, holds, and is scoped by.

    `partition_by` names a payload column that scopes search instead of filtering it: with one
    set, a nearest-neighbor query returns that partition's `k` closest rows rather than a global
    top-k another partition's rows could crowd out. `Memory` partitions by `user_id`; `Knowledge`
    is application-scoped and passes `None`.
    """

    name: str
    dimensions: int
    columns: dict[str, Column]
    partition_by: str | None = None

    @property
    def payload_columns(self) -> tuple[str, ...]:
        """Every payload column, in declaration order, which is also the order rows come back."""
        return tuple(self.columns)

    def encode(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Encode `payload`'s JSON columns, leaving everything else as it is."""
        return {
            column: json.dumps(value) if self.columns[column].json and value is not None else value
            for column, value in payload.items()
        }

    def decode(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Decode `payload`'s JSON columns, the inverse of `encode`."""
        return {
            column: json.loads(value) if self.columns[column].json and value is not None else value
            for column, value in payload.items()
        }


@dataclass
class VectorMatch:
    """One nearest-neighbor result: its id, its decoded payload, and its distance to the query."""

    id: int
    payload: dict[str, Any]
    distance: float


class VectorStore(Protocol):
    """Storage for embeddings and the payload beside each one.

    Every adapter is built from a `VectorSpec` and holds exactly one concern's rows. Ids are
    unique for the life of the store, including after a delete, so a caller can hold one.
    """

    async def add(self, *, payload: dict[str, Any], embedding: list[float]) -> int:
        """Store one payload row and its embedding, returning the new row's id.

        A partitioned store takes the row's partition from `payload[spec.partition_by]` rather
        than from a second argument that could disagree with it.
        """
        ...

    async def nearest(
        self, *, embedding: list[float], k: int, partition: str | None = None
    ) -> list[VectorMatch]:
        """Return the `k` rows closest to `embedding` in `partition`, nearest first.

        `partition` is its own scope rather than a filter, and `None` is a scope too: an
        unpartitioned row and a partitioned one never see each other.
        """
        ...

    async def delete(self, *, item_id: int, partition: str | None = None) -> None:
        """Delete one row and its embedding, if `item_id` exists in `partition`'s scope.

        A no-op when it does not, including when the row belongs to another partition.
        """
        ...

    async def reset(self, *, version: str | None = None) -> None:
        """Delete every row and embedding, recording `version` as what the store now holds.

        One operation rather than a `clear` and a separate write, because the two have to agree:
        a version that outlived the rows it describes would claim a corpus the store no longer
        has, and every adapter here can empty the tables and stamp them in one transaction.
        `None` is for a caller that keeps no version (`Memory`), and leaves `version()` empty.
        """
        ...

    async def version(self) -> str | None:
        """What the last `reset` recorded, or `None` if nothing has been reset or stamped.

        Opaque: a caller that wants to know whether the stored rows are still the ones it would
        write compares its own marker against this. `Knowledge` keeps a hash of its source
        directory here, which is how a second process knows an ingest already happened.
        """
        ...


__all__ = ["Column", "VectorMatch", "VectorSpec", "VectorStore"]
