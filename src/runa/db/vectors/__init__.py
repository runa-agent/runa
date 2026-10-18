"""db/vectors: the vector storage both `Memory` and `Knowledge` are built on.

`Memory` keeps a user's facts and the metadata they came with. `Knowledge` keeps an application's
document chunks and the file each was read from. They keep them the same way: an embedding, a
payload row beside it, and a nearest-neighbor search that returns the closest payloads with their
distances. Only the payload differs, so each concern holds one of these stores directly and maps
its own payload onto it.

Three adapters, one per answer `runa.db` gives about where state lives: `sqlite.py` (a `vec0`
virtual table), `postgres.py` (a `pgvector` column) and `ephemeral.py` (brute force, in process).
`runa.db.vector_store` is how a caller picks one, so nothing above it names a backend.

A backend Runa doesn't ship is one more of these, handed to `Memory(store=...)` or
`Knowledge(store=...)`: `VectorStore`'s three methods for `Memory`, `VersionedVectorStore`'s six
for `Knowledge`, held to `tests/contracts/vector.py` like the three in-tree adapters.

`reset`/`stamp`/`version` is the trio only `Knowledge` uses, and it lives at this layer rather
than above it because a version has to be stored wherever the rows are, durably and atomically,
which only an adapter can do. `Memory` never resets and keeps no version.

Distance is L2 in every adapter: `vec0`'s default, `pgvector`'s `<->`, and `math.sqrt` of the
summed squares in process. `Memory`'s duplicate threshold compares against it, so the three have to
agree on more than the ordering.
"""

import json
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Column:
    """One payload column: whether its value is JSON, and whether a row must have one.

    A `json=True` column is stored encoded (see `VectorSpec.encode`) by every adapter, and comes
    back decoded.
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
    """Storage for embeddings and the payload beside each one: all a `Memory` needs.

    Every in-tree adapter is built from a `VectorSpec` and holds exactly one concern's rows. Ids
    are unique for the life of the store, including after a delete, so a caller can hold one.
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


class VersionedVectorStore(VectorStore, Protocol):
    """A `VectorStore` that also records which corpus it holds: all a `Knowledge` needs."""

    async def reset(self) -> None:
        """Delete every row and embedding, and clear the version along with them.

        An emptied store is always an unversioned one, so it is never mistaken for a current
        corpus; a caller refilling it records what it put there afterwards, with `stamp`.
        """
        ...

    async def stamp(self, version: str) -> None:
        """Record `version` as what the store now holds, leaving the rows alone.

        Called after the last `add`, never before: a version claims rows that are already there.
        """
        ...

    async def version(self) -> str | None:
        """What the last `stamp` recorded, or `None` if nothing has been stamped since a `reset`.

        Opaque: a caller that wants to know whether the stored rows are still the ones it would
        write compares its own marker against this. `Knowledge` keeps a hash of its source
        directory here, which is how a second process knows an ingest already happened.
        """
        ...


__all__ = ["Column", "VectorMatch", "VectorSpec", "VectorStore", "VersionedVectorStore"]
