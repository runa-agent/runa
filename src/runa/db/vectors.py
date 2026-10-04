"""db/vectors.py: the `vec0` table pairing both local vector stores are built on.

`memory/sqlite.py` and `knowledge/sqlite.py` keep different things: a user's facts and the
metadata they came with, an application's document chunks and the file they were read from. They
kept them the same way. A `vec0` virtual table holds the embeddings, a companion table holds the
payload, and the two are joined by rowid, which means one `add` is two inserts that have to agree
on a rowid and one search is a `MATCH`/`k` query against the virtual table joined back for the
payload. That pairing was what the two adapters had identical, down to the `ORDER BY
vectors.distance`, while the payload is what actually differs between them.

So the pairing lives here and the payload stays in each store. A `VectorTable` is told its name,
its payload columns and whether `user_id` partitions its vectors; `memory/sqlite.py` and
`knowledge/sqlite.py` keep their own DDL columns and their own `MemoryMatch`/`KnowledgeMatch`
mapping, which is all either of them was ever really about.

Table names follow from `name`: `memory` owns `memory_items` and `memory_vectors`, the names both
adapters already declared by hand. Where the file lives is `runa.db.sqlite_path`'s answer, as
always, not this module's.
"""

import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from runa.db.sqlite import connect as _connect
from runa.db.sqlite import pack_vector as _pack


class VectorTable:
    """A `vec0` virtual table and the payload table it joins to by rowid.

    `columns` maps each payload column to its SQL declaration, in the order the table declares
    them; `id` and `created_at` are added around them, since every caller wanted both. Passing
    `partition_by="user_id"` makes that column a `vec0` partition key rather than a `WHERE`
    filter, so nearest-neighbor search stays correct per user instead of a global top-k another
    user's items could crowd out.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        name: str,
        columns: dict[str, str],
        dimensions: int,
        partition_by: str | None = None,
    ) -> None:
        """Name this store's two tables and describe the payload table's own columns."""
        self.db_path = db_path
        self.items = f"{name}_items"
        self.vectors = f"{name}_vectors"
        self.columns = columns
        self.dimensions = dimensions
        self.partition_by = partition_by

    @property
    def ddl(self) -> str:
        """The two `CREATE ... IF NOT EXISTS` statements this store's tables need."""
        payload = ["id INTEGER PRIMARY KEY"]
        payload += [f"{column} {decl}" for column, decl in self.columns.items()]
        payload.append("created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        vectors = [f"{self.partition_by} TEXT PARTITION KEY"] if self.partition_by else []
        vectors.append(f"embedding float[{self.dimensions}]")
        return (
            f"CREATE TABLE IF NOT EXISTS {self.items} (\n"
            + ",\n".join(f"    {line}" for line in payload)
            + f"\n);\nCREATE VIRTUAL TABLE IF NOT EXISTS {self.vectors} USING vec0(\n"
            + ",\n".join(f"    {line}" for line in vectors)
            + "\n);\n"
        )

    def _connect(self) -> sqlite3.Connection:
        return _connect(self.db_path, self.ddl, load_vec=True)

    def insert(self, embedding: list[float], **values: Any) -> int:
        """Store one payload row and its `embedding`, returning the new row's id.

        The two inserts share one connection and one commit: a payload row whose vector never
        landed would be invisible to every search, and a vector with no payload row would drop
        out of the join anyway.
        """
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                f"INSERT INTO {self.items} ({columns}) VALUES ({placeholders})",
                tuple(values.values()),
            )
            item_id = cursor.lastrowid
            assert item_id is not None
            if self.partition_by is None:
                conn.execute(
                    f"INSERT INTO {self.vectors} (rowid, embedding) VALUES (?, ?)",
                    (item_id, _pack(embedding)),
                )
            else:
                conn.execute(
                    f"INSERT INTO {self.vectors} (rowid, {self.partition_by}, embedding) "
                    "VALUES (?, ?, ?)",
                    (item_id, values[self.partition_by], _pack(embedding)),
                )
            conn.commit()
        return item_id

    def nearest(
        self,
        embedding: list[float],
        *,
        k: int,
        columns: Sequence[str],
        partition: str | None = None,
    ) -> list[tuple[Any, ...]]:
        """Return the `k` rows closest to `embedding` as `(id, *columns, distance)`, nearest first.

        `partition` scopes the search when this table has a partition key, and is ignored when it
        doesn't. `None` is its own scope rather than "any", which is why it's matched with `IS`:
        one user's items and the unscoped ones never see each other.
        """
        selected = ", ".join(f"items.{column}" for column in columns)
        scope = f"AND vectors.{self.partition_by} IS ?" if self.partition_by else ""
        params: list[Any] = [_pack(embedding), k]
        if self.partition_by:
            params.append(partition)
        with closing(self._connect()) as conn:
            return conn.execute(
                f"""
                SELECT items.id, {selected}, vectors.distance
                FROM {self.vectors} AS vectors
                JOIN {self.items} AS items ON items.id = vectors.rowid
                WHERE vectors.embedding MATCH ? AND vectors.k = ? {scope}
                ORDER BY vectors.distance
                """,
                tuple(params),
            ).fetchall()

    def delete(self, item_id: int, *, partition: str | None = None) -> None:
        """Delete one row and its vector, if `item_id` exists in `partition`'s scope."""
        scope = f"AND {self.partition_by} IS ?" if self.partition_by else ""
        params: list[Any] = [item_id]
        if self.partition_by:
            params.append(partition)
        with closing(self._connect()) as conn:
            conn.execute(f"DELETE FROM {self.vectors} WHERE rowid = ? {scope}", tuple(params))
            conn.execute(f"DELETE FROM {self.items} WHERE id = ? {scope}", tuple(params))
            conn.commit()

    def clear(self) -> None:
        """Delete every row and every vector, leaving both tables in place."""
        with closing(self._connect()) as conn:
            conn.execute(f"DELETE FROM {self.vectors}")
            conn.execute(f"DELETE FROM {self.items}")
            conn.commit()


__all__ = ["VectorTable"]
