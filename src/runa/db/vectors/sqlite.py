"""db/vectors/sqlite.py: the local `VectorStore`, a `vec0` virtual table paired with its payload.

What `runa.db.vector_store()` resolves to for a deployment keeping its own `db/runa.db` file, which
is every deployment that has not set `RUNA_DATABASE_URL`.

`sqlite-vec` holds embeddings in a `vec0` virtual table that stores no payload of its own, so one
store is two tables joined by rowid: one `add` is two inserts that have to agree on a rowid, and
one search is a `MATCH`/`k` query against the virtual table joined back for the payload. That
pairing is what both concerns had identical, down to the `ORDER BY vectors.distance`.

Where the file lives is `runa.db.sqlite_path`'s answer, not this module's.
"""

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from runa.db.sqlite import connect as _connect
from runa.db.sqlite import pack_vector as pack
from runa.db.vectors import VectorMatch, VectorSpec


class SQLiteVectorStore:
    """The local `VectorStore`: a `vec0` table and its payload table in `db/runa.db`."""

    def __init__(self, spec: VectorSpec, db_path: Path) -> None:
        """Name this store's two tables, from `spec.name`, and the file they live in."""
        self.spec = spec
        self.db_path = db_path
        self.items = f"{spec.name}_items"
        self.vectors = f"{spec.name}_vectors"

    @property
    def ddl(self) -> str:
        """The two `CREATE ... IF NOT EXISTS` statements this store's tables need.

        `id` and `created_at` wrap the declared payload columns, since both concerns wanted both.
        A JSON column is `TEXT` here: the value arrives already encoded.
        """
        payload = ["id INTEGER PRIMARY KEY"]
        payload += [
            f"{name} TEXT NOT NULL" if column.required else f"{name} TEXT"
            for name, column in self.spec.columns.items()
        ]
        payload.append("created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        vectors = [f"{self.spec.partition_by} TEXT PARTITION KEY"] if self.spec.partition_by else []
        vectors.append(f"embedding float[{self.spec.dimensions}]")
        return (
            f"CREATE TABLE IF NOT EXISTS {self.items} (\n"
            + ",\n".join(f"    {line}" for line in payload)
            + f"\n);\nCREATE VIRTUAL TABLE IF NOT EXISTS {self.vectors} USING vec0(\n"
            + ",\n".join(f"    {line}" for line in vectors)
            + "\n);\n"
        )

    def _connect(self) -> sqlite3.Connection:
        return _connect(self.db_path, self.ddl, load_vec=True)

    async def add(self, *, payload: dict[str, Any], embedding: list[float]) -> int:
        """Store one payload row and its embedding, returning the new row's id.

        The two inserts share one connection and one commit: a payload row whose vector never
        landed would be invisible to every search, and a vector with no payload row would drop out
        of the join anyway.
        """
        values = self.spec.encode(payload)
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                f"INSERT INTO {self.items} ({columns}) VALUES ({placeholders})",
                tuple(values.values()),
            )
            item_id = cursor.lastrowid
            assert item_id is not None
            if self.spec.partition_by is None:
                conn.execute(
                    f"INSERT INTO {self.vectors} (rowid, embedding) VALUES (?, ?)",
                    (item_id, pack(embedding)),
                )
            else:
                conn.execute(
                    f"INSERT INTO {self.vectors} (rowid, {self.spec.partition_by}, embedding) "
                    "VALUES (?, ?, ?)",
                    (item_id, payload[self.spec.partition_by], pack(embedding)),
                )
            conn.commit()
        return item_id

    async def nearest(
        self, *, embedding: list[float], k: int, partition: str | None = None
    ) -> list[VectorMatch]:
        """Return the `k` rows closest to `embedding` in `partition`, nearest first.

        `None` is its own scope rather than "any", which is why the partition is matched with
        `IS`: one user's rows and the unscoped ones never see each other.
        """
        names = self.spec.payload_columns
        selected = ", ".join(f"items.{name}" for name in names)
        scope = f"AND vectors.{self.spec.partition_by} IS ?" if self.spec.partition_by else ""
        params: list[Any] = [pack(embedding), k]
        if self.spec.partition_by:
            params.append(partition)
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"""
                SELECT items.id, {selected}, vectors.distance
                FROM {self.vectors} AS vectors
                JOIN {self.items} AS items ON items.id = vectors.rowid
                WHERE vectors.embedding MATCH ? AND vectors.k = ? {scope}
                ORDER BY vectors.distance
                """,
                tuple(params),
            ).fetchall()
        return [
            VectorMatch(
                id=row[0],
                payload=self.spec.decode(dict(zip(names, row[1:-1], strict=True))),
                distance=row[-1],
            )
            for row in rows
        ]

    async def delete(self, *, item_id: int, partition: str | None = None) -> None:
        """Delete one row and its embedding, if `item_id` exists in `partition`'s scope."""
        scope = f"AND {self.spec.partition_by} IS ?" if self.spec.partition_by else ""
        params: list[Any] = [item_id]
        if self.spec.partition_by:
            params.append(partition)
        with closing(self._connect()) as conn:
            conn.execute(f"DELETE FROM {self.vectors} WHERE rowid = ? {scope}", tuple(params))
            conn.execute(f"DELETE FROM {self.items} WHERE id = ? {scope}", tuple(params))
            conn.commit()

    async def clear(self) -> None:
        """Delete every row and every embedding, leaving both tables in place."""
        with closing(self._connect()) as conn:
            conn.execute(f"DELETE FROM {self.vectors}")
            conn.execute(f"DELETE FROM {self.items}")
            conn.commit()


__all__ = ["SQLiteVectorStore"]
