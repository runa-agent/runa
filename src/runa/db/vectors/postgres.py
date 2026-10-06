"""db/vectors/postgres.py: the shared `VectorStore`, one `pgvector` column in one table.

The `runa-ai[postgres]` extra, not a core dependency. What `runa.db.vector_store()` resolves to
under a `postgresql://` URL, so every replica searches one set of rows instead of each searching
its own file.

One table rather than `sqlite.py`'s two: `pgvector` stores the embedding as an ordinary column, so
there is nothing to join back to. It also has no partition-key primitive the way `vec0` does, so a
partitioned store filters on the column and indexes it to keep per-partition search fast. Distance
is `<->` (Euclidean/L2), matching the other two adapters.

A JSON column is stored as `TEXT` here, not `jsonb`. `CREATE TABLE IF NOT EXISTS` leaves an
existing table alone, so emitting `jsonb` would give new deployments a column type old ones do not
have, and asyncpg hands back a decoded `dict` for one and a `str` for the other. Moving is a
migration, not a DDL edit, and `Column(json=True)` is what makes it one place to change.
"""

from typing import Any

from runa.db.pool import Shared
from runa.db.vectors import VectorMatch, VectorSpec


class PostgresVectorStore(Shared):
    """The shared `VectorStore`: one table of payload rows and their `pgvector` embeddings."""

    def __init__(self, spec: VectorSpec, url: str) -> None:
        """Name this store's table, from `spec.name`, and the database it lives in."""
        self.spec = spec
        self.items = f"{spec.name}_items"
        self.meta = f"{spec.name}_meta"
        # `ddl` reads all three of the above, so it can only be handed over once they're set.
        super().__init__(url, self.ddl)

    @property
    def ddl(self) -> str:
        """The two `CREATE TABLE IF NOT EXISTS` this store needs, and its partition index if any.

        `{name}_meta` holds at most one row, the version `reset` stamped; see `sqlite.py` for why
        it is a table of its own rather than a column on `{name}_items`.
        """
        lines = ["id BIGSERIAL PRIMARY KEY"]
        lines += [
            f"{name} TEXT NOT NULL" if column.required else f"{name} TEXT"
            for name, column in self.spec.columns.items()
        ]
        lines.append(f"embedding vector({self.spec.dimensions}) NOT NULL")
        lines.append("created_at TIMESTAMPTZ NOT NULL DEFAULT now()")
        body = ",\n".join(f"        {line}" for line in lines)
        index = (
            f"CREATE INDEX IF NOT EXISTS idx_{self.items}_{self.spec.partition_by} "
            f"ON {self.items} ({self.spec.partition_by});"
            if self.spec.partition_by
            else ""
        )
        meta = f"CREATE TABLE IF NOT EXISTS {self.meta} (version TEXT NOT NULL);"
        return (
            f"CREATE TABLE IF NOT EXISTS {self.items} (\n{body}\n    );\n    {index}\n    {meta}\n"
        )

    def _scope(self, start: int) -> str:
        """The `WHERE`/`AND` fragment scoping a query to one partition, or nothing."""
        if self.spec.partition_by is None:
            return ""
        return f"{self.spec.partition_by} IS NOT DISTINCT FROM ${start}"

    async def add(self, *, payload: dict[str, Any], embedding: list[float]) -> int:
        """Store one payload row and its embedding, returning the new row's id."""
        values = self.spec.encode(payload)
        columns = ", ".join(values)
        placeholders = ", ".join(f"${index}" for index in range(1, len(values) + 1))
        pool = await self._pool()
        row = await pool.fetchrow(
            f"""
            INSERT INTO {self.items} ({columns}, embedding)
            VALUES ({placeholders}, ${len(values) + 1})
            RETURNING id
            """,
            *values.values(),
            embedding,
        )
        assert row is not None
        return row["id"]

    async def nearest(
        self, *, embedding: list[float], k: int, partition: str | None = None
    ) -> list[VectorMatch]:
        """Return the `k` rows closest to `embedding` in `partition`, nearest first."""
        names = self.spec.payload_columns
        selected = ", ".join(names)
        scope = self._scope(2)
        params: list[Any] = [embedding]
        if scope:
            params.append(partition)
        pool = await self._pool()
        rows = await pool.fetch(
            f"""
            SELECT id, {selected}, embedding <-> $1 AS distance
            FROM {self.items}
            {f"WHERE {scope}" if scope else ""}
            ORDER BY embedding <-> $1
            LIMIT ${len(params) + 1}
            """,
            *params,
            k,
        )
        return [
            VectorMatch(
                id=row["id"],
                payload=self.spec.decode({name: row[name] for name in names}),
                distance=row["distance"],
            )
            for row in rows
        ]

    async def delete(self, *, item_id: int, partition: str | None = None) -> None:
        """Delete one row and its embedding, if `item_id` exists in `partition`'s scope."""
        scope = self._scope(2)
        params: list[Any] = [item_id]
        if scope:
            params.append(partition)
        pool = await self._pool()
        await pool.execute(
            f"DELETE FROM {self.items} WHERE id = $1 {f'AND {scope}' if scope else ''}",
            *params,
        )

    async def reset(self, *, version: str | None = None) -> None:
        """Delete every row and embedding, recording `version` as what the store now holds.

        One transaction, which is what every replica reading these rows needs: without it a
        concurrent search can see the new version over the old rows.
        """
        pool = await self._pool()
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(f"DELETE FROM {self.items}")
            await conn.execute(f"DELETE FROM {self.meta}")
            if version is not None:
                await conn.execute(f"INSERT INTO {self.meta} (version) VALUES ($1)", version)

    async def version(self) -> str | None:
        """What the last `reset` recorded, or `None` if nothing has been reset or stamped."""
        pool = await self._pool()
        row = await pool.fetchrow(f"SELECT version FROM {self.meta} LIMIT 1")
        return row["version"] if row is not None else None


__all__ = ["PostgresVectorStore"]
