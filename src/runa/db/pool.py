"""db/pool.py: connect-and-create-if-missing plumbing for a shared Postgres.

`db/sqlite.py`'s counterpart for the other side of `runa.db`'s one decision. Every Postgres
user (each concern through `PostgresDatabase` below, and `db/vectors/postgres.py`) owns a
different set of tables in the same database; this only hands out the pool and applies each
caller's tables.

One pool per URL, cached at module level and shared by every adapter built from it, so a session,
a memory store and an exporter on the same database reuse one pool instead of each opening their
own -- the same "just works" ergonomics as `db/sqlite.py`'s shared file, here applied to pool
reuse instead of file reuse. Per *loop* as well as per URL, because a pool's connections belong
to the loop that opened them; that lifetime rule is `runa._loop.LoopCache`, shared with the other
three resources in Runa that have it.
"""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import asyncpg

from runa._loop import LoopCache
from runa.db.handle import Queries, Row
from runa.db.schema import POSTGRES, Declaration, ddl, reconcile

_pools: LoopCache[str, asyncpg.Pool] = LoopCache()


async def _create_pool(url: str) -> asyncpg.Pool:
    """Open a pool on `url`, with `pgvector`'s codec registered on every connection.

    `vector` has to exist as a type before any connection can register its codec, so a bare
    bootstrap connection creates the extension first -- `init=register_vector` on the pool itself
    would otherwise run on a database that doesn't have the type yet.
    """
    from pgvector.asyncpg import register_vector

    bootstrap = await asyncpg.connect(url)
    try:
        await bootstrap.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        await bootstrap.close()
    return await asyncpg.create_pool(url, init=register_vector)


async def get_pool(url: str) -> asyncpg.Pool:
    """Return `url`'s shared pool on the *current* event loop, creating it on first use."""
    return await _pools.aget(url, lambda: _create_pool(url))


async def close_pool(url: str) -> None:
    """Close and forget `url`'s pool on the current loop, if it has one.

    A deployment never needs this: its loop and its pool live as long as the process. A caller
    that closes its own loop does, since the pool would otherwise hold that database's
    connections until the loop is garbage collected.
    """
    pool = _pools.pop(url)
    if pool is not None:
        await pool.close()


async def connect(url: str, *tables: Declaration) -> asyncpg.Pool:
    """Return `url`'s shared pool, with `tables` created and reconciled (idempotent) first.

    `db/sqlite.py`'s `connect` applied to the shared side: `db/schema.py`'s `ddl` creates what's
    missing, then `reconcile` adds the columns a declaration has grown since this database was
    created. A Postgres is usually created by whichever replica connects first and so is already
    current, but "usually" is not the same promise as asking it.
    """
    pool = await get_pool(url)
    async with pool.acquire() as conn:
        await conn.execute(ddl(POSTGRES, *tables))
        for statement in reconcile(POSTGRES, await conn.fetch(POSTGRES.columns_query), *tables):
            await conn.execute(statement)
    return pool


class Shared:
    """A store that lives in the shared Postgres: which database it's in, and its own tables.

    What every Postgres adapter needs and all it needs: the URL it was built with, and its tables
    applied once before its first query. Subclassing this is what keeps the lifetime rule above
    (one pool per URL per loop, tables created on first use) a fact of this module instead of
    something each adapter rediscovers -- five of them had, in twelve places.
    """

    def __init__(self, url: str, *tables: Declaration) -> None:
        """Store which Postgres database this store's tables live in; connected lazily."""
        self.url = url
        self.tables = tables

    async def _pool(self) -> asyncpg.Pool:
        """This store's shared pool, with its own tables created on first use."""
        return await connect(self.url, *self.tables)


class _Statements:
    """`Database`'s five statements over whatever asyncpg object `_target` names."""

    async def _target(self) -> asyncpg.Pool | asyncpg.pool.PoolConnectionProxy:
        raise NotImplementedError

    async def fetch(self, sql: str, *args: Any) -> list[Row]:
        """Every row `sql` selects."""
        return list(await (await self._target()).fetch(sql, *args))

    async def fetchrow(self, sql: str, *args: Any) -> Row | None:
        """The first row `sql` selects, or `None`."""
        return await (await self._target()).fetchrow(sql, *args)

    async def fetchval(self, sql: str, *args: Any) -> Any:
        """The first column of the first row `sql` selects, or `None`."""
        return await (await self._target()).fetchval(sql, *args)

    async def execute(self, sql: str, *args: Any) -> int:
        """Run `sql` and return how many rows it changed.

        asyncpg reports it as a status tag, `DELETE 1` or `INSERT 0 3`: the count is the last word.
        """
        status: str = await (await self._target()).execute(sql, *args)
        count = status.rsplit(" ", 1)[-1]
        return int(count) if count.isdigit() else 0

    async def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        """Run `sql` once per row of `rows`."""
        await (await self._target()).executemany(sql, rows)


class _Transaction(_Statements):
    """The statements of one `PostgresDatabase.transaction`, on its one connection."""

    def __init__(self, conn: asyncpg.pool.PoolConnectionProxy) -> None:
        self._conn = conn

    async def _target(self) -> asyncpg.pool.PoolConnectionProxy:
        return self._conn


class PostgresDatabase(_Statements, Shared):
    """`Database` over this deployment's shared Postgres: the pool, with `tables` applied first."""

    async def _target(self) -> asyncpg.Pool:
        return await self._pool()

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Queries]:
        """One pooled connection inside `BEGIN ... COMMIT`, rolled back if the block raises."""
        pool = await self._pool()
        async with pool.acquire() as conn, conn.transaction():
            yield _Transaction(conn)


__all__ = ["PostgresDatabase", "Shared", "close_pool", "connect", "get_pool"]
