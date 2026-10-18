"""db/pool.py: connect-and-create-if-missing plumbing for a shared Postgres.

`db/sqlite.py`'s counterpart for the other side of `runa.db`'s one decision. Every Postgres
adapter (`session/postgres.py`, `db/vectors/postgres.py`, `cache/postgres.py`,
`tracing/postgres.py`, `eval/postgres.py`) owns a different set of tables in
the same database; this only hands out the pool and applies each caller's tables.

One pool per URL, cached at module level and shared by every adapter built from it, so a session,
a memory store and an exporter on the same database reuse one pool instead of each opening their
own -- the same "just works" ergonomics as `db/sqlite.py`'s shared file, here applied to pool
reuse instead of file reuse. Per *loop* as well as per URL, because a pool's connections belong
to the loop that opened them; that lifetime rule is `runa._loop.LoopCache`, shared with the other
three resources in Runa that have it.
"""

import asyncio
import functools
import threading
from collections.abc import Callable, Coroutine
from typing import Any

import asyncpg

from runa._loop import LoopCache
from runa.db.schema import POSTGRES, Declaration, ddl, reconcile

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _background_loop() -> asyncio.AbstractEventLoop:
    """A daemon event loop for the synchronous callers that still need `asyncpg`.

    `TraceExporter.export` and the whole `runa traces`/`runa ui` read path are synchronous, so
    they have no loop of their own to await on. One long-lived loop on its own thread serves them
    all, and `LoopCache` gives it its own pool, as it would any other loop -- which is the point:
    `asyncio.run()` per call would open and close a loop, and so a pool, per query.
    """
    global _loop
    if _loop is not None and not _loop.is_closed():
        return _loop
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, name="runa-postgres", daemon=True).start()
    return _loop


def run_sync[T](coro: Coroutine[Any, Any, T], *, timeout: float = 30.0) -> T:
    """Run `coro` on the background loop and wait for it, for a synchronous caller.

    This blocks the calling thread, so call it from a thread that has nothing better to do: a CLI
    command, or tracing's own export thread. Calling it from a thread running an event loop
    (`runa serve`'s) does not deadlock -- the work happens on a different loop entirely -- but it
    stalls every task on that loop for the duration, which is what `export_trace_later` and
    `evaluate_agent`'s `asyncio.to_thread` exist to prevent.
    """
    future = asyncio.run_coroutine_threadsafe(coro, _background_loop())
    return future.result(timeout)


def sync[**P, T](method: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, T]:
    """Expose an `async def` store method as the synchronous method its callers actually call.

    A `TraceStore`/`EvalStore`/`SessionStore` is a synchronous interface (`runa traces`,
    `runa ui` and `TraceExporter.export` reach it from a thread with no loop on it) while
    `asyncpg` is not; this pairs each async body with `run_sync`.
    """

    @functools.wraps(method)
    def synchronous(*args: P.args, **kwargs: P.kwargs) -> T:
        return run_sync(method(*args, **kwargs))

    return synchronous


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


__all__ = ["Shared", "close_pool", "connect", "get_pool", "run_sync", "sync"]
