"""db/pool.py: connect-and-create-if-missing plumbing for a shared Postgres.

`db/sqlite.py`'s counterpart for the other side of `runa.db`'s one decision. Every Postgres
adapter (`session/postgres.py`, `memory/postgres.py`, `knowledge/postgres.py`,
`cache/postgres.py`, `tracing/postgres.py`, `eval/postgres.py`) owns a different set of tables in
the same database; this only hands out the pool and applies each caller's DDL.

One pool per URL, cached at module level and shared by every adapter built from it, so a session,
a memory store and an exporter on the same database reuse one pool instead of each opening their
own -- the same "just works" ergonomics as `db/sqlite.py`'s shared file, here applied to pool
reuse instead of file reuse. Per *loop* as well as per URL, because a pool's connections belong
to the loop that opened them; that lifetime rule is `runa._loop.LoopCache`, shared with the other
two resources in Runa that have it.
"""

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any

import asyncpg

from runa._loop import LoopCache

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _background_loop() -> asyncio.AbstractEventLoop:
    """A daemon event loop for the synchronous callers that still need `asyncpg`.

    `TraceExporter.export` and the whole `runa traces`/`runa ui` read path are synchronous, and
    a trace is exported from inside a finishing run, so `asyncio.run()` (which demands there be
    no running loop) is not available. One long-lived loop on its own thread serves them all,
    and `LoopCache` gives it its own pool, as it would any other loop.
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

    Safe to call from inside a running event loop (which `asyncio.run` is not): the work happens
    on a different loop entirely, so it never tries to re-enter the caller's.
    """
    future = asyncio.run_coroutine_threadsafe(coro, _background_loop())
    return future.result(timeout)


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


async def connect(url: str, ddl: str) -> asyncpg.Pool:
    """Return `url`'s shared pool, applying `ddl` (idempotent) first."""
    pool = await get_pool(url)
    async with pool.acquire() as conn:
        await conn.execute(ddl)
    return pool


__all__ = ["close_pool", "connect", "get_pool", "run_sync"]
