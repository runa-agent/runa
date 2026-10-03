"""db/pool.py: connect-and-create-if-missing plumbing for a shared Postgres.

`db/sqlite.py`'s counterpart for the other side of `runa.db`'s one decision. Every Postgres
adapter (`session/postgres.py`, `memory/postgres.py`, `knowledge/postgres.py`,
`cache/postgres.py`, `tracing/postgres.py`, `eval/postgres.py`) owns a different set of tables in
the same database; this only hands out the pool and applies each caller's DDL.

One pool per URL, cached at module level and shared by every adapter built from it, so a session,
a memory store and an exporter on the same database reuse one pool instead of each opening their
own -- the same "just works" ergonomics as `db/sqlite.py`'s shared file, here applied to pool
reuse instead of file reuse.
"""

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any

import asyncpg

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _background_loop() -> asyncio.AbstractEventLoop:
    """A daemon event loop for the synchronous callers that still need `asyncpg`.

    `TraceExporter.export` and the whole `runa traces`/`runa ui` read path are synchronous, and
    a trace is exported from inside a finishing run, so `asyncio.run()` (which demands there be
    no running loop) is not available. One long-lived loop on its own thread serves them all,
    and `get_pool`'s per-loop keying gives it its own pool, as it would any other loop.
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


_pools: dict[tuple[int, str], asyncpg.Pool] = {}
_pools_lock = asyncio.Lock()


async def get_pool(url: str) -> asyncpg.Pool:
    """Return `url`'s shared pool on the *current* event loop, creating it on first use.

    Keyed by `(id(loop), url)`, not just `url`: an `asyncpg.Pool`'s connections belong to the
    loop that created them, so reusing a pool from a since-closed loop (e.g. a second
    `asyncio.run()` call in the same process, as every test in `tests/test_postgres.py` makes)
    would hang forever acquiring a connection tied to a dead loop. One long-lived event loop
    (a real deployment's) still gets exactly one pool per `url`, same as before.

    `vector` has to exist as a type before any connection can register its codec, so a bare
    bootstrap connection creates the extension first -- `init=register_vector` on the pool itself
    would otherwise run on a database that doesn't have the type yet.
    """
    key = (id(asyncio.get_running_loop()), url)
    if key in _pools:
        return _pools[key]
    async with _pools_lock:
        if key not in _pools:
            from pgvector.asyncpg import register_vector

            bootstrap = await asyncpg.connect(url)
            try:
                await bootstrap.execute("CREATE EXTENSION IF NOT EXISTS vector")
            finally:
                await bootstrap.close()
            _pools[key] = await asyncpg.create_pool(url, init=register_vector)
    return _pools[key]


async def connect(url: str, ddl: str) -> asyncpg.Pool:
    """Return `url`'s shared pool, applying `ddl` (idempotent) first."""
    pool = await get_pool(url)
    async with pool.acquire() as conn:
        await conn.execute(ddl)
    return pool


__all__ = ["connect", "get_pool", "run_sync"]
