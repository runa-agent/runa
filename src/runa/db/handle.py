"""db/handle.py: `Database`, the one async handle every SQL concern queries through.

A concern (sessions, the cache, paused runs, traces, eval history) writes its queries once, as
Postgres-numbered SQL (`$1, $2, ...`), and holds a `Database`. Which engine answers is the handle's
business: `db/sqlite.py`'s `SQLiteDatabase` runs each call on a worker thread and rewrites `$n` to
the `?n` SQLite takes, and `db/pool.py`'s `PostgresDatabase` is the asyncpg pool. `runa.db.database`
hands out whichever the environment names.

That makes the SQL the intersection of the two engines: `ON CONFLICT ... DO UPDATE`, `RETURNING`,
`COALESCE`, `CURRENT_TIMESTAMP`, `IN (...)`. A query that needs a phrase only one of them has
belongs in an adapter of its own, as the two vector stores are. A `$n` inside a string literal is
not a parameter, but the SQLite rewrite cannot tell, so a literal never contains one.

A call outside `transaction()` is its own transaction. Reads never create anything: a database
that isn't there yet answers a `fetch` with no rows, and the first write is what makes it.

`run_sync` and `sync` are how a synchronous caller (`TraceExporter.export`, `runa traces`, `runa
ui`) reaches a method written against this handle.
"""

import asyncio
import functools
import threading
from collections.abc import Callable, Coroutine, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol


class Row(Protocol):
    """One result row: subscriptable by column name, or by position for a lone column."""

    def __getitem__(self, key: str | int, /) -> Any:
        """The value in column `key`."""
        ...


class Queries(Protocol):
    """The five statements a concern is written in; `Database` and a `transaction()` both answer."""

    async def fetch(self, sql: str, *args: Any) -> list[Row]:
        """Every row `sql` selects, in the order it selects them."""
        ...

    async def fetchrow(self, sql: str, *args: Any) -> Row | None:
        """The first row `sql` selects, or `None` when it selects none."""
        ...

    async def fetchval(self, sql: str, *args: Any) -> Any:
        """The first column of the first row `sql` selects, or `None` when it selects none."""
        ...

    async def execute(self, sql: str, *args: Any) -> int:
        """Run `sql` and return how many rows it changed."""
        ...

    async def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        """Run `sql` once per row of `rows`."""
        ...


class Database(Queries, Protocol):
    """An async SQL database with its concern's tables already declared to it.

    The tables are the handle's, not each call's: the first write creates them and brings an
    older database up to the declaration (`db/schema.py`), once, so a concern never opens a
    connection or applies a schema itself.
    """

    def transaction(self) -> AbstractAsyncContextManager[Queries]:
        """Run the statements made on the yielded handle together: all commit, or none do."""
        ...


_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _background_loop() -> asyncio.AbstractEventLoop:
    """A daemon event loop for the synchronous callers that still need a `Database`.

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
            threading.Thread(target=_loop.run_forever, name="runa-database", daemon=True).start()
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
    `runa ui` and `TraceExporter.export` reach it from a thread with no loop on it) while a
    `Database` is not; this pairs each async body with `run_sync`.
    """

    @functools.wraps(method)
    def synchronous(*args: P.args, **kwargs: P.kwargs) -> T:
        return run_sync(method(*args, **kwargs))

    return synchronous


__all__ = ["Database", "Queries", "Row", "run_sync", "sync"]
