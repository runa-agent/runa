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
"""

from collections.abc import Sequence
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


__all__ = ["Database", "Queries", "Row"]
