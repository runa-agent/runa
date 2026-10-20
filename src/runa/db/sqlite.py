"""db/sqlite.py: shared connect-and-create-if-missing plumbing for the local SQLite database.

Every SQLite user (each concern through `SQLiteDatabase` below, and the two vector stores through
`db/vectors/sqlite.py`) owns a different set of tables in the same database. This opens the
connection, creating the file and its parent `db/` lazily, and brings each caller's `Table`s up to
their declaration with `db/schema.py`'s `ddl` and `reconcile`: a local file can be older than the
release opening it. That happens once per database per process (`_pending`), however many
connections the adapters open.

That database is a `Location`: a file, or the `MemoryDatabase` `memory://` keeps in this
process. The two are the same SQL through the same adapters; only where the pages live differs.

`run` and `run_read` are how an `async def` reaches `sqlite3`, which blocks: the body runs on a
worker thread, with a connection opened and closed there, so nothing above decides for itself
whether to keep the event loop free. SQLite has one writer per file, so concurrent writers wait on
its busy timeout in a worker thread instead of on the loop.

`read` serves queries that only read: it never creates the file or a table, so a listing over a
database that doesn't exist yet is empty rather than a side effect.

`SQLiteDatabase` is `db/handle.py`'s `Database` over all of the above: a concern's queries,
written once in Postgres's numbered style, run here.

Which database a deployment has is `runa.db.sqlite_database`'s answer, not this module's.
"""

import asyncio
import re
import sqlite3
import struct
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Hashable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, closing, contextmanager
from pathlib import Path
from typing import Any

from runa.db.handle import Queries, Row
from runa.db.schema import SQLITE, Declaration, Table, ddl, reconcile


class MemoryDatabase:
    """One SQLite database held in this process's memory, which every connection to it shares.

    What `memory://` is: the database the local adapters already speak, with no file under it.
    It lives until `close`, holding one connection open for that purpose alone. Each instance is
    a database of its own, never one an earlier instance left behind.

    It is SQLite's `memdb` VFS rather than a `mode=memory&cache=shared` URI: a shared cache
    locks per table and fails a concurrent writer at once, where `memdb` locks like a file does,
    so the adapters' worker threads wait out each other's writes under the usual busy timeout.
    """

    def __init__(self) -> None:
        """Create the database, empty."""
        self.uri = f"file:/runa-{uuid.uuid4().hex}?vfs=memdb"
        self._anchor = sqlite3.connect(self.uri, uri=True, check_same_thread=False)

    def close(self) -> None:
        """Drop the database and everything in it, once no other connection has it open."""
        self._anchor.close()


Location = Path | MemoryDatabase
"""Where a local adapter's tables live: a file, or this process's `MemoryDatabase`."""

_applied: set[tuple[Hashable, tuple[Declaration, ...]]] = set()
"""The `(database, declarations)` pairs this process already brought up to date; see `_pending`."""
_applied_lock = threading.Lock()


def pack_vector(vector: list[float]) -> bytes:
    """Pack `vector` as the little-endian `float[]` blob a `vec0` column expects.

    `db/vectors/sqlite.py` is the only caller, on behalf of both local vector stores.
    """
    return struct.pack(f"{len(vector)}f", *vector)


def connect(db_path: Location, *tables: Declaration, load_vec: bool = False) -> sqlite3.Connection:
    """Open `db_path`, creating a file's parent directory if needed, and apply `tables`.

    Apply means create what's missing and then add the columns a declaration has grown since this
    database was created -- `db/schema.py`'s `ddl` and `reconcile`, in that order, once per
    database per process rather than on every connect; see `_pending`.

    `load_vec=True` loads the `sqlite-vec` extension first, for callers passing the raw SQL of a
    `vec0` virtual table (`db/vectors/sqlite.py`) -- everyone else pays nothing for it.
    """
    if isinstance(db_path, MemoryDatabase):
        conn = sqlite3.connect(db_path.uri, uri=True)
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        created = not db_path.exists()
        conn = sqlite3.connect(db_path)
        if created:
            _forget(db_path)
    # Off by default in SQLite, and per-connection: without it every `REFERENCES ... ON DELETE
    # CASCADE` in an adapter's DDL is decorative, and deleting a trace or an eval run leaves its
    # spans and cases behind as orphans.
    conn.execute("PRAGMA foreign_keys = ON")
    if load_vec:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    if _pending(db_path, tables):
        conn.executescript(ddl(SQLITE, *tables))
        _reconcile(conn, *tables)
        conn.commit()
        _remember(db_path, tables)
    return conn


async def run[T](
    db_path: Location,
    tables: tuple[Declaration, ...],
    body: Callable[[sqlite3.Connection], T],
    *,
    load_vec: bool = False,
) -> T:
    """Run `body` on a worker thread with a `connect`ed connection, and return what it returns.

    The connection is opened and closed on that thread, so no `sqlite3` object is touched from
    two. `body` commits for itself. A thread hop costs a few hundred microseconds against a local
    file's millisecond; awaiting a body that never yields would make one turn's `fsync` every
    concurrent turn's latency.
    """

    def work() -> T:
        with closing(connect(db_path, *tables, load_vec=load_vec)) as conn:
            return body(conn)

    return await asyncio.to_thread(work)


async def run_read[T](
    db_path: Location, tables: tuple[Table, ...], body: Callable[[sqlite3.Connection | None], T]
) -> T:
    """`run` for `read`: `body` gets the connection `read` yields, or `None` when absent."""

    def work() -> T:
        with read(db_path, *tables) as conn:
            return body(conn)

    return await asyncio.to_thread(work)


@contextmanager
def read(db_path: Location, *tables: Table) -> Iterator[sqlite3.Connection | None]:
    """Open `db_path` to read `tables`, or yield `None` when they aren't there to be read.

    `connect`'s read-side counterpart, for the queries behind `runa chat --list/--show`, `runa
    ui`'s pages and a cache lookup. It creates nothing -- not the parent directory, not the file,
    not a missing table -- so answering "no history yet" in a project that has never run an agent
    doesn't leave a `db/runa.db` behind as the side effect of a read. `None` is that answer: the
    file isn't there, or it is and this concern's tables aren't (another concern wrote it first).
    Every caller turns it into whatever its own empty result is.

    Absence is the one condition answered that way, and it is tested rather than inferred from a
    failed query: a locked database, a corrupt file or a malformed statement still raises.

    A database that does exist is brought up to its declaration like any other connection: a read
    selects the columns the current release declares, so `reconcile`'s additive `ALTER TABLE`s
    (and `SchemaOutOfDate` for what it can't add) belong here as much as on a write. Per query
    here, not once per file like `connect`: a read is a CLI command or a dashboard page rather
    than a step in a run, and reconciling it every time is what turns a column that went missing
    underneath a live process into the operator error above instead of a bare `no such column`.
    Rows come back as `sqlite3.Row`, which is what every read path above wants.

    Reading is a local question only. A shared Postgres has no file to leave behind, and
    `db/pool.py` creates its schema once per process rather than per query. A `MemoryDatabase`
    has no file either, so for it only the tables are asked about.
    """
    if isinstance(db_path, MemoryDatabase):
        conn = sqlite3.connect(db_path.uri, uri=True)
    elif db_path.exists():
        conn = sqlite3.connect(db_path)
    else:
        yield None
        return
    try:
        conn.row_factory = sqlite3.Row
        if not {table.name for table in tables} <= _table_names(conn):
            yield None
            return
        _reconcile(conn, *tables)
        conn.commit()
        yield conn
    finally:
        conn.close()


def _pending(db_path: Location, tables: tuple[Declaration, ...]) -> bool:
    """Whether `tables` still have to be applied to `db_path` by this process.

    `ddl` and `reconcile` answer a question about releases, not about queries: whether this file
    predates a table, or a column a declaration has since grown. That answer cannot change while
    a process runs, so re-answering it per connection buys nothing and costs the four-statement
    `CREATE TABLE IF NOT EXISTS` script plus a whole-database `sqlite_master`/`pragma_table_info`
    scan -- on every connection, which here means every call, three of them in one
    session-backed turn.

    Keyed on a file's identity rather than its name, so a database replaced underneath a live
    process isn't trusted for what this process did to the old one. An inode number alone can't
    promise that, since Linux hands a freed one straight to the next file created: a connect that
    creates the file forgets its identity first (`_forget`). A `MemoryDatabase` is keyed on its
    URI, which no later one reuses. Keyed on the declarations too, since a database holds several
    adapters' tables and each applies its own.

    Two threads can both be told yes for the same pair and both apply it. That costs a duplicate
    script, not a wrong database: every statement either side runs is `IF NOT EXISTS` or an
    `ALTER TABLE` for a column neither has yet.
    """
    with _applied_lock:
        return _key(db_path, tables) not in _applied


def _remember(db_path: Location, tables: tuple[Declaration, ...]) -> None:
    """Record that `tables` are now applied to `db_path`, so the next connect can skip it.

    Called after the statements succeed, never before: `reconcile` can raise `SchemaOutOfDate`,
    and a database a human still has to migrate is not one this process is done with.
    """
    with _applied_lock:
        _applied.add(_key(db_path, tables))


def _forget(db_path: Path) -> None:
    """Drop everything recorded for `db_path`'s identity, which a file just created can't have."""
    identity = _identity(db_path)
    with _applied_lock:
        _applied.difference_update(key for key in list(_applied) if key[0] == identity)


def _key(
    db_path: Location, tables: tuple[Declaration, ...]
) -> tuple[Hashable, tuple[Declaration, ...]]:
    """This database's identity paired with the declarations applied to it."""
    return (_identity(db_path), tables)


def _identity(db_path: Location) -> Hashable:
    """A file's device and inode, or a `MemoryDatabase`'s URI."""
    if isinstance(db_path, MemoryDatabase):
        return db_path.uri
    info = db_path.stat()
    return (info.st_dev, info.st_ino)


def _table_names(conn: sqlite3.Connection) -> set[str]:
    """Every table this database actually has."""
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    return {row[0] for row in rows}


def _reconcile(conn: sqlite3.Connection, *tables: Declaration) -> None:
    """Add the columns `tables`'s declarations have grown since this database was created."""
    for statement in reconcile(SQLITE, conn.execute(SQLITE.columns_query), *tables):
        conn.execute(statement)


_PARAMETER = re.compile(r"\$(\d+)")


def _numbered(sql: str) -> str:
    """`sql` with Postgres's `$1, $2, ...` rewritten as SQLite's `?1, ?2, ...`.

    SQLite numbers the same way, so a value is still bound by its index and may be used twice.
    """
    return _PARAMETER.sub(r"?\1", sql)


def _fetch(conn: sqlite3.Connection, sql: str, args: tuple[Any, ...]) -> list[Row]:
    return conn.execute(_numbered(sql), args).fetchall()


def _execute(conn: sqlite3.Connection, sql: str, args: tuple[Any, ...]) -> int:
    return conn.execute(_numbered(sql), args).rowcount


def _executemany(conn: sqlite3.Connection, sql: str, rows: Sequence[Sequence[Any]]) -> None:
    conn.executemany(_numbered(sql), rows)


class SQLiteDatabase:
    """`Database` over a local file, or the process's `MemoryDatabase`.

    Every call is handed to a worker thread (`run`), which opens a connection, runs the statement
    and closes it, so no `sqlite3` object is touched from two threads. A read goes through `read`
    and so creates nothing; a write goes through `connect`, which creates the file and brings
    `tables` up to their declaration.
    """

    def __init__(self, location: Location, *tables: Table) -> None:
        """Store which file or memory database this handle queries, and the tables it owns."""
        self.location = location
        self.tables = tables

    async def fetch(self, sql: str, *args: Any) -> list[Row]:
        """Every row `sql` selects; none when the database or its tables don't exist yet."""
        return (
            await run_read(self.location, self.tables, lambda conn: self._rows(conn, sql, args))
            or []
        )

    async def fetchrow(self, sql: str, *args: Any) -> Row | None:
        """The first row `sql` selects, or `None`."""
        rows = await self.fetch(sql, *args)
        return rows[0] if rows else None

    async def fetchval(self, sql: str, *args: Any) -> Any:
        """The first column of the first row `sql` selects, or `None`."""
        row = await self.fetchrow(sql, *args)
        return row[0] if row is not None else None

    async def execute(self, sql: str, *args: Any) -> int:
        """Run `sql` in a transaction of its own and return how many rows it changed."""

        def body(conn: sqlite3.Connection) -> int:
            changed = _execute(conn, sql, args)
            conn.commit()
            return changed

        return await run(self.location, self.tables, body)

    async def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        """Run `sql` once per row of `rows`, all in one transaction."""

        def body(conn: sqlite3.Connection) -> None:
            _executemany(conn, sql, rows)
            conn.commit()

        await run(self.location, self.tables, body)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Queries]:
        """Hold one connection open across awaits, committing when the block ends cleanly.

        The connection lives on a thread of its own, opened, used and closed there, so awaiting
        between statements never hands it to a different one.
        """
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="runa-sqlite")
        loop = asyncio.get_running_loop()
        conn = await loop.run_in_executor(executor, self._open)
        try:
            yield _Transaction(conn, executor)
            await loop.run_in_executor(executor, conn.commit)
        except BaseException:
            await loop.run_in_executor(executor, conn.rollback)
            raise
        finally:
            await loop.run_in_executor(executor, conn.close)
            executor.shutdown()

    def _open(self) -> sqlite3.Connection:
        conn = connect(self.location, *self.tables)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _rows(conn: sqlite3.Connection | None, sql: str, args: tuple[Any, ...]) -> list[Row] | None:
        return _fetch(conn, sql, args) if conn is not None else None


class _Transaction:
    """The statements of one `SQLiteDatabase.transaction`, each run on its connection's thread."""

    def __init__(self, conn: sqlite3.Connection, executor: ThreadPoolExecutor) -> None:
        self._conn = conn
        self._executor = executor

    async def fetch(self, sql: str, *args: Any) -> list[Row]:
        """Every row `sql` selects."""
        return await self._call(_fetch, sql, args)

    async def fetchrow(self, sql: str, *args: Any) -> Row | None:
        """The first row `sql` selects, or `None`."""
        rows = await self.fetch(sql, *args)
        return rows[0] if rows else None

    async def fetchval(self, sql: str, *args: Any) -> Any:
        """The first column of the first row `sql` selects, or `None`."""
        row = await self.fetchrow(sql, *args)
        return row[0] if row is not None else None

    async def execute(self, sql: str, *args: Any) -> int:
        """Run `sql` and return how many rows it changed."""
        return await self._call(_execute, sql, args)

    async def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        """Run `sql` once per row of `rows`."""
        await self._call(_executemany, sql, rows)

    async def _call[T](
        self, statement: Callable[[sqlite3.Connection, str, Any], T], sql: str, args: Any
    ) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, statement, self._conn, sql, args)
