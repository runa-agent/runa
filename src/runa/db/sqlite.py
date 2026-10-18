"""db/sqlite.py: shared connect-and-create-if-missing plumbing for the local `db/runa.db`.

`db/pool.py`'s counterpart for the local side of `runa.db`'s one decision. Every SQLite adapter
(`session/sqlite.py`, `cache/sqlite.py`, `tracing/sqlite.py`, `eval/sqlite.py`, and the two
vector stores through `db/vectors/sqlite.py`) owns a different set of tables in the same file; this
only opens the connection and brings each caller's tables up to their declaration, so the file
(and its parent `db/`, which `sqlite3.connect` won't create on its own) gets created lazily
regardless of which module writes to it first.

A local file is the side of that decision that outlives releases: a shared Postgres is created by
whichever replica connects first, where `db/runa.db` can be older than the release now opening it.
Which is why what this applies is `db/schema.py`'s two halves, `ddl` and `reconcile`, and why an
adapter hands over its `Table`s rather than a rendered script. Once per file per process, like
`db/pool.py`'s once per database per process, and for the same reason: whether a file predates a
column is a question about the release that created it, so a connection per call (which is what
every adapter here opens) is not a reason to ask it again. `_pending` is that bookkeeping, and
`read` below is the one caller that stays outside it, for a reason of its own.

`read` is the other half of the same decision, for the queries that only read: creating a file to
answer "no history yet" is a side effect no listing should have, and which tables a database has
is one question with one answer, not four adapters each guessing from a failed query.

Where that file lives is `runa.db.sqlite_path`'s answer, not this module's.
"""

import sqlite3
import struct
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from runa.db.schema import SQLITE, Declaration, Table, ddl, reconcile

_applied: set[tuple[int, int, tuple[Declaration, ...]]] = set()
"""The `(file, declarations)` pairs this process has already brought up to date; see `_pending`."""
_applied_lock = threading.Lock()


def pack_vector(vector: list[float]) -> bytes:
    """Pack `vector` as the little-endian `float[]` blob a `vec0` column expects.

    `db/vectors/sqlite.py` is the only caller, on behalf of both local vector stores.
    """
    return struct.pack(f"{len(vector)}f", *vector)


def connect(db_path: Path, *tables: Declaration, load_vec: bool = False) -> sqlite3.Connection:
    """Open `db_path`, creating its parent directory if needed, and apply `tables`.

    Apply means create what's missing and then add the columns a declaration has grown since this
    file was created -- `db/schema.py`'s `ddl` and `reconcile`, in that order, once per file per
    process rather than on every connect; see `_pending`.

    `load_vec=True` loads the `sqlite-vec` extension first, for callers passing the raw SQL of a
    `vec0` virtual table (`db/vectors/sqlite.py`) -- everyone else pays nothing for it.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
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


@contextmanager
def read(db_path: Path, *tables: Table) -> Iterator[sqlite3.Connection | None]:
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

    Reading is a local-file question only. A shared Postgres has no file to leave behind, and
    `db/pool.py` creates its schema once per process rather than per query.
    """
    if not db_path.exists():
        yield None
        return
    conn = sqlite3.connect(db_path)
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


def _pending(db_path: Path, tables: tuple[Declaration, ...]) -> bool:
    """Whether `tables` still have to be applied to `db_path` by this process.

    `ddl` and `reconcile` answer a question about releases, not about queries: whether this file
    predates a table, or a column a declaration has since grown. That answer cannot change while
    a process runs, so re-answering it per connection buys nothing and costs the four-statement
    `CREATE TABLE IF NOT EXISTS` script plus a whole-database `sqlite_master`/`pragma_table_info`
    scan -- on every connection, which here means every call, three of them in one
    session-backed turn.

    Keyed on the file's identity rather than its name: a database deleted underneath a live
    process (a test, rather than a deployment) comes back with a new inode, so the next connect
    creates its tables again instead of trusting what this process did to the old file. Keyed on
    the declarations too, since a file holds several adapters' tables and each applies its own.

    Two threads can both be told yes for the same pair and both apply it. That costs a duplicate
    script, not a wrong database: every statement either side runs is `IF NOT EXISTS` or an
    `ALTER TABLE` for a column neither has yet.
    """
    with _applied_lock:
        return _key(db_path, tables) not in _applied


def _remember(db_path: Path, tables: tuple[Declaration, ...]) -> None:
    """Record that `tables` are now applied to `db_path`, so the next connect can skip it.

    Called after the statements succeed, never before: `reconcile` can raise `SchemaOutOfDate`,
    and a database a human still has to migrate is not one this process is done with.
    """
    with _applied_lock:
        _applied.add(_key(db_path, tables))


def _key(
    db_path: Path, tables: tuple[Declaration, ...]
) -> tuple[int, int, tuple[Declaration, ...]]:
    """This file's identity (device and inode) paired with the declarations applied to it."""
    info = db_path.stat()
    return (info.st_dev, info.st_ino, tables)


def _table_names(conn: sqlite3.Connection) -> set[str]:
    """Every table this database actually has."""
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    return {row[0] for row in rows}


def _reconcile(conn: sqlite3.Connection, *tables: Declaration) -> None:
    """Add the columns `tables`'s declarations have grown since this database was created."""
    for statement in reconcile(SQLITE, conn.execute(SQLITE.columns_query), *tables):
        conn.execute(statement)
