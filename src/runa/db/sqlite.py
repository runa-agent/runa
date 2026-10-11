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
adapter hands over its `Table`s rather than a rendered script.

`read` is the other half of the same decision, for the queries that only read: creating a file to
answer "no history yet" is a side effect no listing should have, and which tables a database has
is one question with one answer, not four adapters each guessing from a failed query.

Where that file lives is `runa.db.sqlite_path`'s answer, not this module's.
"""

import sqlite3
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from runa.db.schema import SQLITE, Declaration, Table, ddl, reconcile


def pack_vector(vector: list[float]) -> bytes:
    """Pack `vector` as the little-endian `float[]` blob a `vec0` column expects.

    `db/vectors/sqlite.py` is the only caller, on behalf of both local vector stores.
    """
    return struct.pack(f"{len(vector)}f", *vector)


def connect(db_path: Path, *tables: Declaration, load_vec: bool = False) -> sqlite3.Connection:
    """Open `db_path`, creating its parent directory if needed, and apply `tables`.

    Apply means create what's missing and then add the columns a declaration has grown since this
    file was created -- `db/schema.py`'s `ddl` and `reconcile`, in that order, on every connect.

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
    conn.executescript(ddl(SQLITE, *tables))
    _reconcile(conn, *tables)
    conn.commit()
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
    failed query. `sqlite3.OperationalError` around the `SELECT` would say the same thing about a
    locked database, a corrupt file, a renamed column and a malformed statement, which is how
    `runa chat --list` once reported no sessions while `--show` printed one of their transcripts.
    Those all still raise.

    A database that does exist is brought up to its declaration like any other connection: a read
    selects the columns the current release declares, so `reconcile`'s additive `ALTER TABLE`s
    (and `SchemaOutOfDate` for what it can't add) belong here as much as on a write. Rows come
    back as `sqlite3.Row`, which is what every read path above wants.

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


def _table_names(conn: sqlite3.Connection) -> set[str]:
    """Every table this database actually has."""
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    return {row[0] for row in rows}


def _reconcile(conn: sqlite3.Connection, *tables: Declaration) -> None:
    """Add the columns `tables`'s declarations have grown since this database was created."""
    for statement in reconcile(SQLITE, conn.execute(SQLITE.columns_query), *tables):
        conn.execute(statement)
