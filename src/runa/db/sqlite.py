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

Where that file lives is `runa.db.sqlite_path`'s answer, not this module's.
"""

import sqlite3
import struct
from pathlib import Path

from runa.db.schema import SQLITE, Declaration, ddl, reconcile


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
    for statement in reconcile(SQLITE, conn.execute(SQLITE.columns_query), *tables):
        conn.execute(statement)
    conn.commit()
    return conn
