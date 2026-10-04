"""db/sqlite.py: shared connect-and-create-if-missing plumbing for the local `db/runa.db`.

`db/pool.py`'s counterpart for the local side of `runa.db`'s one decision. Every SQLite adapter
(`session/sqlite.py`, `memory/sqlite.py`, `knowledge/sqlite.py`, `cache/sqlite.py`,
`tracing/storage.py`, `eval/storage.py`) owns a different set of tables in the same file; this
only opens the connection and applies each caller's DDL, so the file (and its parent `db/`, which
`sqlite3.connect` won't create on its own) gets created lazily regardless of which module writes
to it first.

Where that file lives is `runa.db.sqlite_path`'s answer, not this module's.
"""

import sqlite3
import struct
from pathlib import Path


def pack_vector(vector: list[float]) -> bytes:
    """Pack `vector` as the little-endian `float[]` blob a `vec0` column expects.

    Shared by `memory/sqlite.py`'s and `knowledge/sqlite.py`'s stores.
    """
    return struct.pack(f"{len(vector)}f", *vector)


def connect(db_path: Path, ddl: str, *, load_vec: bool = False) -> sqlite3.Connection:
    """Open `db_path`, creating its parent directory if needed, applying `ddl`.

    `load_vec=True` loads the `sqlite-vec` extension first, for callers whose `ddl` declares a
    `vec0` virtual table (`memory/sqlite.py`, `knowledge/sqlite.py`) -- everyone else pays
    nothing for it.
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
    conn.executescript(ddl)
    conn.commit()
    return conn
