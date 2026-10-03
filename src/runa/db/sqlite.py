"""db/sqlite.py: shared connect-and-create-if-missing plumbing for `db/runa.db`.

`tracing/storage.py` and `eval/storage.py` each own a different set of tables in the
same file; this only opens the connection and applies each caller's DDL, so the file (and its
parent `db/`, which `sqlite3.connect` won't create on its own) gets created lazily regardless of
which module writes to it first.
"""

import sqlite3
import struct
from pathlib import Path

DEFAULT_DB_PATH = Path("db/runa.db")


def pack_vector(vector: list[float]) -> bytes:
    """Pack `vector` as the little-endian `float[]` blob a `vec0` column expects.

    Shared by `runa.memory.SQLiteMemoryStore` and `runa.knowledge.SQLiteKnowledgeStore`.
    """
    return struct.pack(f"{len(vector)}f", *vector)


def connect(db_path: Path, ddl: str, *, load_vec: bool = False) -> sqlite3.Connection:
    """Open `db_path`, creating its parent directory if needed, applying `ddl`.

    `load_vec=True` loads the `sqlite-vec` extension first, for callers whose `ddl` declares a
    `vec0` virtual table (`runa.memory`/`runa.knowledge`) -- everyone else pays nothing for it.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    if load_vec:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    conn.executescript(ddl)
    conn.commit()
    return conn
