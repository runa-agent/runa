"""tracing/sqlite.py: `SQLiteTraceStore`, the local `TraceStore`.

`db/runa.db`'s `traces`/`spans` tables, in the same connect-and-create-if-missing file every
other local adapter writes to (`db/sqlite.py`), so a local app accumulates one `runa.db` with no
setup regardless of which concern wrote to it first.

Same tables, same columns and the same three indexes as `tracing/postgres.py`. Two `CREATE TABLE`
strings rather than one because `REAL` is not `DOUBLE PRECISION`, not because the schemas are
allowed to drift: the Postgres side used to carry indexes this one lacked, which made `runa ui`
fast on a shared deployment and slow on a laptop for no reason anyone had stated.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

from runa.db import DEFAULT_DB_PATH
from runa.db.sqlite import connect as _connect_db
from runa.tracing.store import (
    SPAN_COLUMNS,
    TRACE_COLUMNS,
    span_values,
    to_trace,
    trace_values,
)
from runa.tracing.traces import Trace

_TRACES_TABLE = "traces"
_SPANS_TABLE = "spans"

_DDL = f"""
CREATE TABLE IF NOT EXISTS {_TRACES_TABLE} (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL,
    status TEXT NOT NULL,
    session_id TEXT,
    metadata_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_{_TRACES_TABLE}_session_id ON {_TRACES_TABLE} (session_id);
CREATE INDEX IF NOT EXISTS idx_{_TRACES_TABLE}_start_time ON {_TRACES_TABLE} (start_time DESC);
CREATE TABLE IF NOT EXISTS {_SPANS_TABLE} (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL REFERENCES {_TRACES_TABLE}(id) ON DELETE CASCADE,
    parent_id TEXT,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL,
    status TEXT NOT NULL,
    attributes_json TEXT NOT NULL,
    input TEXT,
    output TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_{_SPANS_TABLE}_trace_id ON {_SPANS_TABLE} (trace_id);
"""

_TRACE_PLACEHOLDERS = ", ".join("?" * len(TRACE_COLUMNS))
_SPAN_PLACEHOLDERS = ", ".join("?" * len(SPAN_COLUMNS))


class SQLiteTraceStore:
    """The local `TraceStore`: `db/runa.db`'s `traces`/`spans` tables."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        """Store which SQLite file this history lives in; the tables are created on first use."""
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = _connect_db(self.db_path, _DDL)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, trace: Trace) -> None:
        """Persist `trace` and every span in it, replacing any existing one with the same id."""
        with closing(self._connect()) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {_TRACES_TABLE} ({', '.join(TRACE_COLUMNS)}) "
                f"VALUES ({_TRACE_PLACEHOLDERS})",
                trace_values(trace),
            )
            conn.executemany(
                f"INSERT OR REPLACE INTO {_SPANS_TABLE} ({', '.join(SPAN_COLUMNS)}) "
                f"VALUES ({_SPAN_PLACEHOLDERS})",
                [span_values(span) for span in trace.spans],
            )
            conn.commit()

    def get(self, trace_id: str) -> Trace | None:
        """Look up one trace by id, with every span it has, or `None` if this file has none."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"SELECT * FROM {_TRACES_TABLE} WHERE id = ?", (trace_id,)
            ).fetchone()
            if row is None:
                return None
            return to_trace(row, self._span_rows(conn, row["id"]))

    def list(
        self,
        *,
        limit: int = 50,
        agent: str | None = None,
        status: str | None = None,
        session_id: str | None = None,
    ) -> list[Trace]:
        """Return the most recent `limit` traces, newest first, optionally filtered."""
        clauses: list[str] = []
        params: list[object] = []
        for column, value in (("name", agent), ("status", status), ("session_id", session_id)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT * FROM {_TRACES_TABLE} {where} ORDER BY start_time DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
            return [to_trace(row, self._span_rows(conn, row["id"])) for row in rows]

    def _span_rows(self, conn: sqlite3.Connection, trace_id: str) -> list[sqlite3.Row]:
        """One trace's span rows, oldest first.

        A query per trace, where `tracing/postgres.py` fetches every listed trace's spans in one
        `trace_id = ANY($1)`. That difference is the backend's, not the interface's: a round trip
        to a local file is cheap enough that batching here would buy an `IN (...)` builder nobody
        can see from outside the store.
        """
        return conn.execute(
            f"SELECT * FROM {_SPANS_TABLE} WHERE trace_id = ? ORDER BY start_time", (trace_id,)
        ).fetchall()


__all__ = ["SQLiteTraceStore"]
