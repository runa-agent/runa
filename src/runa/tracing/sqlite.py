"""tracing/sqlite.py: `SQLiteTraceStore`, the local `TraceStore`.

`db/runa.db`'s `traces`/`spans` tables, in the same connect-and-create-if-missing file every
other local adapter writes to (`db/sqlite.py`), so a local app accumulates one `runa.db` with no
setup regardless of which concern wrote to it first.

The tables are `tracing/store.py`'s `TRACES`/`SPANS`, rendered here in SQLite's dialect. They
used to be a second `CREATE TABLE` string, kept in step with `tracing/postgres.py`'s by a
docstring asking for it: the Postgres side carried indexes this one lacked, which made `runa ui`
fast on a shared deployment and slow on a laptop for no reason anyone had stated.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

from runa.db import DEFAULT_DB_PATH
from runa.db.schema import SQLITE, ddl
from runa.db.sqlite import connect as _connect_db
from runa.tracing.store import (
    SPANS,
    TRACES,
    span_values,
    to_trace,
    trace_values,
)
from runa.tracing.traces import Trace

_DDL = ddl(SQLITE, TRACES, SPANS)


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
                f"INSERT OR REPLACE INTO {TRACES.name} ({', '.join(TRACES.column_names)}) "
                f"VALUES ({TRACES.placeholders(SQLITE)})",
                trace_values(trace),
            )
            conn.executemany(
                f"INSERT OR REPLACE INTO {SPANS.name} ({', '.join(SPANS.column_names)}) "
                f"VALUES ({SPANS.placeholders(SQLITE)})",
                [span_values(span) for span in trace.spans],
            )
            conn.commit()

    def get(self, trace_id: str) -> Trace | None:
        """Look up one trace by id, with every span it has, or `None` if this file has none."""
        with closing(self._connect()) as conn:
            row = conn.execute(f"SELECT * FROM {TRACES.name} WHERE id = ?", (trace_id,)).fetchone()
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
                f"SELECT * FROM {TRACES.name} {where} ORDER BY start_time DESC LIMIT ?",
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
            f"SELECT * FROM {SPANS.name} WHERE trace_id = ? ORDER BY start_time", (trace_id,)
        ).fetchall()


__all__ = ["SQLiteTraceStore"]
