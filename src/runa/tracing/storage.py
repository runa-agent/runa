"""tracing/storage.py: the `traces`/`spans` tables inside `runa.db`.

Same file, same connect-and-create-if-missing pattern as `eval/storage.py`, so a local app
accumulates one `runa.db` with no setup regardless of which of eval or tracing wrote to it first.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from runa.db.sqlite import DEFAULT_DB_PATH
from runa.db.sqlite import connect as _connect_db
from runa.tracing.spans import Span
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
CREATE TABLE IF NOT EXISTS {_SPANS_TABLE} (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL REFERENCES {_TRACES_TABLE}(id),
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
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    return _connect_db(db_path, _DDL)


def save_trace(trace: Trace, *, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Persist `trace` and every span in it, replacing any existing row with the same id."""
    with closing(_connect(db_path)) as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO {_TRACES_TABLE} "
            "(id, name, start_time, end_time, status, session_id, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                trace.id,
                trace.name,
                trace.start_time,
                trace.end_time,
                trace.status,
                trace.session_id,
                json.dumps(trace.metadata, default=str),
            ),
        )
        conn.executemany(
            f"""
            INSERT OR REPLACE INTO {_SPANS_TABLE}
                (id, trace_id, parent_id, name, type, start_time, end_time, status,
                 attributes_json, input, output, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    span.id,
                    span.trace_id,
                    span.parent_id,
                    span.name,
                    span.type,
                    span.start_time,
                    span.end_time,
                    span.status,
                    json.dumps(span.attributes, default=str),
                    _as_text(span.input),
                    _as_text(span.output),
                    span.error,
                )
                for span in trace.spans
            ],
        )
        conn.commit()


def _as_text(value: object) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _row_to_trace(conn: sqlite3.Connection, row: sqlite3.Row) -> Trace:
    span_rows = conn.execute(
        f"SELECT * FROM {_SPANS_TABLE} WHERE trace_id = ? ORDER BY start_time", (row["id"],)
    ).fetchall()
    spans = [
        Span(
            id=span_row["id"],
            trace_id=span_row["trace_id"],
            parent_id=span_row["parent_id"],
            name=span_row["name"],
            type=span_row["type"],
            start_time=span_row["start_time"],
            end_time=span_row["end_time"],
            status=span_row["status"],
            attributes=json.loads(span_row["attributes_json"]),
            input=span_row["input"],
            output=span_row["output"],
            error=span_row["error"],
        )
        for span_row in span_rows
    ]
    return Trace(
        id=row["id"],
        name=row["name"],
        start_time=row["start_time"],
        end_time=row["end_time"],
        session_id=row["session_id"],
        spans=spans,
        metadata=json.loads(row["metadata_json"]),
    )


def get_trace(trace_id: str, *, db_path: Path = DEFAULT_DB_PATH) -> Trace | None:
    """Look up one trace by id, with every span it has, or `None` if it isn't in `db_path`."""
    with closing(_connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(f"SELECT * FROM {_TRACES_TABLE} WHERE id = ?", (trace_id,)).fetchone()
        if row is None:
            return None
        return _row_to_trace(conn, row)


def list_traces(
    *,
    limit: int = 50,
    agent: str | None = None,
    status: str | None = None,
    session_id: str | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[Trace]:
    """Return the most recent `limit` traces, newest first, optionally filtered.

    `agent` matches `Trace.name` (the workflow name `Agent.run` sets to the agent's class name);
    `session_id` matches the `SessionABC` a session-backed run was passed, when it was passed one.
    """
    clauses, params = [], []
    if agent is not None:
        clauses.append("name = ?")
        params.append(agent)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if session_id is not None:
        clauses.append("session_id = ?")
        params.append(session_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with closing(_connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM {_TRACES_TABLE} {where} ORDER BY start_time DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [_row_to_trace(conn, row) for row in rows]


def get_recent_traces(*, limit: int = 20, db_path: Path = DEFAULT_DB_PATH) -> list[Trace]:
    """Return the `limit` most recently started traces, newest first."""
    return list_traces(limit=limit, db_path=db_path)


def get_errors(*, limit: int = 50, db_path: Path = DEFAULT_DB_PATH) -> list[Trace]:
    """Return the `limit` most recent traces whose `status` is `"error"`, newest first."""
    return list_traces(limit=limit, status="error", db_path=db_path)
