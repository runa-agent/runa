"""tracing/postgres.py: traces in Postgres, so more than one replica shares one history.

The default `SQLiteExporter` writes to a `db/runa.db` beside the process. That is exactly right
for one machine and wrong for a deployment: three replicas keep three disjoint trace histories,
and `runa ui` can only ever show whichever one it happens to be looking at. This backend puts the
same two tables (`traces`/`spans`) in Postgres instead, and is picked up automatically whenever
`RUNA_POSTGRES_DSN` is set (see `runa.db.shared_dsn`).

Optional: part of the `runa[postgres]` extra, like `db/postgres.py`, which this builds on for its
pool and for the background loop that lets a synchronous exporter talk to `asyncpg`.
"""

from __future__ import annotations

import json
from typing import Any

from runa.db.postgres import DEFAULT_POSTGRES_DSN, _connect, run_sync
from runa.tracing.spans import Span
from runa.tracing.traces import Trace

_TRACES_TABLE = "traces"
_SPANS_TABLE = "spans"

_DDL = f"""
CREATE TABLE IF NOT EXISTS {_TRACES_TABLE} (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    start_time DOUBLE PRECISION NOT NULL,
    end_time DOUBLE PRECISION,
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
    start_time DOUBLE PRECISION NOT NULL,
    end_time DOUBLE PRECISION,
    status TEXT NOT NULL,
    attributes_json TEXT NOT NULL,
    input TEXT,
    output TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_{_SPANS_TABLE}_trace_id ON {_SPANS_TABLE} (trace_id);
"""


def _as_text(value: object) -> str | None:
    """Render a span's `input`/`output` as text, matching `tracing/storage.py`'s SQLite shape."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


async def _save(trace: Trace, dsn: str) -> None:
    pool = await _connect(dsn, _DDL)
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            f"""
            INSERT INTO {_TRACES_TABLE}
                (id, name, start_time, end_time, status, session_id, metadata_json)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name, start_time = EXCLUDED.start_time,
                end_time = EXCLUDED.end_time, status = EXCLUDED.status,
                session_id = EXCLUDED.session_id, metadata_json = EXCLUDED.metadata_json
            """,
            trace.id,
            trace.name,
            trace.start_time,
            trace.end_time,
            trace.status,
            trace.session_id,
            json.dumps(trace.metadata, default=str),
        )
        if trace.spans:
            await conn.executemany(
                f"""
                INSERT INTO {_SPANS_TABLE}
                    (id, trace_id, parent_id, name, type, start_time, end_time, status,
                     attributes_json, input, output, error)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (id) DO UPDATE SET
                    end_time = EXCLUDED.end_time, status = EXCLUDED.status,
                    attributes_json = EXCLUDED.attributes_json, input = EXCLUDED.input,
                    output = EXCLUDED.output, error = EXCLUDED.error
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


def save_trace(trace: Trace, *, dsn: str = DEFAULT_POSTGRES_DSN) -> None:
    """Persist `trace` and every span in it, replacing any row with the same id."""
    run_sync(_save(trace, dsn))


def _row_to_span(row: Any) -> Span:
    return Span(
        id=row["id"],
        trace_id=row["trace_id"],
        parent_id=row["parent_id"],
        name=row["name"],
        type=row["type"],
        start_time=row["start_time"],
        end_time=row["end_time"],
        status=row["status"],
        attributes=json.loads(row["attributes_json"]),
        input=row["input"],
        output=row["output"],
        error=row["error"],
    )


def _row_to_trace(row: Any, span_rows: list[Any]) -> Trace:
    trace = Trace(
        id=row["id"],
        name=row["name"],
        start_time=row["start_time"],
        end_time=row["end_time"],
        session_id=row["session_id"],
        metadata=json.loads(row["metadata_json"]),
    )
    trace.spans = [_row_to_span(span) for span in span_rows]
    return trace


async def _get(trace_id: str, dsn: str) -> Trace | None:
    pool = await _connect(dsn, _DDL)
    row = await pool.fetchrow(f"SELECT * FROM {_TRACES_TABLE} WHERE id = $1", trace_id)
    if row is None:
        return None
    spans = await pool.fetch(
        f"SELECT * FROM {_SPANS_TABLE} WHERE trace_id = $1 ORDER BY start_time", trace_id
    )
    return _row_to_trace(row, list(spans))


def get_trace(trace_id: str, *, dsn: str = DEFAULT_POSTGRES_DSN) -> Trace | None:
    """Look up one trace by id, with every span it has, or `None` if this database has none."""
    return run_sync(_get(trace_id, dsn))


async def _list(
    dsn: str, limit: int, agent: str | None, status: str | None, session_id: str | None
) -> list[Trace]:
    pool = await _connect(dsn, _DDL)
    clauses: list[str] = []
    params: list[Any] = []
    for column, value in (("name", agent), ("status", status), ("session_id", session_id)):
        if value is not None:
            params.append(value)
            clauses.append(f"{column} = ${len(params)}")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    rows = await pool.fetch(
        f"SELECT * FROM {_TRACES_TABLE} {where} ORDER BY start_time DESC LIMIT ${len(params)}",
        *params,
    )
    if not rows:
        return []
    ids = [row["id"] for row in rows]
    spans = await pool.fetch(
        f"SELECT * FROM {_SPANS_TABLE} WHERE trace_id = ANY($1::text[]) ORDER BY start_time", ids
    )
    by_trace: dict[str, list[Any]] = {trace_id: [] for trace_id in ids}
    for span in spans:
        by_trace[span["trace_id"]].append(span)
    return [_row_to_trace(row, by_trace[row["id"]]) for row in rows]


def list_traces(
    *,
    limit: int = 50,
    agent: str | None = None,
    status: str | None = None,
    session_id: str | None = None,
    dsn: str = DEFAULT_POSTGRES_DSN,
) -> list[Trace]:
    """Return the most recent `limit` traces, newest first, optionally filtered.

    Same filters and ordering as the SQLite backend, so callers cannot tell the two apart.
    """
    return run_sync(_list(dsn, limit, agent, status, session_id))


class PostgresExporter:
    """A `TraceExporter` that persists every finished trace to Postgres.

    Installed automatically when `RUNA_POSTGRES_DSN` is set, in place of the default
    `SQLiteExporter`; construct one explicitly to export to a database other than that one.
    """

    def __init__(self, dsn: str = DEFAULT_POSTGRES_DSN) -> None:
        """Store which Postgres database finished traces are written to."""
        self.dsn = dsn

    def export(self, trace: Trace) -> None:
        """Persist `trace`. Exceptions are caught by the caller; tracing never fails a run."""
        save_trace(trace, dsn=self.dsn)


__all__ = ["PostgresExporter", "get_trace", "list_traces", "save_trace"]
