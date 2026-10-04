"""tracing/postgres.py: `PostgresTraceStore`, traces in a shared database.

The local `SQLiteTraceStore` writes to a `db/runa.db` beside the process. That is exactly right
for one machine and wrong for a deployment: three replicas keep three disjoint trace histories,
and `runa ui` can only ever show whichever one it happens to be looking at. This backend puts the
same two tables (`traces`/`spans`) in Postgres instead, and `runa.db.traces()` picks it up
automatically whenever `RUNA_DATABASE_URL` is a `postgresql://` one.

Optional: part of the `runa[postgres]` extra, like `db/pool.py`, which this builds on for its
pool and for the background loop that lets a synchronous exporter talk to `asyncpg`. The row
marshalling is `tracing/store.py`'s, shared with the SQLite adapter; what is genuinely this
backend's own is the async driver, the `ON CONFLICT` upsert, and the batched span fetch.
"""

from typing import Any

from runa.db.pool import connect as _connect
from runa.db.pool import run_sync
from runa.tracing.config import StoreExporter
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


def _placeholders(columns: tuple[str, ...]) -> str:
    return ", ".join(f"${index}" for index in range(1, len(columns) + 1))


def _assignments(columns: tuple[str, ...], *, keep: str = "id") -> str:
    return ", ".join(f"{column} = EXCLUDED.{column}" for column in columns if column != keep)


class PostgresTraceStore:
    """The shared `TraceStore`: `traces`/`spans` in this deployment's Postgres database.

    `runa.db.traces()` builds this whenever `RUNA_DATABASE_URL` is a `postgresql://` one, so no
    call site has to name it. Constructing one by hand is the escape hatch for a database that is
    not this deployment's shared one.
    """

    def __init__(self, url: str) -> None:
        """Store which Postgres database this history lives in; connected lazily."""
        self.url = url

    def save(self, trace: Trace) -> None:
        """Persist `trace` and every span in it, replacing any existing one with the same id."""
        run_sync(self._save(trace))

    def get(self, trace_id: str) -> Trace | None:
        """Look up one trace by id, with every span it has, or `None` if this database has none."""
        return run_sync(self._get(trace_id))

    def list(
        self,
        *,
        limit: int = 50,
        agent: str | None = None,
        status: str | None = None,
        session_id: str | None = None,
    ) -> list[Trace]:
        """Return the most recent `limit` traces, newest first, optionally filtered."""
        return run_sync(self._list(limit, agent, status, session_id))

    async def _save(self, trace: Trace) -> None:
        pool = await _connect(self.url, _DDL)
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(
                f"""
                INSERT INTO {_TRACES_TABLE} ({", ".join(TRACE_COLUMNS)})
                VALUES ({_placeholders(TRACE_COLUMNS)})
                ON CONFLICT (id) DO UPDATE SET {_assignments(TRACE_COLUMNS)}
                """,
                *trace_values(trace),
            )
            if trace.spans:
                await conn.executemany(
                    f"""
                    INSERT INTO {_SPANS_TABLE} ({", ".join(SPAN_COLUMNS)})
                    VALUES ({_placeholders(SPAN_COLUMNS)})
                    ON CONFLICT (id) DO UPDATE SET {_assignments(SPAN_COLUMNS)}
                    """,
                    [span_values(span) for span in trace.spans],
                )

    async def _get(self, trace_id: str) -> Trace | None:
        pool = await _connect(self.url, _DDL)
        row = await pool.fetchrow(f"SELECT * FROM {_TRACES_TABLE} WHERE id = $1", trace_id)
        if row is None:
            return None
        spans = await pool.fetch(
            f"SELECT * FROM {_SPANS_TABLE} WHERE trace_id = $1 ORDER BY start_time", trace_id
        )
        return to_trace(row, list(spans))

    async def _list(
        self, limit: int, agent: str | None, status: str | None, session_id: str | None
    ) -> list[Trace]:
        pool = await _connect(self.url, _DDL)
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
        # One query for every listed trace's spans, where the SQLite adapter asks per trace: a
        # shared database is a network round trip, so N+1 of them is the cost worth avoiding.
        ids = [row["id"] for row in rows]
        spans = await pool.fetch(
            f"SELECT * FROM {_SPANS_TABLE} WHERE trace_id = ANY($1::text[]) ORDER BY start_time",
            ids,
        )
        by_trace: dict[str, list[Any]] = {trace_id: [] for trace_id in ids}
        for span in spans:
            by_trace[span["trace_id"]].append(span)
        return [to_trace(row, by_trace[row["id"]]) for row in rows]


class PostgresExporter(StoreExporter):
    """A `TraceExporter` that persists every finished trace to Postgres.

    `_default_exporters` builds a plain `StoreExporter` over whatever `runa.db.traces()` resolved,
    so this name is for an app that wants to export to a database other than this deployment's.
    """

    def __init__(self, url: str) -> None:
        """Export to `url`'s `traces`/`spans` tables."""
        super().__init__(PostgresTraceStore(url))


__all__ = ["PostgresExporter", "PostgresTraceStore"]
