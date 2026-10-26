"""tracing/store.py: `TraceStore`, where finished traces go and where `runa ui` reads them from.

The `traces`/`spans` tables in whichever database the deployment has, queried through
`db/handle.py`'s `Database`. Local, that is a `db/runa.db` beside the process: exactly right for
one machine and wrong for a deployment, where three replicas keep three disjoint trace histories
and `runa ui` can only ever show whichever one it happens to be looking at. Which database it is
is `runa.db.traces(...)`'s decision, asked once, so `runa traces`, `runa ui` and the exporter hold
a `TraceStore` and get whichever history the deployment actually has, and the exporter a
deployment installs and the store its traces land in can't disagree.

Everything a row is lives here: `TRACES`/`SPANS` state its columns (`db/schema.py` renders them
for whichever dialect the database speaks), the store's SQL names them, and the marshalling below
the store maps them to and from a `Trace`, in the column order those `INSERT`s depend on.
`listing_clause` is the whole of `list`'s filter, order and cap: which column `agent` means and
how ties break are this module's rule.
"""

import builtins
import json
from dataclasses import asdict
from typing import Any

from pydantic import TypeAdapter

from runa._types import Usage
from runa.db.handle import Database, Row
from runa.db.schema import Column, Index, Table
from runa.exceptions import OperatorError
from runa.tracing.spans import Span
from runa.tracing.traces import Trace

TRACES = Table(
    "traces",
    columns=(
        Column("id", "text", primary_key=True),
        Column("name", "text"),
        Column("start_time", "float"),
        Column("end_time", "float", null=True),
        Column("status", "text"),
        Column("session_id", "text", null=True),
        Column("metadata_json", "text"),
    ),
    indexes=(
        Index("session_id", "session_id"),
        Index("start_time", "start_time DESC"),
    ),
)

SPANS = Table(
    "spans",
    columns=(
        Column("id", "text", primary_key=True),
        Column("trace_id", "text", references=f"{TRACES.name}(id)"),
        Column("parent_id", "text", null=True),
        Column("name", "text"),
        Column("type", "text"),
        Column("start_time", "float"),
        Column("end_time", "float", null=True),
        Column("status", "text"),
        Column("attributes_json", "text"),
        Column("input", "text", null=True),
        Column("output", "text", null=True),
        Column("error", "text", null=True),
        Column("usage_json", "text", null=True),
    ),
    indexes=(Index("trace_id", "trace_id"),),
)

_USAGE = TypeAdapter(Usage)


class TraceNotFound(OperatorError):
    """Raised when a caller names a trace id this deployment has no record of.

    `TraceStore.get` returns `None` for a missing trace, since "is it there" is a question with
    an answer; this is for the surfaces above it -- `runa traces show`, `runa ui`'s trace page,
    `runa eval --add` -- where a missing trace is the end of the request. It lives with the store
    rather than in each of them, so all three raise the same type for the same situation, and
    `require_trace` below is how they reach it, so they share the message too.
    """


class TraceStore:
    """One deployment's trace history: write one, read one, read the recent ones.

    The `traces`/`spans` tables of a `Database`, created on the first `save`. `list` is the only
    query: a caller after failed runs passes `status="error"`.
    """

    def __init__(self, database: Database) -> None:
        """Store the database this history lives in."""
        self.database = database

    async def save(self, trace: Trace) -> None:
        """Persist `trace` and every span in it, replacing any existing one with the same id."""
        async with self.database.transaction() as tx:
            await tx.execute(_upsert(TRACES), *_trace_values(trace))
            if trace.spans:
                await tx.executemany(_upsert(SPANS), [_span_values(span) for span in trace.spans])

    async def get(self, trace_id: str) -> Trace | None:
        """Look up one trace by id, with every span it has, or `None` if this store has none."""
        row = await self.database.fetchrow(f"SELECT * FROM {TRACES.name} WHERE id = $1", trace_id)
        if row is None:
            return None
        spans = await self.database.fetch(
            f"SELECT * FROM {SPANS.name} WHERE trace_id = $1 ORDER BY start_time", trace_id
        )
        return _to_trace(row, spans)

    async def list(
        self,
        *,
        limit: int = 50,
        agent: str | None = None,
        status: str | None = None,
        session_id: str | None = None,
    ) -> builtins.list[Trace]:
        """Return the most recent `limit` traces, newest first, optionally filtered.

        `agent` matches `Trace.name`, which `Agent.run` sets to the agent's declared `name`;
        `session_id` matches the `Session` a session-backed run was passed, when it was
        passed one; `status` is `"ok"` or `"error"`.

        Ties on `start_time` break by trace id descending, whichever database answers, so two
        traces that started in the same instant list in one order whatever the deployment.
        """
        clause, params = listing_clause(
            limit=limit, agent=agent, status=status, session_id=session_id
        )
        rows = await self.database.fetch(f"SELECT * FROM {TRACES.name} {clause}", *params)
        if not rows:
            return []
        # One query for every listed trace's spans rather than one per trace: against a shared
        # database each is a network round trip.
        ids = [row["id"] for row in rows]
        marks = ", ".join(f"${index}" for index in range(1, len(ids) + 1))
        spans = await self.database.fetch(
            f"SELECT * FROM {SPANS.name} WHERE trace_id IN ({marks}) ORDER BY start_time", *ids
        )
        by_trace: dict[str, builtins.list[Row]] = {trace_id: [] for trace_id in ids}
        for span in spans:
            by_trace[span["trace_id"]].append(span)
        return [_to_trace(row, by_trace[row["id"]]) for row in rows]


async def require_trace(store: TraceStore, trace_id: str) -> Trace:
    """Look `trace_id` up in `store`, raising `TraceNotFound` when it has no such trace.

    The raising form of `get`, which is the only form the three surfaces above a store want, so
    the message is worded once. `get` stays, since "is it there" is still a question worth asking.
    """
    trace = await store.get(trace_id)
    if trace is None:
        raise TraceNotFound(f"no trace found with id {trace_id!r}")
    return trace


def listing_clause(
    *,
    limit: int,
    agent: str | None = None,
    status: str | None = None,
    session_id: str | None = None,
) -> tuple[str, tuple[Any, ...]]:
    """Everything after `SELECT ... FROM traces` for `TraceStore.list`, and the values it binds.

    Keeps the traces matching whichever filters were given, newest first (ties broken by `id`),
    at most `limit` of them. `agent` matches `traces.name`, which for a run's trace is the
    agent's declared `name`. The placeholders are numbered, so nothing may be appended after it.
    """
    clauses: list[str] = []
    params: list[Any] = []
    for column, value in (("name", agent), ("status", status), ("session_id", session_id)):
        if value is not None:
            params.append(value)
            clauses.append(f"{column} = ${len(params)}")
    where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
    params.append(limit)
    order = f"ORDER BY start_time DESC, id DESC LIMIT ${len(params)}"
    return f"{where}{order}", tuple(params)


def _upsert(table: Table) -> str:
    """`INSERT` one row of `table`, replacing every column but `id` when that id is taken."""
    assignments = ", ".join(
        f"{column} = excluded.{column}" for column in table.column_names if column != "id"
    )
    return (
        f"INSERT INTO {table.name} ({', '.join(table.column_names)}) "
        f"VALUES ({table.placeholders()}) ON CONFLICT (id) DO UPDATE SET {assignments}"
    )


def _as_text(value: object) -> str | None:
    """Render a span's `input`/`output` as the one text column it is stored in.

    A span's input can arrive as a string already or as the structured value a tool was called
    with; a string is stored as is, anything else as its JSON.
    """
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _trace_values(trace: Trace) -> tuple[Any, ...]:
    """`trace`'s column values, in `TRACES.column_names` order."""
    return (
        trace.id,
        trace.name,
        trace.start_time,
        trace.end_time,
        trace.status,
        trace.session_id,
        json.dumps(trace.metadata, default=str),
    )


def _span_values(span: Span) -> tuple[Any, ...]:
    """`span`'s column values, in `SPANS.column_names` order."""
    return (
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
        None if span.usage is None else json.dumps(asdict(span.usage)),
    )


def _to_span(row: Row) -> Span:
    """One `spans` row as a `Span`."""
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
        usage=None if row["usage_json"] is None else _USAGE.validate_json(row["usage_json"]),
    )


def _to_trace(row: Row, span_rows: list[Row]) -> Trace:
    """One `traces` row plus its `spans` rows as a `Trace`."""
    return Trace(
        id=row["id"],
        name=row["name"],
        start_time=row["start_time"],
        end_time=row["end_time"],
        session_id=row["session_id"],
        spans=[_to_span(span_row) for span_row in span_rows],
        metadata=json.loads(row["metadata_json"]),
    )


__all__ = [
    "SPANS",
    "TRACES",
    "TraceNotFound",
    "TraceStore",
    "listing_clause",
    "require_trace",
]
