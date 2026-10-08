"""tracing/store.py: `TraceStore`, where finished traces go and where `runa ui` reads them from.

One interface, three adapters: `tracing/sqlite.py`, `tracing/postgres.py`,
`tracing/ephemeral.py`. Which one a caller gets is `runa.db.traces(...)`'s decision, asked once,
which is what lets `runa traces`, `runa ui` and the exporter stay backend-agnostic: they hold a
`TraceStore` and get whichever history the deployment actually has.

Asked once is the point. The exporter used to decide twice on a single write -- `tracing/config.py`
picked an exporter by reading the environment, and the `save_trace` it called then read the
environment again -- so the two could disagree.

The two tables live here too, and so does the marshalling: `TRACES`/`SPANS` state what a row is
(`db/schema.py` renders them for whichever dialect an adapter speaks) and
`to_trace`/`trace_values`/`span_values` are the one mapping between such a row and a `Trace`, so
a column read out of order or a `json.dumps` left off is a mistake there is only one place to
make.
"""

import json
from typing import Any, Protocol

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
    ),
    indexes=(Index("trace_id", "trace_id"),),
)


class TraceNotFound(OperatorError):
    """Raised when a caller names a trace id this deployment has no record of.

    `TraceStore.get` returns `None` for a missing trace, since "is it there" is a question with
    an answer; this is for the surfaces above it -- `runa traces show`, `runa ui`'s trace page,
    `runa eval --add` -- where a missing trace is the end of the request. It lives with the store
    rather than in each of them, so all three raise the same type for the same situation.
    """


class TraceStore(Protocol):
    """One deployment's trace history: write one, read one, read the recent ones.

    No inheritance required, every adapter satisfies this by matching shape. `list` is the only
    query: `get_errors`-style convenience readers were a second shape over the same filter, so
    callers pass `status="error"` instead.
    """

    def save(self, trace: Trace) -> None:
        """Persist `trace` and every span in it, replacing any existing one with the same id."""
        ...

    def get(self, trace_id: str) -> Trace | None:
        """Look up one trace by id, with every span it has, or `None` if this store has none."""
        ...

    def list(
        self,
        *,
        limit: int = 50,
        agent: str | None = None,
        status: str | None = None,
        session_id: str | None = None,
    ) -> list[Trace]:
        """Return the most recent `limit` traces, newest first, optionally filtered.

        `agent` matches `Trace.name` (the workflow name `Agent.run` sets to the agent's class
        name); `session_id` matches the `Session` a session-backed run was passed, when it was
        passed one; `status` is `"ok"` or `"error"`.
        """
        ...


def as_text(value: object) -> str | None:
    """Render a span's `input`/`output` as the single text column both backends store.

    A span's input can arrive as a string already or as the structured value a tool was called
    with. Both stores keep one `TEXT` column, so this is where that choice is made -- once,
    rather than in each adapter, where it had already been copied.
    """
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value, default=str)


def trace_values(trace: Trace) -> tuple[Any, ...]:
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


def span_values(span: Span) -> tuple[Any, ...]:
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
        as_text(span.input),
        as_text(span.output),
        span.error,
    )


def to_span(row: Any) -> Span:
    """One `spans` row as a `Span`. `row` is anything subscriptable by column name."""
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


def to_trace(row: Any, span_rows: list[Any]) -> Trace:
    """One `traces` row plus its `spans` rows as a `Trace`.

    `sqlite3.Row` and `asyncpg.Record` are both subscriptable by column name, which is the only
    thing this needs of either, so the mapping is written once for both.
    """
    return Trace(
        id=row["id"],
        name=row["name"],
        start_time=row["start_time"],
        end_time=row["end_time"],
        session_id=row["session_id"],
        spans=[to_span(span_row) for span_row in span_rows],
        metadata=json.loads(row["metadata_json"]),
    )


__all__ = [
    "SPANS",
    "TRACES",
    "TraceNotFound",
    "TraceStore",
    "as_text",
    "span_values",
    "to_span",
    "to_trace",
    "trace_values",
]
