"""tracing/database.py: `DatabaseTraceStore`, the `TraceStore` over a `Database`.

The `traces`/`spans` tables in whichever database the deployment has. Local, that is a `db/runa.db`
beside the process: exactly right for one machine and wrong for a deployment, where three replicas
keep three disjoint trace histories and `runa ui` can only ever show whichever one it happens to be
looking at. `runa.db.traces()` returns this over the shared Postgres whenever `RUNA_DATABASE_URL`
is a `postgresql://` one.

The two tables, the row marshalling and the listing's filter-order-cap clause are
`tracing/store.py`'s; this module's own code is the query shapes.
"""

import builtins
from typing import Any

from runa.db.handle import Database, sync
from runa.tracing.store import (
    SPANS,
    TRACES,
    listing_clause,
    span_values,
    to_trace,
    trace_values,
)
from runa.tracing.traces import Trace


def _assignments(columns: tuple[str, ...], *, keep: str = "id") -> str:
    return ", ".join(f"{column} = excluded.{column}" for column in columns if column != keep)


class DatabaseTraceStore:
    """The `TraceStore` over a `Database`: its `traces`/`spans` tables."""

    def __init__(self, database: Database) -> None:
        """Store the database this history lives in; its tables are created on the first `save`."""
        self.database = database

    @sync
    async def save(self, trace: Trace) -> None:
        """Persist `trace` and every span in it, replacing any existing one with the same id."""
        async with self.database.transaction() as tx:
            await tx.execute(
                f"INSERT INTO {TRACES.name} ({', '.join(TRACES.column_names)}) "
                f"VALUES ({TRACES.placeholders()}) "
                f"ON CONFLICT (id) DO UPDATE SET {_assignments(TRACES.column_names)}",
                *trace_values(trace),
            )
            if trace.spans:
                await tx.executemany(
                    f"INSERT INTO {SPANS.name} ({', '.join(SPANS.column_names)}) "
                    f"VALUES ({SPANS.placeholders()}) "
                    f"ON CONFLICT (id) DO UPDATE SET {_assignments(SPANS.column_names)}",
                    [span_values(span) for span in trace.spans],
                )

    @sync
    async def get(self, trace_id: str) -> Trace | None:
        """Look up one trace by id, with every span it has, or `None` if this database has none."""
        row = await self.database.fetchrow(f"SELECT * FROM {TRACES.name} WHERE id = $1", trace_id)
        if row is None:
            return None
        spans = await self.database.fetch(
            f"SELECT * FROM {SPANS.name} WHERE trace_id = $1 ORDER BY start_time", trace_id
        )
        return to_trace(row, spans)

    @sync
    async def list(
        self,
        *,
        limit: int = 50,
        agent: str | None = None,
        status: str | None = None,
        session_id: str | None = None,
    ) -> builtins.list[Trace]:
        """Return the most recent `limit` traces, newest first, optionally filtered."""
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
        by_trace: dict[str, builtins.list[Any]] = {trace_id: [] for trace_id in ids}
        for span in spans:
            by_trace[span["trace_id"]].append(span)
        return [to_trace(row, by_trace[row["id"]]) for row in rows]


__all__ = ["DatabaseTraceStore"]
