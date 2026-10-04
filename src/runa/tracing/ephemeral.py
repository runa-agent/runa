"""tracing/ephemeral.py: `EphemeralTraceStore`, trace history that dies with the process.

What `runa.db.traces()` resolves to under `RUNA_DATABASE_URL=memory://`. No file, no server, no
`skipif` on a live TCP probe: a test that wants to assert what `runa ui` shows sets one
environment variable and gets a real `TraceStore`.

Rows are kept marshalled, as `tracing/store.py`'s `trace_values`/`span_values` produce them and
`to_trace` reads them back, rather than as the `Trace` objects that were handed in. That is what
makes this a store rather than a dict: a span's structured input comes back as the same text the
SQL adapters would have stored, and a caller that mutates a `Trace` it got from `get` cannot
reach into the history behind it.
"""

from typing import Any

from runa.tracing.store import (
    SPAN_COLUMNS,
    TRACE_COLUMNS,
    span_values,
    to_trace,
    trace_values,
)
from runa.tracing.traces import Trace

_traces: dict[str, dict[str, Any]] = {}
_spans: dict[str, list[dict[str, Any]]] = {}


def reset() -> None:
    """Drop every stored trace. `runa.db.reset_ephemeral()` is how a test reaches this."""
    _traces.clear()
    _spans.clear()


class EphemeralTraceStore:
    """The in-process `TraceStore`: this process's trace history, gone when it exits.

    A handle, not a container: every instance reads and writes the one module-level history, so
    two `db.traces()` calls see each other's writes the way two connections to one file do.
    """

    def save(self, trace: Trace) -> None:
        """Persist `trace` and every span in it, replacing any existing one with the same id."""
        _traces[trace.id] = dict(zip(TRACE_COLUMNS, trace_values(trace), strict=True))
        _spans[trace.id] = [
            dict(zip(SPAN_COLUMNS, span_values(span), strict=True)) for span in trace.spans
        ]

    def get(self, trace_id: str) -> Trace | None:
        """Look up one trace by id, with every span it has, or `None` if this process has none."""
        row = _traces.get(trace_id)
        if row is None:
            return None
        return to_trace(row, self._span_rows(trace_id))

    def list(
        self,
        *,
        limit: int = 50,
        agent: str | None = None,
        status: str | None = None,
        session_id: str | None = None,
    ) -> list[Trace]:
        """Return the most recent `limit` traces, newest first, optionally filtered."""
        rows = [
            row
            for row in _traces.values()
            if (agent is None or row["name"] == agent)
            and (status is None or row["status"] == status)
            and (session_id is None or row["session_id"] == session_id)
        ]
        rows.sort(key=lambda row: row["start_time"], reverse=True)
        return [to_trace(row, self._span_rows(row["id"])) for row in rows[:limit]]

    def _span_rows(self, trace_id: str) -> list[dict[str, Any]]:
        return sorted(_spans.get(trace_id, []), key=lambda span: span["start_time"])


__all__ = ["EphemeralTraceStore", "reset"]
