"""`runa.tracing`: automatic hierarchical observability for `Agent.run`/`run_sync`.

The public surface is deliberately small: `Trace` and `Span` are the only two concepts, matching
`result.trace`. Nothing here needs to be called for tracing to happen, `runa.runner.Runner`
builds and exports a `Trace` for every run itself, with no separate registration step. `trace`/
`span` and `observe` are the advanced, optional API described in the design.

Reading history back is `runa.db.traces()`, not a function here: it hands over a `TraceStore`
(`tracing/store.py`) already pointed at whichever backend this deployment has, so nothing that
queries traces names a file or a URL.
"""

from runa.tracing.config import (
    ConsoleExporter,
    SQLiteExporter,
    StoreExporter,
    TraceExporter,
    add_exporter,
    observe,
)
from runa.tracing.manual import span, trace
from runa.tracing.spans import Span, SpanStatus, SpanType
from runa.tracing.store import TraceStore
from runa.tracing.traces import Trace

__all__ = [
    "ConsoleExporter",
    "SQLiteExporter",
    "Span",
    "SpanStatus",
    "SpanType",
    "StoreExporter",
    "Trace",
    "TraceExporter",
    "TraceStore",
    "add_exporter",
    "observe",
    "span",
    "trace",
]
