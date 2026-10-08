"""cli/traces.py: `runa traces list/show/errors` over whichever store this deployment has.

Thin formatting over the `TraceStore` `runa.db.traces()` hands back, no query logic of its own,
matching how `cli/sessions.py` only formats what a `SessionStore` already exposes. Neither which
backend that is nor which project it belongs to appears here: `runa.db` was told both before this
command was dispatched.
"""

from runa import db
from runa.tracing import Trace, require_trace


def _summary(trace: Trace) -> str:
    duration = f"{trace.duration:.2f}s" if trace.duration is not None else "..."
    return f"{trace.id}  {trace.name:<24} {trace.status:<6} {duration}"


def list_traces_cli(*, limit: int = 50) -> str:
    """List the most recent `limit` traces, newest first."""
    traces = db.traces().list(limit=limit)
    if not traces:
        return "no traces found"
    return "\n".join(_summary(trace) for trace in traces)


def show_trace(trace_id: str) -> str:
    """Render one trace's full span tree."""
    return str(require_trace(db.traces(), trace_id))


def list_errors_cli(*, limit: int = 50) -> str:
    """List the most recent `limit` traces that errored, newest first."""
    traces = db.traces().list(limit=limit, status="error")
    if not traces:
        return "no errors found"
    return "\n".join(_summary(trace) for trace in traces)
