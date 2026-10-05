"""cli/traces.py: `runa traces list/show/errors` over whichever store this deployment has.

Thin formatting over the `TraceStore` `runa.db.traces(root)` hands back, no query logic of its
own, matching how `cli/sessions.py` only formats what a `SessionStore` already exposes. Which
backend that is never appears here: `root` names a project directory, not a file.
"""

from pathlib import Path

from runa import db
from runa.tracing import Trace, TraceNotFound


def _summary(trace: Trace) -> str:
    duration = f"{trace.duration:.2f}s" if trace.duration is not None else "..."
    return f"{trace.id}  {trace.name:<24} {trace.status:<6} {duration}"


def list_traces_cli(*, root: Path, limit: int = 50) -> str:
    """List the most recent `limit` traces, newest first."""
    traces = db.traces(root).list(limit=limit)
    if not traces:
        return "no traces found"
    return "\n".join(_summary(trace) for trace in traces)


def show_trace(trace_id: str, *, root: Path) -> str:
    """Render one trace's full span tree."""
    trace = db.traces(root).get(trace_id)
    if trace is None:
        raise TraceNotFound(f"no trace found with id {trace_id!r}")
    return str(trace)


def list_errors_cli(*, root: Path, limit: int = 50) -> str:
    """List the most recent `limit` traces that errored, newest first."""
    traces = db.traces(root).list(limit=limit, status="error")
    if not traces:
        return "no errors found"
    return "\n".join(_summary(trace) for trace in traces)
