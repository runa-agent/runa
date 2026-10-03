"""cli/traces.py: `runa traces list/show/errors` over `db/runa.db`.

Thin formatting over `runa.tracing.storage`'s query API, no separate query logic lives here,
matching how `cli/runs.py` only formats what `runa.eval`/session storage already expose.
"""

from pathlib import Path

from runa.cli._project import resolve_db_path
from runa.tracing import Trace, get_errors, get_trace, list_traces


class TraceNotFound(Exception):
    """Raised when `runa traces show` names a trace id `db/runa.db` has no record of."""


def _summary(trace: Trace) -> str:
    duration = f"{trace.duration:.2f}s" if trace.duration is not None else "..."
    return f"{trace.id}  {trace.name:<24} {trace.status:<6} {duration}"


def list_traces_cli(*, root: Path, limit: int = 50) -> str:
    """List the most recent `limit` traces, newest first."""
    traces = list_traces(limit=limit, db_path=resolve_db_path(root))
    if not traces:
        return "no traces found"
    return "\n".join(_summary(trace) for trace in traces)


def show_trace(trace_id: str, *, root: Path) -> str:
    """Render one trace's full span tree."""
    trace = get_trace(trace_id, db_path=resolve_db_path(root))
    if trace is None:
        raise TraceNotFound(f"no trace found with id {trace_id!r}")
    return str(trace)


def list_errors_cli(*, root: Path, limit: int = 50) -> str:
    """List the most recent `limit` traces that errored, newest first."""
    traces = get_errors(limit=limit, db_path=resolve_db_path(root))
    if not traces:
        return "no errors found"
    return "\n".join(_summary(trace) for trace in traces)
