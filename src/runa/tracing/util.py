"""tracing/util.py: fresh, opaque ids for a `Trace`/`Span`."""

import uuid


def gen_trace_id() -> str:
    """Generate a fresh, opaque trace id."""
    return uuid.uuid4().hex


def gen_span_id() -> str:
    """Generate a fresh, opaque span id."""
    return uuid.uuid4().hex


__all__ = ["gen_span_id", "gen_trace_id"]
