"""tracing/manual.py: `trace`/`span`, the manual/advanced tracing API.

Standalone from `run_internal`'s automatic per-`Agent.run()` tracing: these build and export their
own `Trace`, for instrumenting code that isn't itself an agent run. An `Agent.run()` inside a
`trace` block still produces its own `Trace`, stamped with the enclosing one's id as `group_id`,
so a code-driven workflow's runs are grouped without passing anything to `run`.
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from typing import Any

from runa.tracing.config import exporters
from runa.tracing.spans import Span
from runa.tracing.traces import Trace

_current: ContextVar[Trace | None] = ContextVar("runa_current_trace", default=None)


def current_trace() -> Trace | None:
    """The innermost open `trace` block's `Trace`, or `None` outside one."""
    return _current.get()


def _export(finished: Trace) -> None:
    from runa.lifecycle import logger

    for exporter in exporters():
        try:
            exporter.export(finished)
        except Exception:  # noqa: BLE001 -- tracing must never break the caller
            logger.warning("tracing: exporter %r failed", exporter, exc_info=True)


class trace:
    """Create and export a standalone `Trace` to group manual `span()`s under.

    Usage::

        with tracing.trace("nightly-batch") as t:
            with tracing.span("step-1", t):
                ...
    """

    def __init__(
        self, name: str, *, group_id: str | None = None, metadata: dict[str, Any] | None = None
    ) -> None:
        """Build the `Trace` this block will export on exit."""
        merged_metadata = dict(metadata or {})
        if group_id is not None:
            merged_metadata["group_id"] = group_id
        self._trace = Trace(
            id=uuid.uuid4().hex, name=name, start_time=time.time(), metadata=merged_metadata
        )

    def __enter__(self) -> Trace:
        """Return the `Trace` to pass as `span()`'s `trace` argument."""
        self._token = _current.set(self._trace)
        return self._trace

    def __exit__(self, *exc_info: object) -> None:
        """Close and export the `Trace`, regardless of whether the block raised."""
        _current.reset(self._token)
        self._trace.end_time = time.time()
        _export(self._trace)


class span:
    """Add a manual `"custom"` span to `trace`, optionally nested under `parent`.

    See `trace`'s docstring for usage; pass an already-open `Span` as `parent` to nest one span
    inside another.
    """

    def __init__(
        self, name: str, trace: Trace, *, parent: Span | None = None, data: Any = None
    ) -> None:
        """Store where this span belongs; it's created and appended to `trace` on `__enter__`."""
        self._trace = trace
        self._name = name
        self._parent_id = parent.id if parent is not None else None
        self._data = data
        self._span: Span | None = None

    def __enter__(self) -> Span:
        """Open the span, appending it to the enclosing `trace`."""
        self._span = Span(
            id=uuid.uuid4().hex,
            trace_id=self._trace.id,
            parent_id=self._parent_id,
            name=self._name,
            type="custom",
            start_time=time.time(),
            input=self._data,
        )
        self._trace.spans.append(self._span)
        return self._span

    def __exit__(self, *exc_info: object) -> None:
        """Close the span, marking it an error if the block raised."""
        assert self._span is not None
        self._span.end_time = time.time()
        if exc_info[0] is not None:
            self._span.status = "error"
            self._span.error = str(exc_info[1])


__all__ = ["span", "trace"]
