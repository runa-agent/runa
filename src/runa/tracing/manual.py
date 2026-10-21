"""tracing/manual.py: `trace`/`span`, the manual/advanced tracing API.

Standalone from `run_internal`'s automatic per-`Agent.run()` tracing: these build and export their
own `Trace`, for instrumenting code that isn't itself an agent run. An `Agent.run()` inside a
`trace` block still produces its own `Trace`, stamped with the enclosing one's id as `group_id`,
so a code-driven workflow's runs are grouped without passing anything to `run`.
"""

import time
from contextvars import ContextVar
from typing import Any

from runa._loop import current_loop
from runa.tracing.export import export_trace, export_trace_later
from runa.tracing.spans import Span, close_span, open_span
from runa.tracing.traces import Trace
from runa.tracing.util import gen_trace_id

_current: ContextVar[Trace | None] = ContextVar("runa_current_trace", default=None)


def current_trace() -> Trace | None:
    """The innermost open `trace` block's `Trace`, or `None` outside one."""
    return _current.get()


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
            id=gen_trace_id(), name=name, start_time=time.time(), metadata=merged_metadata
        )

    def __enter__(self) -> Trace:
        """Return the `Trace` to pass as `span()`'s `trace` argument."""
        self._token = _current.set(self._trace)
        return self._trace

    def __exit__(self, *exc_info: object) -> None:
        """Close and export the `Trace`, regardless of whether the block raised.

        Exported inline when no loop runs on this thread, since the only work that holds up is the
        code that opened the block. Inside a coroutine it is queued, as a finishing run's is
        (`export_trace_later`): exporting inline there would stall the loop, or on Runa's own
        background loop wait on itself.
        """
        _current.reset(self._token)
        self._trace.end_time = time.time()
        if current_loop() is None:
            export_trace(self._trace)
        else:
            export_trace_later(self._trace)


class span:
    """Add a manual `"custom"` span to `trace`, optionally nested under `parent`.

    See `trace`'s docstring for usage; pass an already-open `Span` as `parent` to nest one span
    inside another. `data` is the span's input and whatever the block sets as `span.output` its
    output, each kept only as the `observe` policy in force when it's recorded allows.
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
        self._span = open_span(
            self._trace, self._name, "custom", parent_id=self._parent_id, input=self._data
        )
        return self._span

    def __exit__(self, *exc_info: object) -> None:
        """Close the span, marking it an error if the block raised."""
        assert self._span is not None
        error = str(exc_info[1]) if exc_info[0] is not None else None
        close_span(self._span, output=self._span.output, error=error)


__all__ = ["span", "trace"]
