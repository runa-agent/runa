"""spans.py: tracing span helpers shared by the turn loop, guardrails, and tool execution."""

import time
from dataclasses import dataclass
from typing import Any

from runa.tracing import config as tracing_config
from runa.tracing.config import exporters
from runa.tracing.spans import Span
from runa.tracing.traces import Trace
from runa.tracing.util import gen_span_id


@dataclass(frozen=True)
class _Spans:
    """Where a span opened right here belongs: which trace, and under which parent.

    One value instead of the `(trace, parent_id)` pair every step used to take alongside its own
    arguments. A step that opens spans inside its own (a tool call, around its guardrails) hands
    on `under(span)` rather than remembering to swap one of two parameters.
    """

    trace: Trace
    parent_id: str | None = None

    def open(self, name: str, span_type: Any, *, input: Any = None) -> Span:
        """Start a span under this scope's parent, recorded on the trace straight away."""
        span = Span(
            id=gen_span_id(),
            trace_id=self.trace.id,
            parent_id=self.parent_id,
            name=name,
            type=span_type,
            start_time=time.time(),
        )
        if input is not None and tracing_config.capture_inputs():
            span.input = tracing_config.apply_policy(input, max_bytes=tracing_config.input_limit())
        self.trace.spans.append(span)
        return span

    def under(self, span: Span) -> _Spans:
        """This same trace, with `span` as the parent: the scope inside a span just opened."""
        return _Spans(self.trace, span.id)


def _close_span(span: Span, *, error: str | None = None, output: Any = None) -> None:
    span.end_time = time.time()
    if output is not None and tracing_config.capture_outputs():
        max_bytes = (
            tracing_config.tool_result_limit()
            if span.type == "tool"
            else tracing_config.output_limit()
        )
        span.output = tracing_config.apply_policy(output, max_bytes=max_bytes)
    if error is not None:
        span.status = "error"
        span.error = error


def _export(trace: Trace) -> None:
    from runa.lifecycle import logger

    for exporter in exporters():
        try:
            exporter.export(trace)
        except Exception:  # noqa: BLE001 -- tracing must never break an agent run
            logger.warning("tracing: exporter %r failed", exporter, exc_info=True)


__all__ = ["_Spans", "_close_span", "_export"]
