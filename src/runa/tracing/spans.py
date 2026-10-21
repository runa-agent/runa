"""tracing/spans.py: `Span`, one meaningful operation inside a `Trace`.

Deliberately restricted to the controlled vocabulary the design calls for (`SpanType`) instead of
inventing separate classes per operation kind: an agent turn, an LLM call, a tool call, a
retrieval, a handoff, a delegate call, or a guardrail check are all just a `Span` with a different
`type` and whatever `attributes`/`input`/`output` that kind of operation actually produced.

`open_span`/`close_span` are the one way a span is made and finished, whoever makes it: they put
its input and output through the privacy policy (`tracing.config.kept`), so a span's content is
never kept without it.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from runa.tracing.config import kept
from runa.tracing.util import gen_span_id

if TYPE_CHECKING:
    from runa.tracing.traces import Trace

SpanType = Literal[
    "agent", "llm", "tool", "retrieval", "handoff", "delegate", "guardrail", "custom"
]
SpanStatus = Literal["ok", "error"]


@dataclass
class Span:
    """One traceable operation: an agent turn, an LLM call, a tool call, a handoff, ...

    `input`/`output`/`error` stay `None` when that data isn't available or was withheld by the
    active privacy policy (see `tracing.config.observe`); missing data is never fabricated.
    """

    id: str
    trace_id: str
    parent_id: str | None
    name: str
    type: SpanType
    start_time: float
    end_time: float | None = None
    status: SpanStatus = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)
    input: Any = None
    output: Any = None
    error: str | None = None

    @property
    def duration(self) -> float | None:
        """Seconds between `start_time` and `end_time`, or `None` while the span is still open."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


def open_span(
    trace: Trace,
    name: str,
    span_type: SpanType,
    *,
    parent_id: str | None = None,
    input: Any = None,
) -> Span:
    """Start a span on `trace`, recorded on it straight away, keeping `input` as policy allows."""
    span = Span(
        id=gen_span_id(),
        trace_id=trace.id,
        parent_id=parent_id,
        name=name,
        type=span_type,
        start_time=time.time(),
        input=kept(input, "input"),
    )
    trace.spans.append(span)
    return span


def close_span(span: Span, *, output: Any = None, error: str | None = None) -> None:
    """Finish `span`, keeping `output` as policy allows and marking it failed if `error` is given.

    A `"tool"` span's output is a tool result, held to `max_tool_result_bytes`.
    """
    span.end_time = time.time()
    span.output = kept(output, "tool_result" if span.type == "tool" else "output")
    if error is not None:
        span.status = "error"
        span.error = error


__all__ = ["Span", "SpanStatus", "SpanType", "close_span", "open_span"]
