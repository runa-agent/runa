"""tracing/spans.py: `Span`, one meaningful operation inside a `Trace`.

Deliberately restricted to the controlled vocabulary the design calls for (`SpanType`) instead of
inventing separate classes per operation kind: an agent turn, an LLM call, a tool call, a
retrieval, a handoff, a delegate call, or a guardrail check are all just a `Span` with a different
`type` and whatever `attributes`/`input`/`output` that kind of operation actually produced.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

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


__all__ = ["Span", "SpanStatus", "SpanType"]
