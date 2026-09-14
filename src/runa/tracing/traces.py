"""tracing/traces.py: `Trace`, one logical agent execution and its `Span` tree."""

from dataclasses import dataclass, field
from typing import Any

from runa.tracing.spans import Span, SpanStatus

_TYPE_LABELS: dict[str, str] = {
    "agent": "Agent",
    "llm": "LLM",
    "tool": "Tool",
    "retrieval": "Retrieval",
    "handoff": "Handoff",
    "guardrail": "Guardrail",
    "custom": "Custom",
}


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "..."
    return f"{seconds:.2f}s"


def _fmt_tokens(output: Any) -> str | None:
    if not isinstance(output, dict):
        return None
    usage = output.get("usage")
    if not isinstance(usage, dict):
        return None
    total = usage.get("total_tokens")
    if total is None:
        input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
        if input_tokens is None or output_tokens is None:
            return None
        total = input_tokens + output_tokens
    return f"tokens: {total:,}"


@dataclass
class Trace:
    """One logical agent execution: a start/end time, metadata, and its `Span` tree.

    `spans` is a flat list: each `Span.parent_id` (or `None`, for a root span) is what gives it
    shape; `__str__` walks that structure to render the tree shown in the "Human-readable trace
    representation" section of the design.
    """

    id: str
    name: str
    start_time: float
    end_time: float | None = None
    spans: list[Span] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float | None:
        """Seconds between `start_time` and `end_time`, or `None` while the trace is still open."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def status(self) -> SpanStatus:
        """`"error"` if any span in this trace errored, `"ok"` otherwise."""
        return "error" if any(span.status == "error" for span in self.spans) else "ok"

    @property
    def errors(self) -> list[Span]:
        """Every span in this trace whose `status` is `"error"`."""
        return [span for span in self.spans if span.status == "error"]

    def _children(self) -> dict[str | None, list[Span]]:
        children: dict[str | None, list[Span]] = {}
        for span in sorted(self.spans, key=lambda s: s.start_time):
            children.setdefault(span.parent_id, []).append(span)
        return children

    def _render_span(self, span: Span, children: dict[str | None, list[Span]]) -> list[str]:
        glyph = "✓" if span.status == "ok" else "✗"
        label = _TYPE_LABELS.get(span.type, span.type)
        lines = [f"{label} {span.name} [{_fmt_duration(span.duration)}] {glyph}"]
        if span.status == "error" and span.error:
            lines.append(f"error: {span.error}")
        tokens = _fmt_tokens(span.output) if span.type == "llm" else None
        if tokens:
            lines.append(tokens)

        kids = children.get(span.id, [])
        rendered: list[str] = [lines[0]]
        for extra in lines[1:]:
            rendered.append(f"   {extra}")
        for index, kid in enumerate(kids):
            last = index == len(kids) - 1
            branch = "└─ " if last else "├─ "
            continuation = "   " if last else "│  "
            kid_lines = self._render_span(kid, children)
            rendered.append(f"{branch}{kid_lines[0]}")
            rendered.extend(f"{continuation}{line}" for line in kid_lines[1:])
        return rendered

    def __str__(self) -> str:
        """Render this trace as a box-drawing tree of its spans, roughly matching the design doc."""
        glyph = "✓" if self.status == "ok" else "✗"
        header = f"Trace {self.name} [{_fmt_duration(self.duration)}] {glyph}"
        children = self._children()
        roots = children.get(None, [])
        if not roots:
            return header

        lines = [header, "│"]
        for index, root in enumerate(roots):
            last = index == len(roots) - 1
            branch = "└─ " if last else "├─ "
            continuation = "   " if last else "│  "
            root_lines = self._render_span(root, children)
            lines.append(f"{branch}{root_lines[0]}")
            lines.extend(f"{continuation}{line}" for line in root_lines[1:])
        return "\n".join(lines)


__all__ = ["Trace"]
