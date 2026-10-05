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
    "delegate": "Delegate",
    "guardrail": "Guardrail",
    "custom": "Custom",
}


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "..."
    return f"{seconds:.2f}s"


def _display_name(span: Span) -> str:
    """`span.name`, minus the `"transfer_to_"` a `Handoff.tool_name` always carries for the model.

    The prefix is real and stays on the wire -- the model calls the tool by that name -- it is
    just noise in a person-readable row, where the target agent's name says enough.
    """
    return span.name.removeprefix("transfer_to_") if span.type == "handoff" else span.name


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


@dataclass(frozen=True)
class SpanRow:
    """One span as a person reads it: the tree's shape and its labels, no characters chosen yet.

    What `Trace.walk` yields. It carries everything a renderer needs to draw a row and nothing
    about how to draw it, so `runa traces show` can turn these into box-drawing lines and `runa
    ui` into nested `<div>`s without either re-deriving which spans are whose children, what a
    span type is called, or what a handoff is named once the model's prefix comes off.

    `span` is the raw `Span` for anything a renderer wants that isn't person-readable yet: its
    `input`/`output`/`attributes`/`status`/`error`.
    """

    span: Span
    label: str
    name: str
    duration: str
    tokens: str | None
    children: tuple[SpanRow, ...]

    @property
    def hands_off(self) -> bool:
        """Whether every later sibling is the work of the agent this row handed off to.

        `run_loop.py` parents a handoff's aftermath to the same turn-level span as the handoff
        itself, so a row's position in the sibling sequence is the only place that boundary
        exists. A renderer that groups siblings visually needs it; one that only lists them can
        ignore it.
        """
        return self.span.type == "handoff"


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
    session_id: str | None = None
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

    @property
    def elapsed(self) -> str:
        """`duration` as a person reads it: `"1.23s"`, or `"..."` while the trace is still open."""
        return _fmt_duration(self.duration)

    def walk(self) -> tuple[SpanRow, ...]:
        """This trace's root spans as `SpanRow`s, each carrying its own ordered children.

        The only traversal of the span tree. `spans` is flat and shaped solely by `parent_id`, so
        recovering the tree -- and ordering siblings by start time -- is a fact about traces, and
        it is learned here rather than again in every surface that shows one.
        """
        children: dict[str | None, list[Span]] = {}
        for span in sorted(self.spans, key=lambda s: s.start_time):
            children.setdefault(span.parent_id, []).append(span)

        def rows(parent_id: str | None) -> tuple[SpanRow, ...]:
            return tuple(
                SpanRow(
                    span=span,
                    label=_TYPE_LABELS.get(span.type, span.type),
                    name=_display_name(span),
                    duration=_fmt_duration(span.duration),
                    tokens=_fmt_tokens(span.output) if span.type == "llm" else None,
                    children=rows(span.id),
                )
                for span in children.get(parent_id, [])
            )

        return rows(None)

    def _render_row(self, row: SpanRow) -> list[str]:
        glyph = "✓" if row.span.status == "ok" else "✗"
        lines = [f"{row.label} {row.name} [{row.duration}] {glyph}"]
        if row.span.status == "error" and row.span.error:
            lines.append(f"error: {row.span.error}")
        if row.tokens:
            lines.append(row.tokens)
        if row.hands_off:
            lines.append(f"→ {row.name} takes over")

        rendered = [lines[0], *(f"   {extra}" for extra in lines[1:])]
        rendered.extend(self._render_siblings(row.children))
        return rendered

    def _render_siblings(self, rows: tuple[SpanRow, ...]) -> list[str]:
        lines: list[str] = []
        for index, row in enumerate(rows):
            last = index == len(rows) - 1
            branch, continuation = ("└─ ", "   ") if last else ("├─ ", "│  ")
            head, *rest = self._render_row(row)
            lines.append(f"{branch}{head}")
            lines.extend(f"{continuation}{line}" for line in rest)
        return lines

    def __str__(self) -> str:
        """Render this trace as a box-drawing tree of its spans, roughly matching the design doc."""
        glyph = "✓" if self.status == "ok" else "✗"
        header = f"Trace {self.name} [{self.elapsed}] {glyph}"
        roots = self.walk()
        return "\n".join([header, "│", *self._render_siblings(roots)]) if roots else header


__all__ = ["SpanRow", "Trace"]
