"""web/traces.py: the Traces pages -- every run's span tree, from `db/runa.db`.

Data comes straight from `runa.tracing` (`list_traces`/`get_trace`/`get_errors`, the exact same
query API `cli/traces.py` formats as text); this module only turns a `Trace`'s `Span` tree into
an HTML waterfall.
"""

import json
from pathlib import Path
from typing import Any

from runa.cli._project import resolve_db_path
from runa.tracing import Trace, get_errors, get_trace, list_traces
from runa.tracing.spans import Span
from runa.tracing.traces import _TYPE_LABELS, _fmt_duration, _fmt_tokens
from runa.web._html import back_link, chip, empty, escape, page, pre

__all__ = ["TraceNotFound", "render_detail", "render_list"]


class TraceNotFound(Exception):
    """Raised when `render_detail` names a trace id `db/runa.db` has no record of."""


def _children_map(spans: list[Span]) -> dict[str | None, list[Span]]:
    children: dict[str | None, list[Span]] = {}
    for span in sorted(spans, key=lambda s: s.start_time):
        children.setdefault(span.parent_id, []).append(span)
    return children


def _format_value(value: Any) -> str:
    """Pretty-print a span's `input`/`output` for display.

    `tracing/storage.py` round-trips a dict/list `input`/`output` through SQLite as compact JSON
    text (only `attributes` comes back as a real dict, see `_row_to_trace`), so a string value
    here may itself be JSON worth re-indenting -- attempted first, falling back to the raw text
    for a value that was always a plain string.
    """
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        if isinstance(parsed, dict | list):
            return json.dumps(parsed, indent=2, default=str)
        return value
    try:
        return json.dumps(value, indent=2, default=str)
    except TypeError:
        return str(value)


def _span_details(span: Span) -> str:
    fields = []
    if span.input is not None:
        fields.append(f'<div class="field-label">Input</div>{pre(_format_value(span.input))}')
    if span.output is not None:
        fields.append(f'<div class="field-label">Output</div>{pre(_format_value(span.output))}')
    if span.attributes:
        fields.append(
            f'<div class="field-label">Attributes</div>{pre(_format_value(span.attributes))}'
        )
    if not fields:
        return ""
    body = "".join(fields)
    return f'<details><summary>details</summary><div class="span-body">{body}</div></details>'


def _render_span(span: Span, children: dict[str | None, list[Span]]) -> str:
    dot = "ok" if span.status == "ok" else "error"
    label = _TYPE_LABELS.get(span.type, span.type)
    tokens = _fmt_tokens(span.output) if span.type == "llm" else None
    tokens_html = f'<span class="span-tokens">{escape(tokens)}</span>' if tokens else ""
    row = (
        f'<div class="span-row"><span class="dot {dot}"></span>'
        f'<span class="span-type">{escape(label)}</span>'
        f'<span class="span-name">{escape(span.name)}</span>'
        f'<span class="span-duration">{escape(_fmt_duration(span.duration))}</span>'
        f"{tokens_html}</div>"
    )
    error = f'<div class="error-text">{escape(span.error)}</div>' if span.error else ""
    kids = children.get(span.id, [])
    kids_html = (
        f"<ul>{''.join(f'<li>{_render_span(kid, children)}</li>' for kid in kids)}</ul>"
        if kids
        else ""
    )
    return f'<div class="span-node">{row}{error}{_span_details(span)}{kids_html}</div>'


def _tree(trace: Trace) -> str:
    children = _children_map(trace.spans)
    roots = children.get(None, [])
    if not roots:
        return empty("no spans recorded")
    items = "".join(f"<li>{_render_span(root, children)}</li>" for root in roots)
    return f'<ul class="span-tree">{items}</ul>'


def render_list(*, root: Path, status: str | None = None) -> str:
    """Render `/traces`: the most recent runs, newest first, optionally errors-only."""
    db_path = resolve_db_path(root)
    traces = (
        get_errors(limit=100, db_path=db_path)
        if status == "error"
        else list_traces(limit=100, db_path=db_path)
    )
    filters = (
        f'<a href="/traces" class="chip{"" if status != "error" else " accent"}">all</a> '
        f'<a href="/traces?status=error" class="chip{" accent" if status == "error" else ""}">'
        "errors only</a>"
    )
    if not traces:
        body = empty("no traces yet -- run an Agent to produce one")
    else:
        rows = "".join(
            f'<a class="row" href="/traces/{escape(trace.id)}">'
            f'<span class="dot {"ok" if trace.status == "ok" else "error"}"></span>'
            f'<span class="primary">{escape(trace.name)}</span>'
            f'<span class="meta">{escape(_fmt_duration(trace.duration))}</span>'
            f'<span class="meta">{escape(trace.id)}</span></a>'
            for trace in traces
        )
        body = f'<div class="list">{rows}</div>'
    return page(
        title="Traces",
        active="Traces",
        body=f'<h1>Traces</h1><p class="subtitle">{filters}</p>{body}',
    )


def render_detail(trace_id: str, *, root: Path) -> str:
    """Render `/traces/{trace_id}`: that trace's header plus its full span waterfall."""
    trace = get_trace(trace_id, db_path=resolve_db_path(root))
    if trace is None:
        raise TraceNotFound(f"no trace found with id {trace_id!r}")
    status = "ok" if trace.status == "ok" else "error"
    header = (
        f"<h1>{escape(trace.name)} {chip(trace.status, status)}</h1>"
        f'<p class="subtitle">{escape(trace.id)} · {escape(_fmt_duration(trace.duration))}</p>'
    )
    body = back_link("/traces", "traces") + header + _tree(trace)
    return page(title=trace.name, active="Traces", body=body)
