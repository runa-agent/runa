"""web/traces.py: the trace detail page -- one run's span tree, from `db/runa.db`.

Data comes straight from `runa.tracing` (`get_trace`, the same query API `cli/traces.py show`
uses); this module only turns a `Trace`'s `Span` tree into an HTML waterfall. No standalone list
page: `web/sessions.py`'s merged timeline is the primary way to reach a trace; this is the
"open trace"/direct-by-id destination (see `web/app.py`'s module docstring).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from runa.cli._project import resolve_db_path
from runa.cli.eval import has_case, traced_input
from runa.tracing import Trace, get_trace
from runa.tracing.spans import Span
from runa.tracing.traces import _TYPE_LABELS, _fmt_duration, _fmt_tokens
from runa.web._html import chip, empty, escape, page, pre

__all__ = ["TraceNotFound", "render_detail"]


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


def _span_body(span: Span) -> str:
    """Input/output/attributes for one span, or `""` if it has none worth showing."""
    fields = []
    if span.input is not None:
        fields.append(f'<div class="field-label">Input</div>{pre(_format_value(span.input))}')
    if span.output is not None:
        fields.append(f'<div class="field-label">Output</div>{pre(_format_value(span.output))}')
    if span.attributes:
        fields.append(
            f'<div class="field-label">Attributes</div>{pre(_format_value(span.attributes))}'
        )
    return "".join(fields)


def _span_display_name(span: Span) -> str:
    """`span.name`, minus the `"transfer_to_"` a `Handoff.tool_name` always carries for the model.

    That prefix is real and stays on the wire (the model calls it by that name), it's just noise
    once you're looking at a person-readable span row -- the target agent's name says enough.
    """
    if span.type == "handoff" and span.name.startswith("transfer_to_"):
        return span.name.removeprefix("transfer_to_")
    return span.name


def _render_span(span: Span, children: dict[str | None, list[Span]]) -> str:
    dot = "ok" if span.status == "ok" else "error"
    label = _TYPE_LABELS.get(span.type, span.type)
    tokens = _fmt_tokens(span.output) if span.type == "llm" else None
    tokens_html = f'<span class="span-tokens">{escape(tokens)}</span>' if tokens else ""
    row_content = (
        f'<span class="dot {dot}"></span>'
        f'<span class="span-type">{escape(label)}</span>'
        f'<span class="span-name">{escape(_span_display_name(span))}</span>'
        f'<span class="span-duration">{escape(_fmt_duration(span.duration))}</span>'
        f"{tokens_html}"
    )
    body = _span_body(span)
    row = (
        f'<details><summary class="span-row">{row_content}</summary>'
        f'<div class="span-body">{body}</div></details>'
        if body
        else f'<div class="span-row">{row_content}</div>'
    )
    error = f'<div class="error-text">{escape(span.error)}</div>' if span.error else ""
    kids = children.get(span.id, [])
    kids_html = f"<ul>{_siblings_html(kids, children)}</ul>" if kids else ""
    return f'<div class="span-node">{row}{error}{kids_html}</div>'


def _siblings_html(spans: list[Span], children: dict[str | None, list[Span]]) -> str:
    """Render `spans` (one span's children, or the trace's roots) in order.

    A handoff span gets a divider right after it: every span until the next handoff (if any) is
    the target agent's, not a child of the handoff itself -- `run_loop.py` parents them all to the
    same turn-level span, handoff or not, so this is the one place that distinction is visible.
    """
    items = []
    for span in spans:
        items.append(f"<li>{_render_span(span, children)}</li>")
        if span.type == "handoff":
            name = escape(_span_display_name(span))
            items.append(f'<li class="handoff-divider">Handoff &middot; {name}</li>')
    return "".join(items)


def _tree(trace: Trace) -> str:
    children = _children_map(trace.spans)
    roots = children.get(None, [])
    if not roots:
        return empty("no spans recorded")
    return f'<ul class="span-tree">{_siblings_html(roots, children)}</ul>'


def _add_to_evals(trace: Trace, *, root: Path) -> str:
    """The "Add to evals" form, or where the case already is once it's been added.

    Nothing to add when the trace recorded no user input (see `cli/eval.py`'s `traced_input`).
    """
    traced = traced_input(trace)
    if traced is None:
        return ""
    agent_name, input = traced
    eval_file = f"evals/{agent_name}.jsonl"
    if has_case(root / eval_file, input):
        run = "<code>runa eval</code>"
        return f'<div class="card">In <code>{escape(eval_file)}</code>. Run {run}.</div>'
    action = escape(f"/traces/{quote(trace.id)}/eval")
    return (
        f'<form class="card add-eval" method="post" action="{action}">'
        f'<div class="field-label">Add this input to <code>{escape(eval_file)}</code></div>'
        '<input name="expected" placeholder="What a good answer says (optional)">'
        '<button type="submit">Add to evals</button>'
        "</form>"
    )


def render_detail(trace_id: str, *, root: Path) -> str:
    """Render `/traces/{trace_id}`: that trace's header, its span waterfall, and "Add to evals".

    No nav tab is active here: this page is reached from a session's trace card ("open trace"), an
    evaluation case, or a direct link, not browsed from a list, so nothing in `NAV_ITEMS` does.
    """
    trace = get_trace(trace_id, db_path=resolve_db_path(root))
    if trace is None:
        raise TraceNotFound(f"no trace found with id {trace_id!r}")
    status = "ok" if trace.status == "ok" else "error"
    session_link = (
        f' · session <a href="/sessions/{escape(trace.session_id)}">{escape(trace.session_id)}</a>'
        if trace.session_id
        else ""
    )
    header = (
        f"<h1>{escape(trace.name)} {chip(trace.status, status)}</h1>"
        f'<p class="subtitle">{escape(trace.id)} · {escape(_fmt_duration(trace.duration))}'
        f"{session_link}</p>"
    )
    body = header + _add_to_evals(trace, root=root) + _tree(trace)
    return page(title=trace.name, active="", body=body)
