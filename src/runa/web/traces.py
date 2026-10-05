"""web/traces.py: the trace detail page -- one run's span tree, from this deployment's store.

Data comes straight from the `TraceStore` `runa.db.traces(root)` hands back, the same one
`cli/traces.py show` reads; this module only turns a `Trace`'s `Span` tree into an HTML
waterfall. No standalone list page: `web/sessions.py`'s merged timeline is the primary way to
reach a trace; this is the "open trace"/direct-by-id destination (see `web/app.py`'s docstring).
"""

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from runa import db
from runa.eval.corpus import has_case, traced_input
from runa.tracing import SpanRow, Trace, TraceNotFound
from runa.tracing.spans import Span
from runa.web._html import chip, empty, escape, page, pre

__all__ = ["render_detail"]


def _format_value(value: Any) -> str:
    """Pretty-print a span's `input`/`output` for display.

    `tracing/sqlite.py` round-trips a dict/list `input`/`output` through SQLite as compact JSON
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


def _render_row(row: SpanRow) -> str:
    dot = "ok" if row.span.status == "ok" else "error"
    tokens_html = f'<span class="span-tokens">{escape(row.tokens)}</span>' if row.tokens else ""
    row_content = (
        f'<span class="dot {dot}"></span>'
        f'<span class="span-type">{escape(row.label)}</span>'
        f'<span class="span-name">{escape(row.name)}</span>'
        f'<span class="span-duration">{escape(row.duration)}</span>'
        f"{tokens_html}"
    )
    body = _span_body(row.span)
    rendered = (
        f'<details><summary class="span-row">{row_content}</summary>'
        f'<div class="span-body">{body}</div></details>'
        if body
        else f'<div class="span-row">{row_content}</div>'
    )
    error = f'<div class="error-text">{escape(row.span.error)}</div>' if row.span.error else ""
    kids = f"<ul>{_siblings_html(row.children)}</ul>" if row.children else ""
    return f'<div class="span-node">{rendered}{error}{kids}</div>'


def _siblings_html(rows: tuple[SpanRow, ...]) -> str:
    """Render one row's children, or the trace's roots, in order.

    A row that `hands_off` gets a divider right after it, because everything following it belongs
    to the agent it handed to (see `SpanRow.hands_off`); drawing that boundary is this page's
    choice, knowing which rows have one is not.
    """
    items = []
    for row in rows:
        items.append(f"<li>{_render_row(row)}</li>")
        if row.hands_off:
            items.append(f'<li class="handoff-divider">Handoff &middot; {escape(row.name)}</li>')
    return "".join(items)


def _tree(trace: Trace) -> str:
    roots = trace.walk()
    if not roots:
        return empty("no spans recorded")
    return f'<ul class="span-tree">{_siblings_html(roots)}</ul>'


def _add_to_evals(trace: Trace, *, root: Path) -> str:
    """The "Add to evals" form, or where the case already is once it's been added.

    Nothing to add when the trace recorded no user input (see `eval/corpus.py`'s `traced_input`).
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
    trace = db.traces(root).get(trace_id)
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
        f'<p class="subtitle">{escape(trace.id)} · {escape(trace.elapsed)}'
        f"{session_link}</p>"
    )
    body = header + _add_to_evals(trace, root=root) + _tree(trace)
    return page(title=trace.name, active="", body=body)
