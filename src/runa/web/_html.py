"""web/_html.py: the page shell, shared CSS, and small render helpers every `web/*.py` page uses.

Hand-rolled strings, not a templating engine -- four small pages don't earn a Jinja dependency,
and it keeps every page a plain, testable function (the same spirit as `Trace.__str__`). No
JavaScript: `<details>` covers every collapsible bit a trace/session/eval page needs.
"""

from __future__ import annotations

import html
from typing import Literal

NAV_ITEMS = ("Agents", "Sessions", "Evaluations")

_STYLE = """
:root {
  --bg: #fafafa; --surface: #ffffff; --border: #e4e4e7; --text: #18181b; --muted: #71717a;
  --accent: #4f46e5; --ok: #16a34a; --error: #dc2626; --mono: ui-monospace, SFMono-Regular,
  Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0b0b0d; --surface: #18181b; --border: #29292e; --text: #e4e4e7; --muted: #9a9aa2;
    --accent: #818cf8; --ok: #4ade80; --error: #f87171;
  }
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--text); margin: 0;
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
a { color: inherit; text-decoration: none; }
header {
  position: sticky; top: 0; z-index: 10; background: var(--bg);
  border-bottom: 1px solid var(--border); padding: 0 24px;
}
.header-inner {
  max-width: 1040px; margin: 0 auto; display: flex; align-items: center; gap: 28px; height: 56px;
}
.brand { font-weight: 600; letter-spacing: -0.02em; font-size: 15px; }
nav { display: flex; gap: 4px; }
nav a {
  padding: 6px 12px; border-radius: 7px; color: var(--muted); font-size: 13.5px; font-weight: 500;
}
nav a:hover { color: var(--text); }
nav a.active { color: var(--accent); font-weight: 600;
  background: color-mix(in srgb, var(--accent) 12%, var(--surface)); }
main { max-width: 1040px; margin: 0 auto; padding: 32px 24px 80px; }
h1 { font-size: 20px; margin: 0 0 4px; letter-spacing: -0.01em; }
.subtitle { color: var(--muted); font-size: 13.5px; margin: 0 0 24px; }
.empty { color: var(--muted); padding: 48px 0; text-align: center; font-size: 14px; }
.list { display: flex; flex-direction: column; gap: 1px; background: var(--border);
  border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
.row { background: var(--surface); padding: 13px 16px; display: flex; align-items: center;
  gap: 12px; }
a.row:hover { background: var(--bg); }
.row .primary { font-weight: 500; flex: 1; min-width: 0; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.row .meta { color: var(--muted); font-size: 12.5px; font-family: var(--mono); }
.dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dot.ok { background: var(--ok); } .dot.error { background: var(--error); }
.chip { display: inline-flex; align-items: center; padding: 2px 8px; border-radius: 999px;
  font-size: 11.5px; font-weight: 500; background: var(--bg); border: 1px solid var(--border);
  color: var(--muted); }
.chip.ok { color: var(--ok);
  border-color: color-mix(in srgb, var(--ok) 40%, var(--border)); }
.chip.error { color: var(--error);
  border-color: color-mix(in srgb, var(--error) 40%, var(--border)); }
.chip.accent { color: var(--accent);
  border-color: color-mix(in srgb, var(--accent) 40%, var(--border)); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 18px 20px; margin-bottom: 16px; }
.card h2 { font-size: 15px; margin: 0 0 10px; }
.field-label { color: var(--muted); font-size: 11.5px; text-transform: uppercase;
  letter-spacing: 0.04em; margin-bottom: 4px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
pre { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px;
  font-family: var(--mono); font-size: 12.5px; overflow-x: auto; white-space: pre-wrap;
  word-break: break-word; margin: 8px 0 0; }
code { font-family: var(--mono); font-size: 0.92em; background: var(--surface);
  border: 1px solid var(--border); border-radius: 4px; padding: 1px 5px; }
details summary { cursor: pointer; color: var(--muted); font-size: 12.5px; user-select: none; }
details summary:hover { color: var(--text); }
.back { color: var(--muted); font-size: 13px; margin-bottom: 16px; display: inline-block; }
.back:hover { color: var(--text); }
.span-tree, .span-tree ul { list-style: none; margin: 0; padding-left: 22px; }
.span-tree { padding-left: 0; }
.span-tree ul { border-left: 1px solid var(--border); }
.span-node { padding: 4px 0; }
.span-row { display: flex; align-items: baseline; gap: 8px; padding: 5px 8px; border-radius: 6px; }
.span-row:hover { background: var(--surface); }
summary.span-row { color: var(--text); font-size: inherit; cursor: pointer; user-select: none; }
.handoff-divider { display: flex; align-items: center; gap: 10px; color: var(--muted);
  font-size: 11px; padding: 6px 8px; }
.handoff-divider::before, .handoff-divider::after { content: ""; flex: 1;
  border-top: 1px dashed var(--border); }
.span-type { font-size: 10.5px; font-weight: 600; text-transform: uppercase; letter-spacing:
  0.03em; color: var(--muted); width: 62px; flex-shrink: 0; }
.span-name { font-weight: 500; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; }
.span-duration { color: var(--muted); font-size: 12px; font-family: var(--mono); }
.span-tokens { color: var(--muted); font-size: 12px; font-family: var(--mono); }
.span-body { padding-left: 78px; }
.error-text { color: var(--error); font-family: var(--mono); font-size: 12.5px; margin: 4px 0
  0 78px; }
.bubble { border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px;
  margin-bottom: 10px; background: var(--surface); }
.bubble .role { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing:
  0.04em; color: var(--accent); margin-bottom: 4px; }
.bubble .text { white-space: pre-wrap; word-break: break-word; }
.bubble .time { color: var(--muted); font-size: 11px; font-family: var(--mono); margin-top: 6px; }
.trace-card { border: 1px dashed var(--border); border-radius: 10px; padding: 10px 14px;
  margin-bottom: 10px; }
.trace-card summary { display: flex; align-items: center; gap: 6px; font-size: 12.5px;
  font-family: var(--mono); color: var(--muted); cursor: pointer; user-select: none; }
.trace-card summary:hover { color: var(--text); }
.trace-card-tree { margin-top: 12px; }
.score-bar { height: 6px; border-radius: 999px; background: var(--border); overflow: hidden;
  width: 120px; }
.score-bar > div { height: 100%; background: var(--ok); }
"""


def escape(value: object) -> str:
    """HTML-escape any value for safe interpolation (span input/output is untrusted app data)."""
    return html.escape(str(value), quote=True)


def page(*, title: str, active: str, body: str) -> str:
    """Wrap `body` in the shared shell: `<head>`/nav/`<main>`, with `active` bolded in the nav."""
    nav = "".join(
        f'<a href="/{item.lower()}" class="{"active" if item == active else ""}">{item}</a>'
        for item in NAV_ITEMS
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)} · runa</title>
<style>{_STYLE}</style>
</head>
<body>
<header><div class="header-inner">
  <a class="brand" href="/agents">runa</a>
  <nav>{nav}</nav>
</div></header>
<main>{body}</main>
</body>
</html>"""


def chip(text: str, kind: Literal["default", "ok", "error", "accent"] = "default") -> str:
    """A small rounded label, e.g. a status or a tag."""
    cls = "chip" if kind == "default" else f"chip {kind}"
    return f'<span class="{cls}">{escape(text)}</span>'


def chips(items: list[str], kind: Literal["default", "ok", "error", "accent"] = "default") -> str:
    """A `chips` row, or a muted "none" chip when `items` is empty."""
    if not items:
        return chip("none")
    return f'<div class="chips">{"".join(chip(item, kind) for item in items)}</div>'


def empty(message: str) -> str:
    """A centered muted placeholder for a page with nothing to show yet."""
    return f'<div class="empty">{escape(message)}</div>'


def empty_hint(message: str, command: str) -> str:
    """`empty`, plus a `<code>`-styled command to run next."""
    return f'<div class="empty">{escape(message)} <code>{escape(command)}</code></div>'


def back_link(href: str, label: str) -> str:
    """A small "‹ back to X" link above a detail page's content."""
    return f'<a class="back" href="{escape(href)}">‹ {escape(label)}</a>'


def pre(text: str) -> str:
    """A `<pre>` block for raw text/JSON, wrapped and escaped."""
    return f"<pre>{escape(text)}</pre>"
