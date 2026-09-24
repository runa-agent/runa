"""web/sessions.py: the Sessions pages -- conversation transcripts from `db/runa.db`.

Data comes from `runa.cli.sessions` (`session_rows`/`session_messages`) and `runa.tracing`
(`list_traces`); `render_detail` merges the two into one story, sorted by timestamp, rather than
showing messages and traces as two separate lists a reader has to cross-reference by hand.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from runa.cli._project import resolve_db_path
from runa.cli.sessions import SessionNotFound, session_messages, session_rows
from runa.tracing import Trace, list_traces
from runa.tracing.traces import _fmt_duration
from runa.web._html import back_link, empty, empty_hint, escape, page
from runa.web.traces import _tree

__all__ = ["SessionNotFound", "render_detail", "render_list"]


def render_list(*, root: Path) -> str:
    """Render `/sessions`: every session id in `db/runa.db`, most recently updated first."""
    rows = list(reversed(session_rows(root=root)))
    if not rows:
        body = empty_hint("no sessions yet, run", "runa chat <agent>")
    else:
        items = "".join(
            f'<a class="row" href="/sessions/{escape(session_id)}">'
            f'<span class="primary">{escape(session_id)}</span>'
            f'<span class="meta">{escape(updated_at)}</span></a>'
            for session_id, updated_at in rows
        )
        body = f'<div class="list">{items}</div>'
    return page(
        title="Sessions",
        active="Sessions",
        body=f'<h1>Sessions</h1><p class="subtitle">Conversation history persisted by '
        f"SQLiteSession.</p>{body}",
    )


def _epoch(created_at: str) -> float:
    """`created_at`'s epoch seconds, comparable with `Trace.start_time`/`end_time` (both UTC)."""
    parsed = datetime.fromisoformat(created_at)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _bubble(message: dict[str, str]) -> str:
    return (
        f'<div class="bubble"><div class="role">{escape(message["role"])}</div>'
        f'<div class="text">{escape(message["text"])}</div>'
        f'<div class="time">{escape(message["created_at"])}</div></div>'
    )


def _trace_card(trace: Trace, turn: int) -> str:
    """A collapsed summary of one turn's trace; folds open in place into its full span tree.

    Labeled by turn number, not `trace.name` (the agent) -- every trace in a session is already
    that one agent's, repeating its name on each card would just be noise.
    """
    dot = "ok" if trace.status == "ok" else "error"
    span_count = len(trace.spans)
    return (
        f'<details class="trace-card"><summary><span class="dot {dot}"></span> trace &middot; '
        f"turn {turn} &middot; {escape(_fmt_duration(trace.duration))} &middot; "
        f"{span_count} span{'' if span_count == 1 else 's'}</summary>"
        f'<div class="trace-card-tree">{_tree(trace)}</div></details>'
    )


def render_detail(session_id: str, *, root: Path) -> str:
    """Render `/sessions/{session_id}`: its messages and traces, merged into one timeline."""
    messages = session_messages(session_id, root=root)
    traces = list_traces(session_id=session_id, db_path=resolve_db_path(root))
    turn_of = {
        trace.id: turn
        for turn, trace in enumerate(sorted(traces, key=lambda trace: trace.start_time), start=1)
    }

    events: list[tuple[float, int, str]] = [
        (_epoch(message["created_at"]), 1, _bubble(message)) for message in messages
    ]
    events += [
        (
            trace.end_time if trace.end_time is not None else trace.start_time,
            0,
            _trace_card(trace, turn_of[trace.id]),
        )
        for trace in traces
    ]
    events.sort(key=lambda event: event[:2])
    timeline = "".join(html for _, _, html in events) or empty("no messages")

    body = back_link("/sessions", "sessions") + f"<h1>{escape(session_id)}</h1>" + timeline
    return page(title=session_id, active="Sessions", body=body)
