"""web/sessions.py: the Sessions pages, conversation transcripts from this deployment's store.

Data comes from the `SessionStore` and `TraceStore` `runa.db` resolves for this deployment;
`render_detail` merges the two into one story, sorted by timestamp, rather than showing messages
and traces as two separate lists a reader has to cross-reference by hand.

The listing is newest first, which is the order the store returns and the order `runa chat --list`
prints. It used to be this page reversing an oldest-first read of its own.
"""

from datetime import UTC, datetime
from pathlib import Path

from runa import db
from runa.session.store import SessionMessage, SessionNotFound
from runa.tracing import Trace
from runa.web._html import back_link, empty, empty_hint, escape, page
from runa.web.traces import _tree

__all__ = ["SessionNotFound", "render_detail", "render_list"]


def render_list(*, root: Path) -> str:
    """Render `/sessions`: every session id this deployment has, most recently updated first."""
    sessions = db.sessions(root).listing()
    if not sessions:
        body = empty_hint("no sessions yet, run", "runa chat <agent>")
    else:
        items = "".join(
            f'<a class="row" href="/sessions/{escape(session.id)}">'
            f'<span class="primary">{escape(session.id)}</span>'
            f'<span class="meta">{escape(session.updated_at)}</span></a>'
            for session in sessions
        )
        body = f'<div class="list">{items}</div>'
    return page(
        title="Sessions",
        active="Sessions",
        body=f'<h1>Sessions</h1><p class="subtitle">Conversation history persisted across '
        f"turns.</p>{body}",
    )


def _epoch(created_at: str) -> float:
    """`created_at`'s epoch seconds, comparable with `Trace.start_time`/`end_time` (both UTC)."""
    parsed = datetime.fromisoformat(created_at)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _bubble(message: SessionMessage) -> str:
    return (
        f'<div class="bubble"><div class="role">{escape(message.role)}</div>'
        f'<div class="text">{escape(message.text)}</div>'
        f'<div class="time">{escape(message.created_at)}</div></div>'
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
        f"turn {turn} &middot; {escape(trace.elapsed)} &middot; "
        f"{span_count} span{'' if span_count == 1 else 's'}</summary>"
        f'<div class="trace-card-tree">{_tree(trace)}</div></details>'
    )


def render_detail(session_id: str, *, root: Path) -> str:
    """Render `/sessions/{session_id}`: its messages and traces, merged into one timeline."""
    messages = db.sessions(root).messages(session_id)
    traces = db.traces(root).list(session_id=session_id)
    turn_of = {
        trace.id: turn
        for turn, trace in enumerate(sorted(traces, key=lambda trace: trace.start_time), start=1)
    }

    events: list[tuple[float, int, str]] = [
        (_epoch(message.created_at), 1, _bubble(message)) for message in messages
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
