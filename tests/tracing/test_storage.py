"""Tests for `runa.tracing.storage`: `save_trace`/`get_trace`/`list_traces`/`get_errors`."""

from pathlib import Path

from runa.tracing import Span, Trace
from runa.tracing.storage import get_errors, get_trace, list_traces, save_trace


def _trace(
    id: str, name: str, status: str, start_time: float = 0.0, session_id: str | None = None
) -> Trace:
    span = Span(
        id=f"{id}-span",
        trace_id=id,
        parent_id=None,
        name=name,
        type="agent",
        start_time=start_time,
        end_time=start_time + 1.0,
        status=status,  # type: ignore[arg-type]
        error="boom" if status == "error" else None,
    )
    return Trace(
        id=id,
        name=name,
        start_time=start_time,
        end_time=start_time + 1.0,
        session_id=session_id,
        spans=[span],
    )


def test_save_trace_then_get_trace_round_trips_the_span(tmp_path: Path) -> None:
    """`get_trace` returns a saved trace with its span's fields intact."""
    db_path = tmp_path / "runa.db"
    trace = _trace("t1", "SupportAgent", "ok")

    save_trace(trace, db_path=db_path)
    fetched = get_trace("t1", db_path=db_path)

    assert fetched is not None
    assert fetched.name == "SupportAgent"
    assert fetched.status == "ok"
    assert len(fetched.spans) == 1
    assert fetched.spans[0].name == "SupportAgent"


def test_get_trace_returns_none_for_an_unknown_id(tmp_path: Path) -> None:
    """Looking up a trace id that was never saved returns `None`, not an error."""
    db_path = tmp_path / "runa.db"

    assert get_trace("nope", db_path=db_path) is None


def test_list_traces_orders_newest_first_and_respects_limit(tmp_path: Path) -> None:
    """`list_traces` returns the most recently started traces first, capped at `limit`."""
    db_path = tmp_path / "runa.db"
    for index in range(3):
        save_trace(
            _trace(f"t{index}", f"Agent{index}", "ok", start_time=float(index)), db_path=db_path
        )

    traces = list_traces(limit=2, db_path=db_path)

    assert [t.id for t in traces] == ["t2", "t1"]


def test_list_traces_filters_by_agent_and_status(tmp_path: Path) -> None:
    """`list_traces(agent=..., status=...)` filters on `Trace.name`/`Trace.status`."""
    db_path = tmp_path / "runa.db"
    save_trace(_trace("t1", "SupportAgent", "ok", start_time=0.0), db_path=db_path)
    save_trace(_trace("t2", "SupportAgent", "error", start_time=1.0), db_path=db_path)
    save_trace(_trace("t3", "OtherAgent", "ok", start_time=2.0), db_path=db_path)

    assert [t.id for t in list_traces(agent="SupportAgent", db_path=db_path)] == ["t2", "t1"]
    assert [t.id for t in list_traces(status="error", db_path=db_path)] == ["t2"]


def test_session_id_round_trips_and_filters_list_traces(tmp_path: Path) -> None:
    """`Trace.session_id` survives a save/get round trip and filters `list_traces`."""
    db_path = tmp_path / "runa.db"
    save_trace(_trace("t1", "SupportAgent", "ok", session_id="s1"), db_path=db_path)
    save_trace(_trace("t2", "SupportAgent", "ok", start_time=1.0, session_id="s2"), db_path=db_path)
    save_trace(_trace("t3", "SupportAgent", "ok", start_time=2.0), db_path=db_path)

    assert get_trace("t1", db_path=db_path).session_id == "s1"  # type: ignore[union-attr]
    assert get_trace("t3", db_path=db_path).session_id is None  # type: ignore[union-attr]
    assert [t.id for t in list_traces(session_id="s1", db_path=db_path)] == ["t1"]


def test_get_errors_returns_only_error_traces(tmp_path: Path) -> None:
    """`get_errors` is exactly `list_traces(status="error")`."""
    db_path = tmp_path / "runa.db"
    save_trace(_trace("t1", "SupportAgent", "ok"), db_path=db_path)
    save_trace(_trace("t2", "SupportAgent", "error"), db_path=db_path)

    errors = get_errors(db_path=db_path)

    assert [t.id for t in errors] == ["t2"]
