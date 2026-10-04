"""The `TraceStore` contract, run against every adapter that needs no server.

One set of assertions, parametrized over the backends `runa.db` can resolve without a live
Postgres. That is the point of the seam: a trace saved through a `TraceStore` reads back the same
way whichever store it was, so `runa traces`, `runa ui` and the exporter cannot be looking at a
history that behaves differently from the one they were tested against.

`tests/test_postgres_observability.py` runs the same contract against `PostgresTraceStore`, which
is the one adapter that still needs a server to be running.
"""

from pathlib import Path

import pytest

from runa import db
from runa.tracing import Span, Trace
from runa.tracing.store import TraceStore


@pytest.fixture(params=["sqlite", "ephemeral"])
def store(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TraceStore:
    """A `TraceStore`, resolved by `runa.db` the way an app's would be.

    Built through `db.traces(...)` rather than by naming an adapter, so the resolution the rest
    of Runa depends on is exercised by every test here too.
    """
    if request.param == "ephemeral":
        monkeypatch.setenv(db.DATABASE_URL_ENV, "memory://")
        return db.traces()
    monkeypatch.delenv(db.DATABASE_URL_ENV, raising=False)
    return db.traces(tmp_path)


def _trace(
    id: str,
    name: str = "SupportAgent",
    status: str = "ok",
    start_time: float = 0.0,
    session_id: str | None = None,
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


def test_save_then_get_round_trips_the_span(store: TraceStore) -> None:
    """`get` returns a saved trace with its span's fields intact."""
    store.save(_trace("t1"))

    fetched = store.get("t1")

    assert fetched is not None
    assert fetched.name == "SupportAgent"
    assert fetched.status == "ok"
    assert len(fetched.spans) == 1
    assert fetched.spans[0].name == "SupportAgent"


def test_get_returns_none_for_an_unknown_id(store: TraceStore) -> None:
    """Looking up a trace id that was never saved returns `None`, not an error."""
    assert store.get("nope") is None


def test_save_replaces_a_trace_with_the_same_id(store: TraceStore) -> None:
    """A second `save` of one id updates it rather than adding a duplicate."""
    store.save(_trace("t1", name="First"))
    store.save(_trace("t1", name="Second"))

    assert [trace.name for trace in store.list()] == ["Second"]


def test_list_orders_newest_first_and_respects_limit(store: TraceStore) -> None:
    """`list` returns the most recently started traces first, capped at `limit`."""
    for index in range(3):
        store.save(_trace(f"t{index}", name=f"Agent{index}", start_time=float(index)))

    assert [trace.id for trace in store.list(limit=2)] == ["t2", "t1"]


def test_list_filters_by_agent_and_status(store: TraceStore) -> None:
    """`list(agent=..., status=...)` filters on `Trace.name`/`Trace.status`."""
    store.save(_trace("t1", start_time=0.0))
    store.save(_trace("t2", status="error", start_time=1.0))
    store.save(_trace("t3", name="OtherAgent", start_time=2.0))

    assert [t.id for t in store.list(agent="SupportAgent")] == ["t2", "t1"]
    assert [t.id for t in store.list(status="error")] == ["t2"]


def test_session_id_round_trips_and_filters_list(store: TraceStore) -> None:
    """`Trace.session_id` survives a save/get round trip and filters `list`."""
    store.save(_trace("t1", session_id="s1"))
    store.save(_trace("t2", start_time=1.0, session_id="s2"))
    store.save(_trace("t3", start_time=2.0))

    first = store.get("t1")
    third = store.get("t3")

    assert first is not None and first.session_id == "s1"
    assert third is not None and third.session_id is None
    assert [trace.id for trace in store.list(session_id="s1")] == ["t1"]


def test_structured_span_input_comes_back_as_text(store: TraceStore) -> None:
    """A span's dict `input` is stored as JSON text, identically in every backend.

    The assertion that keeps the ephemeral adapter honest: it holds marshalled rows rather than
    the `Trace` it was handed, so a caller cannot get a dict back from one store and a string
    from another.
    """
    trace = _trace("t1")
    trace.spans[0].input = {"city": "Paris"}

    store.save(trace)
    fetched = store.get("t1")

    assert fetched is not None
    assert fetched.spans[0].input == '{"city": "Paris"}'


def test_a_trace_a_caller_mutates_does_not_change_the_history(store: TraceStore) -> None:
    """Editing a `Trace` that came out of `get` leaves the stored one alone."""
    store.save(_trace("t1"))

    fetched = store.get("t1")
    assert fetched is not None
    fetched.name = "Mutated"

    again = store.get("t1")
    assert again is not None
    assert again.name == "SupportAgent"
