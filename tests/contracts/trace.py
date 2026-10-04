"""contracts/trace.py: the one `TraceStore` contract, so every backend is held to it.

A trace saved through a `TraceStore` has to read back the same way whichever store it was, or
`runa traces`, `runa ui` and the exporter are looking at a history that behaves differently from
the one they were tested against.

Each check takes the store and one tag it is free to write under; every trace id, agent name and
session id it uses is derived from that tag, and every listing it asserts on is filtered by one of
them, so the same checks hold against a live Postgres whose tables already hold other runs' rows.
"""

from collections.abc import Callable

from runa.tracing import Span, Trace
from runa.tracing.store import TraceStore

Check = Callable[[TraceStore, str], None]


def _trace(
    id: str,
    *,
    name: str = "SupportAgent",
    status: str = "ok",
    start_time: float = 0.0,
    session_id: str | None = None,
) -> Trace:
    """One finished trace with one span, the smallest thing worth saving."""
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


def check_save_then_get_round_trips_the_span(store: TraceStore, tag: str) -> None:
    """`get` returns a saved trace with its span's fields intact."""
    store.save(_trace(f"{tag}-1", name=tag))

    fetched = store.get(f"{tag}-1")

    assert fetched is not None
    assert fetched.name == tag
    assert fetched.status == "ok"
    assert len(fetched.spans) == 1
    assert fetched.spans[0].name == tag


def check_get_returns_none_for_an_unknown_id(store: TraceStore, tag: str) -> None:
    """Looking up a trace id that was never saved returns `None`, not an error."""
    assert store.get(f"{tag}-never-saved") is None


def check_save_replaces_a_trace_with_the_same_id(store: TraceStore, tag: str) -> None:
    """A second `save` of one id updates it rather than adding a duplicate trace or span."""
    trace = _trace(f"{tag}-1", name=f"{tag}-first")
    store.save(trace)
    trace.name = f"{tag}-second"
    store.save(trace)

    assert store.list(agent=f"{tag}-first") == []
    renamed = store.list(agent=f"{tag}-second")
    assert [found.id for found in renamed] == [f"{tag}-1"]
    assert len(renamed[0].spans) == 1


def check_list_orders_newest_first_and_respects_limit(store: TraceStore, tag: str) -> None:
    """`list` returns the most recently started traces first, capped at `limit`."""
    for index in range(3):
        store.save(_trace(f"{tag}-{index}", name=tag, start_time=float(index)))

    assert [found.id for found in store.list(agent=tag, limit=2)] == [f"{tag}-2", f"{tag}-1"]


def check_list_filters_by_agent_and_status(store: TraceStore, tag: str) -> None:
    """`list(agent=..., status=...)` filters on `Trace.name`/`Trace.status`."""
    store.save(_trace(f"{tag}-1", name=tag, start_time=0.0))
    store.save(_trace(f"{tag}-2", name=tag, status="error", start_time=1.0))
    store.save(_trace(f"{tag}-3", name=f"{tag}-other", start_time=2.0))

    assert [found.id for found in store.list(agent=tag)] == [f"{tag}-2", f"{tag}-1"]
    assert [found.id for found in store.list(agent=tag, status="error")] == [f"{tag}-2"]


def check_session_id_round_trips_and_filters_list(store: TraceStore, tag: str) -> None:
    """`Trace.session_id` survives a save/get round trip and filters `list`."""
    store.save(_trace(f"{tag}-1", name=tag, session_id=f"{tag}-s1"))
    store.save(_trace(f"{tag}-2", name=tag, start_time=1.0, session_id=f"{tag}-s2"))
    store.save(_trace(f"{tag}-3", name=tag, start_time=2.0))

    first = store.get(f"{tag}-1")
    third = store.get(f"{tag}-3")

    assert first is not None and first.session_id == f"{tag}-s1"
    assert third is not None and third.session_id is None
    assert [found.id for found in store.list(session_id=f"{tag}-s1")] == [f"{tag}-1"]


def check_structured_span_input_comes_back_as_text(store: TraceStore, tag: str) -> None:
    """A span's dict `input` is stored as JSON text, identically in every backend.

    The assertion that keeps the ephemeral adapter honest: it holds marshalled rows rather than
    the `Trace` it was handed, so a caller cannot get a dict back from one store and a string
    from another.
    """
    trace = _trace(f"{tag}-1", name=tag)
    trace.spans[0].input = {"city": "Paris"}

    store.save(trace)
    fetched = store.get(f"{tag}-1")

    assert fetched is not None
    assert fetched.spans[0].input == '{"city": "Paris"}'


def check_each_listed_trace_carries_only_its_own_spans(store: TraceStore, tag: str) -> None:
    """A listing never cross-assigns spans, however the backend fetches them.

    Postgres fetches every listed trace's spans in one query where SQLite fetches them per trace,
    which is exactly the kind of difference a shared contract is for.
    """
    store.save(_trace(f"{tag}-1", name=tag, start_time=0.0))
    store.save(_trace(f"{tag}-2", name=tag, start_time=1.0))

    found = store.list(agent=tag)

    assert len(found) == 2
    for trace in found:
        assert [span.trace_id for span in trace.spans] == [trace.id]


def check_an_empty_listing_is_an_empty_list(store: TraceStore, tag: str) -> None:
    """No traces for an agent is an empty list, not an error or a fetch for nothing."""
    assert store.list(agent=f"{tag}-never-ran") == []


def check_a_trace_a_caller_mutates_does_not_change_the_history(store: TraceStore, tag: str) -> None:
    """Editing a `Trace` that came out of `get` leaves the stored one alone."""
    store.save(_trace(f"{tag}-1", name=tag))

    fetched = store.get(f"{tag}-1")
    assert fetched is not None
    fetched.name = "Mutated"

    again = store.get(f"{tag}-1")
    assert again is not None
    assert again.name == tag


CONTRACT: list[Check] = [
    check_save_then_get_round_trips_the_span,
    check_get_returns_none_for_an_unknown_id,
    check_save_replaces_a_trace_with_the_same_id,
    check_list_orders_newest_first_and_respects_limit,
    check_list_filters_by_agent_and_status,
    check_session_id_round_trips_and_filters_list,
    check_structured_span_input_comes_back_as_text,
    check_each_listed_trace_carries_only_its_own_spans,
    check_an_empty_listing_is_an_empty_list,
    check_a_trace_a_caller_mutates_does_not_change_the_history,
]
