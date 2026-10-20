"""The `TraceStore` contract, run against every backend that needs no server.

The checks live in `tests/contracts/trace.py` and are driven here over the backends `runa.db` can
resolve without a live Postgres. That is the point of the seam: a trace saved through a
`TraceStore` reads back the same way whichever store it was, so `runa traces`, `runa ui` and the
exporter cannot be looking at a history that behaves differently from the one they were tested
against.

`tests/test_postgres_observability.py` runs the same checks over a shared Postgres, which is the
one backend that needs a server to be running.
"""

import asyncio
from pathlib import Path

import pytest
from contracts.trace import CONTRACT, Check

from runa import db
from runa.project import use
from runa.tracing.store import TraceStore, listing_clause


@pytest.fixture(params=["sqlite", "memory"])
def store(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TraceStore:
    """A `TraceStore`, resolved by `runa.db` the way an app's would be.

    Built through `db.traces()` rather than by naming an adapter, so the resolution the rest
    of Runa depends on is exercised by every check here too.
    """
    if request.param == "memory":
        monkeypatch.setenv(db.DATABASE_URL_ENV, "memory://")
        return db.traces()
    monkeypatch.delenv(db.DATABASE_URL_ENV, raising=False)
    use(tmp_path)
    return db.traces()


@pytest.mark.parametrize("check", CONTRACT, ids=lambda check: check.__name__)
def test_trace_store_contract(store: TraceStore, check: Check) -> None:
    """Every local backend answers the `TraceStore` contract the same way."""
    asyncio.run(check(store, "t"))


def test_listing_clause_numbers_the_limit_after_the_filters() -> None:
    """The cap is bound last, which is why the clause carries it rather than leaving it out.

    A caller handed the `WHERE` alone would have to count the filters that went into it to
    number its own `LIMIT`, and that arithmetic is the thing worth writing once.
    """
    clause, params = listing_clause(limit=5, agent="SupportAgent", session_id="user-42")

    assert clause == (
        "WHERE name = $1 AND session_id = $2 ORDER BY start_time DESC, id DESC LIMIT $3"
    )
    assert params == ("SupportAgent", "user-42", 5)


def test_listing_clause_with_no_filters_is_still_a_whole_tail() -> None:
    """No filter is no `WHERE`, so a listing composes the same string either way."""
    clause, params = listing_clause(limit=50)

    assert clause == "ORDER BY start_time DESC, id DESC LIMIT $1"
    assert params == (50,)
