"""The `TraceStore` contract, run against every adapter that needs no server.

The checks live in `tests/contracts/trace.py` and are driven here over the backends `runa.db` can
resolve without a live Postgres. That is the point of the seam: a trace saved through a
`TraceStore` reads back the same way whichever store it was, so `runa traces`, `runa ui` and the
exporter cannot be looking at a history that behaves differently from the one they were tested
against.

`tests/test_postgres_observability.py` runs the same checks against `PostgresTraceStore`, which is
the one adapter that needs a server to be running.
"""

from pathlib import Path

import pytest
from contracts.trace import CONTRACT, Check

from runa import db
from runa.tracing.store import TraceStore


@pytest.fixture(params=["sqlite", "ephemeral"])
def store(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TraceStore:
    """A `TraceStore`, resolved by `runa.db` the way an app's would be.

    Built through `db.traces()` rather than by naming an adapter, so the resolution the rest
    of Runa depends on is exercised by every check here too.
    """
    if request.param == "ephemeral":
        monkeypatch.setenv(db.DATABASE_URL_ENV, "memory://")
        return db.traces()
    monkeypatch.delenv(db.DATABASE_URL_ENV, raising=False)
    db.use_project(tmp_path)
    return db.traces()


@pytest.mark.parametrize("check", CONTRACT, ids=lambda check: check.__name__)
def test_trace_store_contract(store: TraceStore, check: Check) -> None:
    """Every local backend answers the `TraceStore` contract the same way."""
    check(store, "t")
