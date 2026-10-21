"""Tests for the local `PausedRunStore` backends: a SQLite file and `memory://`'s.

The contract itself lives in `tests/contracts/paused_run.py` and is driven here over every backend
a plain `make test` can reach. A shared Postgres is held to the same checks in
`test_postgres.py`, which needs a live database.
"""

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest
from contracts.paused_run import CONTRACT, Check

from runa import db
from runa.db.sqlite import SQLiteDatabase
from runa.run_state.store import PAUSED_RUNS, PausedRunStore


def _store_at(db_path: Path) -> PausedRunStore:
    """A paused-run store over one specific SQLite file."""
    return PausedRunStore(SQLiteDatabase(db_path, PAUSED_RUNS))


_BACKENDS: dict[str, Callable[[Path], PausedRunStore]] = {
    "sqlite": lambda tmp_path: _store_at(tmp_path / "runa.db"),
    "shared": lambda tmp_path: db.paused_runs(),
}


@pytest.mark.parametrize("check", CONTRACT, ids=lambda check: check.__name__)
@pytest.mark.parametrize("backend", list(_BACKENDS), ids=list(_BACKENDS))
def test_paused_run_store_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, check: Check
) -> None:
    """Every local backend answers the `PausedRunStore` contract the same way."""
    monkeypatch.setenv(db.DATABASE_URL_ENV, "memory://")
    store = _BACKENDS[backend](tmp_path)

    asyncio.run(check(store, "refund_agent"))


def test_loading_from_a_project_with_no_database_creates_none(tmp_path: Path) -> None:
    """A miss is a read, so it doesn't leave a `db/runa.db` behind."""
    db_path = tmp_path / "db" / "runa.db"

    assert asyncio.run(_store_at(db_path).load("refund_agent", "nope")) is None
    assert not db_path.exists()


def test_a_pause_survives_the_store_that_saved_it(tmp_path: Path) -> None:
    """Another process (here, another instance) on the same file resumes it."""
    db_path = tmp_path / "runa.db"

    async def _run():
        run_id = await _store_at(db_path).save("refund_agent", {"a": 1})
        return await _store_at(db_path).load("refund_agent", run_id)

    assert asyncio.run(_run()) == {"a": 1}
