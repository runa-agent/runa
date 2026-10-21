"""Shared test fixtures.

Every test stands in for a fresh process, so every test ends with `runa._app.reset()`: the one
call that forgets every process fact Runa holds -- the app `project.use` named and its import,
tracing's default policy, and every `memory://` table. A new piece of process state is held there
too, so it is reset here without a fixture of its own.

Tracing is automatic (see `runa.tracing`), so any test that drives a real run through
`Agent.run`/`run_sync`/`run_streamed` or `evaluate_agent` produces a real finished trace
that the default store exporter persists wherever `runa.db` resolved. No test should touch the
real `runa.db` in the process's cwd, and a paused run is kept there too, so every test runs under
`memory://` unless it sets its own `RUNA_DATABASE_URL`; a test of the default itself, a project's
own `db/runa.db`, asks for `project_database`. A test that watches exports calls `add_exporter`:
the reset below forgets it with the rest of the process.
"""

import pytest

from runa import _app, db


@pytest.fixture(autouse=True)
def _fresh_process(monkeypatch: pytest.MonkeyPatch):
    """Start every test as a fresh process under `memory://`, and forget it all afterwards."""
    _app.reset()
    monkeypatch.setenv(db.DATABASE_URL_ENV, "memory://")
    yield
    _app.reset()


@pytest.fixture
def project_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset `RUNA_DATABASE_URL`, so state lands in the project's own `db/runa.db` by default."""
    monkeypatch.delenv(db.DATABASE_URL_ENV, raising=False)
