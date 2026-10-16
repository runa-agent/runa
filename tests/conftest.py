"""Shared test fixtures.

Every test stands in for a fresh process, so every test ends with `runa._app.reset()`: the one
call that forgets every process fact Runa holds -- the app `project.use` named and its import,
tracing's default policy, and every `memory://` table. A new piece of process state is held there
too, so it is reset here without a fixture of its own.

Tracing is automatic (see `runa.tracing`), so any test that drives a real run through
`Agent.run`/`run_sync`/`run_streamed` or `evaluate_agent` produces a real finished trace
that the default store exporter would otherwise persist to `runa.db` in the process's cwd. No
test should touch that real file, so every test runs with trace export disabled by default; a
test that specifically wants to exercise an exporter (e.g. a fail-open test) can still pass its
own `exporter=` to `observe.scoped(...)`, which overrides this for the scope of its `with` block.

That default is installed with `observe(...)`, not an `observe.scoped(...)` block, because
process-wide is what it means: a scoped block covers one task (see `tracing/config.py`), so held
open around a test it would say nothing about the default the test's own tasks inherit, and would
mask an `observe(...)` call under test.
"""

import pytest

from runa import _app
from runa.tracing import observe


@pytest.fixture(autouse=True)
def _fresh_process():
    """Start every test as a fresh process with trace export off, and forget it all afterwards."""
    _app.reset()
    observe(exporter=[])
    yield
    _app.reset()
