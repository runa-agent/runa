"""Shared test fixtures.

Tracing is automatic (see `runa.tracing`), so any test that drives a real run through
`Agent.run`/`run_sync`/`run_streamed` or `run_agent_for_eval` produces a real finished trace
that the default store exporter would otherwise persist to `runa.db` in the process's cwd. No
test should touch that real file, so every test runs with trace export disabled by default; a
test that specifically wants to exercise an exporter (e.g. a fail-open test) can still pass its
own `exporter=` to `observe(...)`, which simply overrides this for the scope of its `with` block.

That default is installed with the bare form, not a `with` block, because process-wide is what it
means: a `with observe(...)` is scoped to one task (see `tracing/config.py`), so held open around
a test it would say nothing about the default the test's own tasks inherit, and would mask a bare
`observe(...)` call under test.

The `memory://` backend keeps its tables at module level, the way a database keeps them on disk,
so they outlive a test the same way a file would. Emptying them around every test is what makes
`RUNA_DATABASE_URL=memory://` usable as a fixture: a test gets an empty store whether or not the
one before it used the same backend.
"""

import pytest

from runa import db
from runa.tracing import config, observe


@pytest.fixture(autouse=True)
def _no_default_trace_export():
    """Disable the default store exporter for the duration of every test."""
    previous = config._default
    observe(exporter=[])
    yield
    config._set_default(previous)


@pytest.fixture(autouse=True)
def _empty_ephemeral_stores():
    """Empty every `memory://` store before and after each test."""
    db.reset_ephemeral()
    yield
    db.reset_ephemeral()
