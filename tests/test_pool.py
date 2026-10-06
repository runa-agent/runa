"""Tests for `db/pool.py`'s two pairings, neither of which needs a live database.

`tests/test_postgres.py` and `tests/test_postgres_observability.py` drive the adapters built on
these against a real Postgres, and skip wholesale when there isn't one. What is checked here is
the pairing itself, on the machine of anyone running `make test`: that a synchronous caller
inside a running loop still reaches an async body, and that each adapter hands `connect` its own
tables rather than another concern's.
"""

import asyncio

import pytest


def test_sync_reaches_an_async_body_from_inside_a_running_loop() -> None:
    """The case `asyncio.run` can't serve: a synchronous store method called mid-run."""
    pytest.importorskip("asyncpg")
    from runa.db.pool import sync

    class Store:
        def __init__(self) -> None:
            self.saved: list[str] = []

        @sync
        async def save(self, item: str) -> str:
            """Record `item`, as a `TraceExporter` would."""
            self.saved.append(item)
            return "saved"

    store = Store()

    async def mid_run() -> str:
        # Where `TraceExporter.export` is called from: a finishing run, so a loop is already
        # running on this thread, which is exactly what `asyncio.run` refuses to nest into.
        return store.save("trace-1")

    assert asyncio.run(mid_run()) == "saved"
    assert store.save("trace-2") == "saved"  # and from no loop at all, as `runa traces` does
    assert store.saved == ["trace-1", "trace-2"]


def test_sync_keeps_the_async_bodys_name_and_docstring() -> None:
    """The decorated method is still the documented one: `runa ui`'s help and docs read it."""
    pytest.importorskip("asyncpg")
    from runa.eval.postgres import PostgresEvalStore

    assert PostgresEvalStore.save.__name__ == "save"
    assert PostgresEvalStore.baseline.__doc__ is not None
    assert "latest run" in PostgresEvalStore.baseline.__doc__


def test_a_shared_store_connects_to_its_own_url_with_its_own_ddl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_pool` is the one place `(url, ddl)` becomes a pool, where twelve call sites used to be."""
    pytest.importorskip("asyncpg")
    from runa.db import pool

    calls: list[tuple[str, str]] = []

    async def _fake_connect(url: str, ddl: str) -> str:
        calls.append((url, ddl))
        return "pool"

    monkeypatch.setattr(pool, "connect", _fake_connect)
    store = pool.Shared("postgresql://example/db", "CREATE TABLE IF NOT EXISTS mine ()")

    assert asyncio.run(store._pool()) == "pool"
    assert calls == [("postgresql://example/db", "CREATE TABLE IF NOT EXISTS mine ()")]


def test_every_postgres_adapter_brings_its_own_tables() -> None:
    """Each adapter hands up the DDL for the tables it owns, five concerns in one database."""
    pytest.importorskip("asyncpg")
    from runa.cache.postgres import PostgresCache
    from runa.db.vectors.postgres import PostgresVectorStore
    from runa.eval.postgres import PostgresEvalStore
    from runa.memory.vector import spec
    from runa.session.postgres import PostgresSession, PostgresSessionStore
    from runa.tracing.postgres import PostgresTraceStore

    url = "postgresql://example/db"

    assert "cache_entries" in PostgresCache(url)._ddl
    assert "eval_runs" in PostgresEvalStore(url)._ddl
    assert "spans" in PostgresTraceStore(url)._ddl
    assert "agent_messages" in PostgresSessionStore(url)._ddl
    # Both sides of the session pair reach the same two tables, from different base `__init__`s.
    assert PostgresSession("s", url)._ddl == PostgresSessionStore(url)._ddl
    # The vector store's DDL is computed from its spec, so it can only be handed over once the
    # spec is stored: a store built before that would create a table named `_items`.
    vectors = PostgresVectorStore(spec(4), url)
    assert "memory_items" in vectors._ddl
    assert "vector(4)" in vectors._ddl
