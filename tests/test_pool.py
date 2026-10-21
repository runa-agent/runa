"""Tests for the pairings `db/handle.py` and `db/pool.py` make, none of which needs a live database.

`tests/test_postgres.py` and `tests/test_postgres_observability.py` drive the adapters built on
these against a real Postgres, and skip wholesale when there isn't one. What is checked here is
the pairing itself, on the machine of anyone running `make test`: that a synchronous caller
inside a running loop still reaches an async body, and that each adapter hands `connect` its own
tables rather than another concern's.
"""

import asyncio
from typing import Any

import pytest


def test_run_sync_reaches_an_async_body_from_inside_a_running_loop() -> None:
    """The case `asyncio.run` can't serve: a synchronous exporter called mid-run."""
    from runa._loop import run_sync

    saved: list[str] = []

    async def save(item: str) -> str:
        """Record `item`, as a `TraceStore` would."""
        saved.append(item)
        return "saved"

    async def mid_run() -> str:
        # Where `TraceExporter.export` is called from: a finishing run, so a loop is already
        # running on this thread, which is exactly what `asyncio.run` refuses to nest into.
        return run_sync(save("trace-1"))

    assert asyncio.run(mid_run()) == "saved"
    assert run_sync(save("trace-2")) == "saved"  # and from no loop at all, as `runa traces` does
    assert saved == ["trace-1", "trace-2"]


def test_a_shared_store_connects_to_its_own_url_with_its_own_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_pool` is the one place `(url, tables)` becomes a pool; twelve call sites used to be."""
    pytest.importorskip("asyncpg")
    from runa.db import pool
    from runa.db.schema import Declaration

    calls: list[tuple[str, tuple[Declaration, ...]]] = []

    async def _fake_connect(url: str, *tables: Declaration) -> str:
        calls.append((url, tables))
        return "pool"

    monkeypatch.setattr(pool, "connect", _fake_connect)
    store = pool.Shared("postgresql://example/db", "CREATE TABLE IF NOT EXISTS mine ()")

    assert asyncio.run(store._pool()) == "pool"
    assert calls == [("postgresql://example/db", ("CREATE TABLE IF NOT EXISTS mine ()",))]


def test_every_concern_hands_its_database_its_own_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each concern declares the tables it owns, five concerns in one database."""
    pytest.importorskip("asyncpg")
    from runa import db
    from runa.db.schema import POSTGRES, ddl
    from runa.db.vectors.postgres import PostgresVectorStore
    from runa.memory import _spec as spec

    url = "postgresql://example/db"
    monkeypatch.setenv(db.DATABASE_URL_ENV, url)

    def script(store: Any) -> str:
        return ddl(POSTGRES, *store.tables)

    assert "cache_entries" in script(db.cache().database)  # type: ignore[attr-defined]
    assert "eval_runs" in script(db.evals().database)  # type: ignore[attr-defined]
    assert "spans" in script(db.traces().database)  # type: ignore[attr-defined]
    assert "paused_runs" in script(db.paused_runs().database)  # type: ignore[attr-defined]
    assert "agent_messages" in script(db.sessions().database)  # type: ignore[attr-defined]
    # Both sides of the session pair reach the same two tables.
    assert db.at(url).session("s").database.tables == db.sessions().database.tables  # type: ignore[attr-defined]
    # The vector store's DDL is computed from its spec, so it can only be handed over once the
    # spec is stored: a store built before that would create a table named `_items`.
    vectors = PostgresVectorStore(spec(4), url)
    assert "memory_items" in script(vectors)
    assert "vector(4)" in script(vectors)


def test_a_shared_database_applies_its_tables_once_and_a_read_creates_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writes apply `(url, tables)` once per process; a read of absent tables answers empty."""
    pytest.importorskip("asyncpg")
    from runa.db import pool
    from runa.db.schema import Column, Table

    notes = Table("notes", columns=(Column("id", "text", primary_key=True),))
    applied: list[str] = []
    present: list[tuple[str, str]] = []

    class Conn:
        async def execute(self, sql: str) -> None:
            applied.append(sql)

        async def fetch(self, sql: str) -> list[tuple[str, str]]:
            return present

    class Acquire:
        async def __aenter__(self) -> Conn:
            return Conn()

        async def __aexit__(self, *exc: object) -> None:
            return None

    class Pool:
        def acquire(self) -> Acquire:
            return Acquire()

    async def _get_pool(url: str) -> Pool:
        return Pool()

    monkeypatch.setattr(pool, "get_pool", _get_pool)
    monkeypatch.setattr(pool, "_applied", set())
    url = "postgresql://example/db"

    async def scenario() -> None:
        assert await pool.connect_to_read(url, notes) is None
        assert applied == []  # a read created nothing
        await pool.connect(url, notes)
        await pool.connect(url, notes)
        assert len(applied) == 1  # and the second write applied nothing
        assert await pool.connect_to_read(url, notes) is not None  # remembered

    asyncio.run(scenario())
