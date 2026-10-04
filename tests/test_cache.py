"""Tests for the local `Cache` backends: `MemoryCache`, `SQLiteCache`, and `memory://`'s shared one.

The contract itself lives in `cache_contract.py` and is driven here over every backend a plain
`make test` can reach. `PostgresCache` is held to the same checks in `test_postgres.py`, which
needs a live database.

What stays here is each backend's own promise, the part the contract deliberately says nothing
about: whether a value outlives the object that wrote it.
"""

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest
from cache_contract import CONTRACT, Check

from runa import db
from runa.cache import Cache, MemoryCache, SQLiteCache

_BACKENDS: dict[str, Callable[[Path], Cache]] = {
    "memory": lambda tmp_path: MemoryCache(),
    "sqlite": lambda tmp_path: SQLiteCache(tmp_path / "runa.db"),
    "shared": lambda tmp_path: db.cache(),
}


@pytest.mark.parametrize("check", CONTRACT, ids=lambda check: check.__name__)
@pytest.mark.parametrize("backend", list(_BACKENDS), ids=list(_BACKENDS))
def test_cache_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, check: Check
) -> None:
    """Every local backend answers the `Cache` contract the same way."""
    monkeypatch.setenv(db.DATABASE_URL_ENV, "memory://")
    cache = _BACKENDS[backend](tmp_path)

    asyncio.run(check(cache, "key"))


def test_sqlite_cache_persists_across_instances(tmp_path: Path) -> None:
    """A value survives past the `SQLiteCache` instance that wrote it, unlike `MemoryCache`."""
    db_path = tmp_path / "runa.db"

    async def _run():
        await SQLiteCache(db_path).set("key", "value")
        return await SQLiteCache(db_path).get("key")

    assert asyncio.run(_run()) == "value"


def test_memory_cache_does_not_persist_across_instances(tmp_path: Path) -> None:
    """A fresh `MemoryCache` starts empty, even for a key another instance set."""

    async def _run():
        await MemoryCache().set("key", "value")
        return await MemoryCache().get("key")

    assert asyncio.run(_run()) is None


def test_shared_memory_cache_is_one_cache_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under `memory://`, two `db.cache()` calls are the same cache, or every write is a no-op."""
    monkeypatch.setenv(db.DATABASE_URL_ENV, "memory://")

    async def _run():
        await db.cache().set("key", "value")
        return await db.cache().get("key")

    assert asyncio.run(_run()) == "value"
