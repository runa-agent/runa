"""Tests for `runa.cache.redis.RedisCache`.

Needs a live Redis reachable at `RUNA_TEST_REDIS_URL` (defaults to a local one);
the whole module is skipped if it isn't reachable, since CI provisions one as a service
container (see `.github/workflows/ci.yml`) but a plain `make test` locally may not have one
running.

Every test uses a fresh `uuid4` key so tests can share the live database without colliding on
keys. `test_clear_removes_every_entry` is the exception -- it wipes the whole database -- so it's
written to make no assumption about what else is in it, and any test relying on state from an
earlier test would break regardless of run order, not just because of this one.
"""

import asyncio
import os
import uuid

import pytest
import redis.asyncio as redis_asyncio

from runa.cache.redis import RedisCache

_URL = os.environ.get("RUNA_TEST_REDIS_URL", "redis://localhost:6379/0")


def _reachable() -> bool:
    async def _check() -> None:
        client = redis_asyncio.from_url(_URL)
        try:
            await client.ping()
        finally:
            await client.aclose()

    try:
        asyncio.run(_check())
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _reachable(), reason=f"no Redis reachable at {_URL}")


@pytest.fixture
def key() -> str:
    """A fresh key per test, so tests sharing one live database never collide."""
    return uuid.uuid4().hex


def test_get_on_missing_key_returns_none(key: str) -> None:
    """A key that was never set is a miss, not an error."""
    cache = RedisCache(_URL)

    assert asyncio.run(cache.get(key)) is None


def test_set_then_get_round_trips_the_value(key: str) -> None:
    """A value set without a `ttl` comes back unchanged."""
    cache = RedisCache(_URL)

    async def _run():
        await cache.set(key, {"a": 1, "b": [1, 2, 3]})
        return await cache.get(key)

    assert asyncio.run(_run()) == {"a": 1, "b": [1, 2, 3]}


def test_set_overwrites_an_existing_value(key: str) -> None:
    """Setting an already-used key replaces its value rather than erroring or appending."""
    cache = RedisCache(_URL)

    async def _run():
        await cache.set(key, "first")
        await cache.set(key, "second")
        return await cache.get(key)

    assert asyncio.run(_run()) == "second"


def test_delete_removes_the_key(key: str) -> None:
    """A deleted key is a miss afterward."""
    cache = RedisCache(_URL)

    async def _run():
        await cache.set(key, "value")
        await cache.delete(key)
        return await cache.get(key)

    assert asyncio.run(_run()) is None


def test_delete_on_missing_key_does_not_raise(key: str) -> None:
    """Deleting a key that was never set is a no-op, not an error."""
    cache = RedisCache(_URL)

    asyncio.run(cache.delete(key))


def test_expired_entry_behaves_as_a_miss(key: str) -> None:
    """A short-lived entry is gone once its `ttl` (in seconds) elapses."""
    cache = RedisCache(_URL)

    async def _run():
        await cache.set(key, "value", ttl=0.05)
        await asyncio.sleep(0.2)
        return await cache.get(key)

    assert asyncio.run(_run()) is None


def test_unexpired_entry_is_still_a_hit(key: str) -> None:
    """A key set with a `ttl` far in the future is still readable."""
    cache = RedisCache(_URL)

    async def _run():
        await cache.set(key, "value", ttl=3600)
        return await cache.get(key)

    assert asyncio.run(_run()) == "value"


def test_clear_removes_every_entry(key: str) -> None:
    """`clear` empties the whole cache, not just one key."""
    cache = RedisCache(_URL)

    async def _run():
        await cache.set(key, "value")
        await cache.clear()
        return await cache.get(key)

    assert asyncio.run(_run()) is None
