"""tests/cache_contract.py: the one `Cache` contract, so every backend is held to it.

`runa.cache` is a Protocol with four backends behind it, and `runa.db.cache()` hands back whichever
one `RUNA_DATABASE_URL` resolved. A deployment that moves to Postgres is relying on the new
backend behaving like the old one, so what is worth testing is the contract rather than each
adapter's SQL: a miss returns `None`, a second `set` overwrites, an expired entry reads as a miss.

Each check takes the cache and one key prefix it is free to write under. The prefix is what lets
the same checks run against a live shared Postgres, where every test in a session shares one
`cache_entries` table and a fixed `"key"` would collide. `test_cache.py` drives these over the
local backends on every `make test`; `test_postgres.py` drives them over `PostgresCache` when a
live database is reachable.
"""

from collections.abc import Callable, Coroutine
from typing import Any

from runa.cache import Cache

Check = Callable[[Cache, str], Coroutine[Any, Any, None]]


async def check_get_on_missing_key_returns_none(cache: Cache, key: str) -> None:
    """A key that was never set is a miss, not an error."""
    assert await cache.get(key) is None


async def check_set_then_get_round_trips_the_value(cache: Cache, key: str) -> None:
    """A value set without a `ttl` comes back unchanged and never expires."""
    await cache.set(key, {"a": 1, "b": [1, 2, 3]})

    assert await cache.get(key) == {"a": 1, "b": [1, 2, 3]}


async def check_set_overwrites_an_existing_value(cache: Cache, key: str) -> None:
    """Setting an already-used key replaces its value rather than erroring or appending."""
    await cache.set(key, "first")
    await cache.set(key, "second")

    assert await cache.get(key) == "second"


async def check_set_overwrites_an_expired_entrys_ttl(cache: Cache, key: str) -> None:
    """A second `set` replaces the old entry's expiry too, reviving an already-expired key."""
    await cache.set(key, "stale", ttl=-1)
    await cache.set(key, "fresh", ttl=3600)

    assert await cache.get(key) == "fresh"


async def check_delete_removes_the_key(cache: Cache, key: str) -> None:
    """A deleted key is a miss afterward."""
    await cache.set(key, "value")
    await cache.delete(key)

    assert await cache.get(key) is None


async def check_delete_on_missing_key_does_not_raise(cache: Cache, key: str) -> None:
    """Deleting a key that was never set is a no-op, not an error."""
    await cache.delete(key)


async def check_clear_removes_every_entry(cache: Cache, key: str) -> None:
    """`clear` empties the whole cache, not just one key."""
    await cache.set(f"{key}-a", 1)
    await cache.set(f"{key}-b", 2)
    await cache.clear()

    assert await cache.get(f"{key}-a") is None
    assert await cache.get(f"{key}-b") is None


async def check_expired_entry_behaves_as_a_miss(cache: Cache, key: str) -> None:
    """A key set with a `ttl` in the past is already expired the moment it's read."""
    await cache.set(key, "value", ttl=-1)

    assert await cache.get(key) is None


async def check_expired_entry_stays_a_miss_when_read_twice(cache: Cache, key: str) -> None:
    """Expiry is lazy in the SQL backends, so the read that evicts must not resurrect the key."""
    await cache.set(key, "value", ttl=-1)
    await cache.get(key)

    assert await cache.get(key) is None


async def check_unexpired_entry_is_still_a_hit(cache: Cache, key: str) -> None:
    """A key set with a `ttl` far in the future is still readable."""
    await cache.set(key, "value", ttl=3600)

    assert await cache.get(key) == "value"


CONTRACT: list[Check] = [
    check_get_on_missing_key_returns_none,
    check_set_then_get_round_trips_the_value,
    check_set_overwrites_an_existing_value,
    check_set_overwrites_an_expired_entrys_ttl,
    check_delete_removes_the_key,
    check_delete_on_missing_key_does_not_raise,
    check_clear_removes_every_entry,
    check_expired_entry_behaves_as_a_miss,
    check_expired_entry_stays_a_miss_when_read_twice,
    check_unexpired_entry_is_still_a_hit,
]
