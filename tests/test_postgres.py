"""Tests for the Postgres adapters: session, memory and knowledge stores.

Needs a live Postgres with the `pgvector` extension reachable at `RUNA_TEST_POSTGRES_DSN`
(defaults to a local one); the whole module is skipped if it isn't reachable,
since CI provisions one as a service container (see `.github/workflows/ci.yml`) but a plain
`make test` locally may not have one running.

Every test uses a fresh `uuid4` session/user id so tests can share the live database without
cleaning up after each other or colliding on identifiers.

All tests run on one shared event loop instead of a fresh `asyncio.run()` each: `db/pool.py`
caches a connection pool per `(loop, url)` (see its `get_pool` docstring), so a fresh loop per
test would leak that test's whole pool -- `asyncio.run()` tears the loop down without ever
`close()`-ing what was cached against it, and a Postgres instance only accepts so many
connections before `TooManyConnectionsError`.
"""

import asyncio
import os
import uuid
from collections.abc import Coroutine
from typing import Any

import asyncpg
import pytest

import runa.db.pool as pool_module
from runa.knowledge.postgres import PostgresKnowledgeStore
from runa.memory.postgres import PostgresMemoryStore
from runa.session.postgres import PostgresSession

_DSN = os.environ.get("RUNA_TEST_POSTGRES_DSN", "postgresql://runa:runa@localhost:5432/runa")
_DIMENSIONS = 4

_loop = asyncio.new_event_loop()


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run `coro` on this module's one shared loop, so every test reuses the same pool."""
    return _loop.run_until_complete(coro)


def _reachable() -> bool:
    async def _check() -> None:
        conn = await asyncpg.connect(_DSN, timeout=2)
        await conn.close()

    try:
        run(_check())
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _reachable(), reason=f"no Postgres reachable at {_DSN}")


@pytest.fixture(scope="module", autouse=True)
def _close_pool_after_module() -> Any:
    """Close this module's pool and loop once every test has run, instead of leaking them."""
    yield
    pool = pool_module._pools.pop((id(_loop), _DSN), None)
    if pool is not None:
        run(pool.close())
    _loop.close()


@pytest.fixture
def unique_id() -> str:
    """A fresh id per test, so tests sharing one live database never collide."""
    return uuid.uuid4().hex


def test_add_items_then_get_items_round_trips_in_order(unique_id: str) -> None:
    """Items come back oldest-first, matching the order they were added in."""
    session = PostgresSession(unique_id, _DSN)

    async def _run():
        await session.add_items([{"role": "user", "content": "hi"}])
        await session.add_items([{"role": "assistant", "content": "hello"}])
        return await session.get_items()

    items = run(_run())
    assert items == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_get_items_with_limit_returns_the_latest_n_in_order(unique_id: str) -> None:
    """`limit` returns the most recent items, still oldest-first."""
    session = PostgresSession(unique_id, _DSN)

    async def _run():
        for i in range(3):
            await session.add_items([{"role": "user", "content": str(i)}])
        return await session.get_items(limit=2)

    items = run(_run())
    assert [item["content"] for item in items] == ["1", "2"]


def test_pop_item_removes_and_returns_the_most_recent_item(unique_id: str) -> None:
    """`pop_item` removes the last item added and returns it."""
    session = PostgresSession(unique_id, _DSN)

    async def _run():
        await session.add_items([{"role": "user", "content": "first"}])
        await session.add_items([{"role": "user", "content": "second"}])
        popped = await session.pop_item()
        remaining = await session.get_items()
        return popped, remaining

    popped, remaining = run(_run())
    assert popped == {"role": "user", "content": "second"}
    assert remaining == [{"role": "user", "content": "first"}]


def test_pop_item_on_empty_session_returns_none(unique_id: str) -> None:
    """Popping from a session with no history returns `None`, not an error."""
    session = PostgresSession(unique_id, _DSN)

    assert run(session.pop_item()) is None


def test_set_items_replaces_the_entire_history(unique_id: str) -> None:
    """`set_items` drops whatever history existed and stores exactly the given items."""
    session = PostgresSession(unique_id, _DSN)

    async def _run():
        await session.add_items([{"role": "user", "content": "old"}])
        await session.set_items([{"role": "user", "content": "new"}])
        return await session.get_items()

    assert run(_run()) == [{"role": "user", "content": "new"}]


def test_clear_session_removes_everything(unique_id: str) -> None:
    """A cleared session has no items left, and can be written to again afterward."""
    session = PostgresSession(unique_id, _DSN)

    async def _run():
        await session.add_items([{"role": "user", "content": "hi"}])
        await session.clear_session()
        after_clear = await session.get_items()
        await session.add_items([{"role": "user", "content": "again"}])
        return after_clear, await session.get_items()

    after_clear, after_rewrite = run(_run())
    assert after_clear == []
    assert after_rewrite == [{"role": "user", "content": "again"}]


def test_memory_store_add_then_search_returns_the_closest_match_first(unique_id: str) -> None:
    """A search embedding closest to one stored item ranks it first."""
    store = PostgresMemoryStore(_DSN, dimensions=_DIMENSIONS)

    async def _run():
        await store.add(
            user_id=unique_id, text="cats", embedding=[1.0, 0.0, 0.0, 0.0], metadata=None
        )
        await store.add(
            user_id=unique_id, text="finance", embedding=[0.0, 0.0, 0.0, 1.0], metadata=None
        )
        return await store.search(user_id=unique_id, embedding=[1.0, 0.0, 0.0, 0.0], k=2)

    matches = run(_run())
    assert [m.text for m in matches] == ["cats", "finance"]
    assert matches[0].distance <= matches[1].distance


def test_memory_store_metadata_round_trips(unique_id: str) -> None:
    """Metadata passed to `add` comes back through `search` as the same dict."""
    store = PostgresMemoryStore(_DSN, dimensions=_DIMENSIONS)

    async def _run():
        await store.add(
            user_id=unique_id,
            text="cats",
            embedding=[1.0, 0.0, 0.0, 0.0],
            metadata={"source": "notes"},
        )
        return await store.search(user_id=unique_id, embedding=[1.0, 0.0, 0.0, 0.0], k=1)

    matches = run(_run())
    assert matches[0].metadata == {"source": "notes"}


def test_memory_store_isolates_by_user_id(unique_id: str) -> None:
    """A search scoped to one `user_id` never returns another user's item."""
    store = PostgresMemoryStore(_DSN, dimensions=_DIMENSIONS)
    other_id = uuid.uuid4().hex

    async def _run():
        await store.add(
            user_id=unique_id, text="mine", embedding=[1.0, 0.0, 0.0, 0.0], metadata=None
        )
        return await store.search(user_id=other_id, embedding=[1.0, 0.0, 0.0, 0.0], k=5)

    assert run(_run()) == []


def test_memory_store_none_user_id_is_its_own_scope(unique_id: str) -> None:
    """Items with `user_id=None` don't leak into a search scoped to a real user, or vice versa."""
    store = PostgresMemoryStore(_DSN, dimensions=_DIMENSIONS)

    async def _run():
        item_id = await store.add(
            user_id=None, text=unique_id, embedding=[1.0, 0.0, 0.0, 0.0], metadata=None
        )
        try:
            scoped = await store.search(user_id=unique_id, embedding=[1.0, 0.0, 0.0, 0.0], k=5)
            unscoped = await store.search(user_id=None, embedding=[1.0, 0.0, 0.0, 0.0], k=5)
        finally:
            await store.delete(user_id=None, memory_id=item_id)
        return scoped, unscoped

    scoped, unscoped = run(_run())
    assert scoped == []
    assert unique_id in [m.text for m in unscoped]


def test_memory_store_delete_removes_the_item(unique_id: str) -> None:
    """A deleted item no longer comes back from `search`."""
    store = PostgresMemoryStore(_DSN, dimensions=_DIMENSIONS)

    async def _run():
        item_id = await store.add(
            user_id=unique_id, text="cats", embedding=[1.0, 0.0, 0.0, 0.0], metadata=None
        )
        await store.delete(user_id=unique_id, memory_id=item_id)
        return await store.search(user_id=unique_id, embedding=[1.0, 0.0, 0.0, 0.0], k=5)

    assert run(_run()) == []


def test_knowledge_store_add_then_search_returns_the_closest_chunk_first() -> None:
    """A search embedding closest to one stored chunk ranks it first."""
    store = PostgresKnowledgeStore(_DSN, dimensions=_DIMENSIONS)

    async def _run():
        await store.clear()
        await store.add(text="chunk a", source="a.md", embedding=[1.0, 0.0, 0.0, 0.0])
        await store.add(text="chunk b", source="b.md", embedding=[0.0, 0.0, 0.0, 1.0])
        matches = await store.search(embedding=[1.0, 0.0, 0.0, 0.0], k=2)
        await store.clear()
        return matches

    matches = run(_run())
    assert [m.text for m in matches] == ["chunk a", "chunk b"]
    assert matches[0].source == "a.md"


def test_knowledge_store_clear_removes_every_chunk() -> None:
    """`clear` empties the whole knowledge base, not just the chunks from one source."""
    store = PostgresKnowledgeStore(_DSN, dimensions=_DIMENSIONS)

    async def _run():
        await store.add(text="chunk a", source="a.md", embedding=[1.0, 0.0, 0.0, 0.0])
        await store.clear()
        return await store.search(embedding=[1.0, 0.0, 0.0, 0.0], k=5)

    assert run(_run()) == []
