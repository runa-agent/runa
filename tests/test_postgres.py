"""The session, vector, cache and paused-run contracts, run against the Postgres adapters.

The checks are `tests/contracts/`, the same ones `tests/test_session_store.py`,
`tests/test_vectors.py`, `tests/test_cache.py` and `tests/test_paused_runs.py` drive over the
local backends. That is the whole point of running them here: a deployment that sets
`RUNA_DATABASE_URL=postgresql://...` is relying on these adapters behaving like the file it had
before, and the lazy expiry, the upserts and the scoping only ever execute on this side.

Needs a live Postgres with the `pgvector` extension reachable at `RUNA_TEST_POSTGRES_DSN`
(defaults to a local one); the whole module is skipped if it isn't reachable, since CI provisions
one as a service container (see `.github/workflows/ci.yml`) but a plain `make test` locally may
not have one running.

Every check gets a fresh `uuid4` tag, which is what lets them share one live database with each
other and with past runs without cleaning up or colliding on identifiers.

All checks run on one shared event loop instead of a fresh `asyncio.run()` each: `db/pool.py`
caches a connection pool per loop (see `runa._loop.LoopCache`), so a fresh loop per test would
leave that test's whole pool behind until it was collected, and a Postgres instance only accepts
so many connections before `TooManyConnectionsError`.
"""

import asyncio
import os
import uuid
from collections.abc import Coroutine
from typing import Any

import asyncpg
import pytest
from contracts import cache as cache_contract
from contracts import paused_run as paused_run_contract
from contracts import session as session_contract
from contracts import vector as vector_contract

import runa.db.pool as pool_module
from runa.cache import ENTRIES
from runa.cache.database import DatabaseCache
from runa.db.pool import PostgresDatabase
from runa.db.vectors import VectorSpec
from runa.db.vectors.postgres import PostgresVectorStore
from runa.run_state.database import DatabasePausedRunStore
from runa.run_state.store import PAUSED_RUNS
from runa.session.database import DatabaseSessionStore
from runa.session.postgres import PostgresSession
from runa.session.store import MESSAGES, SESSIONS

_DSN = os.environ.get("RUNA_TEST_POSTGRES_DSN", "postgresql://runa:runa@localhost:5432/runa")

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
    run(pool_module.close_pool(_DSN))
    _loop.close()


@pytest.fixture
def unique_id() -> str:
    """A fresh tag per test, so tests sharing one live database never collide."""
    return uuid.uuid4().hex


@pytest.mark.parametrize("check", session_contract.CONTRACT, ids=lambda check: check.__name__)
def test_session_contract(unique_id: str, check: session_contract.Check) -> None:
    """The Postgres session pair answers the same contract the local backends do.

    Both sides, built directly rather than through `runa.db`, so what is under test is the adapter
    rather than the resolution: `test_postgres_observability.py` covers the routing itself.
    """
    pair = session_contract.SessionPair(
        write=lambda session_id, agent=None: PostgresSession(session_id, _DSN, agent),
        read=DatabaseSessionStore(PostgresDatabase(_DSN, SESSIONS, MESSAGES)),
        tag=unique_id,
    )

    run(check(pair))


@pytest.mark.parametrize("check", vector_contract.CONTRACT, ids=lambda check: check.__name__)
def test_vector_store_contract(unique_id: str, check: vector_contract.Check) -> None:
    """`PostgresVectorStore` answers the same `VectorStore` contract the local backends do.

    Its `pgvector` DDL, its `<->` ordering and its `IS NOT DISTINCT FROM` partition filter are
    only ever exercised here, so a divergence from the two local adapters would otherwise surface
    first in a deployment that had set `RUNA_DATABASE_URL`.
    """

    def build(spec: VectorSpec) -> PostgresVectorStore:
        return PostgresVectorStore(spec, _DSN)

    run(check(build, unique_id))


@pytest.mark.parametrize("check", cache_contract.CONTRACT, ids=lambda check: check.__name__)
def test_cache_contract(unique_id: str, check: cache_contract.Check) -> None:
    """A cache over Postgres answers the same `Cache` contract the local backends do.

    Its lazy expiry, its `ON CONFLICT` upsert and its `clear` are only ever exercised here, so a
    plain `make test` without a live Postgres does not cover them and CI does.
    """
    run(check(DatabaseCache(PostgresDatabase(_DSN, ENTRIES)), unique_id))


@pytest.mark.parametrize("check", paused_run_contract.CONTRACT, ids=lambda check: check.__name__)
def test_paused_run_store_contract(unique_id: str, check: paused_run_contract.Check) -> None:
    """A paused-run store over Postgres answers the same contract the local backend does.

    Its concurrent claims race across real pooled connections only here.
    """
    run(check(DatabasePausedRunStore(PostgresDatabase(_DSN, PAUSED_RUNS)), unique_id))
