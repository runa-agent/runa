"""The trace and eval contracts run against Postgres, plus the routing that gets a caller there.

The checks are `tests/contracts/trace.py` and `tests/contracts/eval.py`, the same ones
`tests/tracing/test_store.py` and `tests/eval/test_store.py` drive over the local backends. These
adapters exist because SQLite is per-process: without them, three replicas keep three disjoint
trace histories and a dashboard that can only ever show one. So the contract under test is that
they are indistinguishable from the others through the interface, and this backend's own
mechanisms -- the `ON CONFLICT` upsert, the batched span fetch -- are checks in those lists rather
than assertions written twice.

What is left here is what no other backend has: the `PostgresExporter`, and `RUNA_DATABASE_URL`
routing, which is the promise that moving a deployment is one environment variable.

Needs a live Postgres at `RUNA_TEST_POSTGRES_DSN` (defaults to a local one); skipped wholesale when
there isn't one, exactly like `test_postgres.py`.
"""

import asyncio
import os
import uuid
from collections.abc import Coroutine
from typing import Any

import asyncpg
import pytest
from contracts import eval as eval_contract
from contracts import trace as trace_contract

import runa.db.pool as pool_module
from runa import db
from runa.tracing.spans import Span
from runa.tracing.traces import Trace

_DSN = os.environ.get("RUNA_TEST_POSTGRES_DSN", "postgresql://runa:runa@localhost:5432/runa")


def _reachable() -> bool:
    async def _check() -> None:
        conn = await asyncpg.connect(_DSN, timeout=2)
        await conn.close()

    try:
        asyncio.new_event_loop().run_until_complete(_check())
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _reachable(), reason=f"no Postgres reachable at {_DSN}")


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run `coro` on the same background loop the synchronous backends use."""
    return pool_module.run_sync(coro)


@pytest.fixture
def unique_id() -> str:
    """A fresh tag per test, so tests sharing one live database never collide."""
    return uuid.uuid4().hex


@pytest.mark.parametrize("check", trace_contract.CONTRACT, ids=lambda check: check.__name__)
def test_trace_store_contract(unique_id: str, check: trace_contract.Check) -> None:
    """`PostgresTraceStore` answers the same `TraceStore` contract the local backends do."""
    from runa.tracing.postgres import PostgresTraceStore

    check(PostgresTraceStore(_DSN), unique_id)


@pytest.mark.parametrize("check", eval_contract.CONTRACT, ids=lambda check: check.__name__)
def test_eval_store_contract(unique_id: str, check: eval_contract.Check) -> None:
    """`PostgresEvalStore` answers the same `EvalStore` contract the local backends do.

    A shared baseline is why it exists: CI and a laptop must grade against the same run.
    """
    from runa.eval.postgres import PostgresEvalStore

    check(PostgresEvalStore(_DSN), unique_id)


def _trace(name: str) -> Trace:
    """One finished trace with one span, to hand to an exporter."""
    trace_id = uuid.uuid4().hex
    trace = Trace(id=trace_id, name=name, start_time=1000.0, end_time=1002.5)
    trace.spans.append(
        Span(
            id=f"{trace_id}-s0",
            trace_id=trace_id,
            parent_id=None,
            name="llm",
            type="llm",
            start_time=1000.0,
            end_time=1001.0,
            status="ok",
            input={"prompt": "hi"},
            output={"usage": {"total_tokens": 7}},
        )
    )
    return trace


def test_the_exporter_persists_a_finished_trace(unique_id: str) -> None:
    """`PostgresExporter` is a `StoreExporter` over this store; it must actually write."""
    from runa.tracing.postgres import PostgresExporter, PostgresTraceStore

    trace = _trace(unique_id)
    PostgresExporter(_DSN).export(trace)

    assert PostgresTraceStore(_DSN).get(trace.id) is not None


def test_the_shared_dsn_env_var_routes_traces_to_postgres(
    monkeypatch: pytest.MonkeyPatch, unique_id: str
) -> None:
    """The whole point of `RUNA_DATABASE_URL`: no code change, and reads follow writes."""
    monkeypatch.setenv(db.DATABASE_URL_ENV, _DSN)
    trace = _trace(unique_id)

    db.traces().save(trace)  # no path, no dsn: the env var decides

    assert db.traces().get(trace.id) is not None


def test_without_the_env_var_traces_stay_in_sqlite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, unique_id: str
) -> None:
    """The default is unchanged: an app that sets nothing still gets its local file."""
    monkeypatch.delenv(db.DATABASE_URL_ENV, raising=False)
    trace = _trace(unique_id)

    db.use_project(tmp_path)
    db.traces().save(trace)

    assert (tmp_path / "db" / "runa.db").exists()
    assert db.traces().get(trace.id) is not None


def test_sessions_are_readable_from_the_shared_database(
    monkeypatch: pytest.MonkeyPatch, unique_id: str
) -> None:
    """`runa sessions` and `runa ui`'s Sessions page must follow the variable too.

    They used to read `db/runa.db` with raw SQL no matter what, so a Postgres deployment showed
    an empty session list beside Traces and Evaluations pages that worked.
    """
    monkeypatch.setenv(db.DATABASE_URL_ENV, _DSN)
    session = db.session(unique_id)
    run(session.add_items([{"role": "user", "content": "hello"}]))

    store = db.sessions()

    assert unique_id in [summary.id for summary in store.listing()]
    assert [message.text for message in store.messages(unique_id)] == ["hello"]
    run(session.clear_session())


def test_evals_are_readable_from_the_shared_database(
    monkeypatch: pytest.MonkeyPatch, unique_id: str
) -> None:
    """`runa eval` and the Evaluations page follow the variable the same way."""
    monkeypatch.setenv(db.DATABASE_URL_ENV, _DSN)

    run_id = db.evals().save(eval_contract.report(unique_id))

    assert db.evals().get(run_id) is not None
