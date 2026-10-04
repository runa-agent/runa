"""Tests for `PostgresTraceStore`, `PostgresEvalStore` and `PostgresSessionStore`.

Needs a live Postgres at `RUNA_TEST_POSTGRES_DSN` (defaults to a local one); skipped wholesale
when there isn't one, exactly like `test_postgres.py`, since CI provisions one as a service
container but a plain `make test` locally may not.

These exist because SQLite is per-process: without them, three replicas keep three disjoint
trace histories and a dashboard that can only ever show one. The contract under test is that the
Postgres stores are indistinguishable from the others through the interface, which is why what
each adapter shares with them -- the ordering, the marshalling, the timestamp format -- is
asserted once in `tests/tracing/test_store.py`, `tests/eval/test_store.py` and
`tests/test_session_store.py` against the backends that need no server. What is left here is
this backend's own: the upsert, the batched span fetch, and the `RUNA_DATABASE_URL` routing.
"""

import asyncio
import os
import uuid
from collections.abc import Coroutine
from typing import Any

import asyncpg
import pytest

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
def traces() -> Any:
    """The Postgres `TraceStore`, built directly so it is the one under test."""
    from runa.tracing.postgres import PostgresTraceStore

    return PostgresTraceStore(_DSN)


@pytest.fixture
def evals() -> Any:
    """The Postgres `EvalStore`, built directly so it is the one under test."""
    from runa.eval.postgres import PostgresEvalStore

    return PostgresEvalStore(_DSN)


def _trace(name: str = "Agent", *, session_id: str | None = None, error: bool = False) -> Trace:
    trace_id = uuid.uuid4().hex
    trace = Trace(id=trace_id, name=name, start_time=1000.0, end_time=1002.5, session_id=session_id)
    trace.spans.append(
        Span(
            id=f"{trace_id}-s0",
            trace_id=trace_id,
            parent_id=None,
            name="llm",
            type="llm",
            start_time=1000.0,
            end_time=1001.0,
            status="error" if error else "ok",
            error="boom" if error else None,
            input={"prompt": "hi"},
            output={"usage": {"total_tokens": 7}},
        )
    )
    return trace


def test_a_trace_round_trips_with_its_spans(traces: Any) -> None:
    """What went in comes back out, spans and structured input/output included."""
    trace = _trace()
    traces.save(trace)

    loaded = traces.get(trace.id)

    assert loaded is not None
    assert loaded.name == "Agent"
    assert len(loaded.spans) == 1
    assert loaded.spans[0].type == "llm"
    assert loaded.spans[0].input == '{"prompt": "hi"}'


def test_an_unknown_trace_id_is_none(traces: Any) -> None:
    """Same contract as every other backend: a miss is `None`, not an exception."""
    assert traces.get(uuid.uuid4().hex) is None


def test_saving_the_same_trace_twice_replaces_it(traces: Any) -> None:
    """The `ON CONFLICT` upsert: one export per run, but a re-export must not duplicate rows."""
    trace = _trace()
    traces.save(trace)
    trace.name = "Renamed"
    traces.save(trace)

    loaded = traces.get(trace.id)

    assert loaded is not None
    assert loaded.name == "Renamed"
    assert len(loaded.spans) == 1


def test_traces_can_be_filtered_by_session(traces: Any) -> None:
    """`runa ui`'s session timeline needs this filter to behave as it does locally."""
    session_id = uuid.uuid4().hex
    traces.save(_trace(session_id=session_id))
    traces.save(_trace())

    found = traces.list(session_id=session_id)

    assert len(found) == 1
    assert found[0].session_id == session_id


def test_traces_can_be_filtered_by_error_status(traces: Any) -> None:
    """`runa traces errors` is a status filter; a trace is an error when a span is."""
    name = uuid.uuid4().hex
    traces.save(_trace(name=name, error=True))

    found = traces.list(agent=name, status="error")

    assert len(found) == 1
    assert found[0].status == "error"


def test_listing_attaches_each_trace_its_own_spans(traces: Any) -> None:
    """The batched span fetch, this backend's own, must not cross-assign spans between traces."""
    name = uuid.uuid4().hex
    first, second = _trace(name=name), _trace(name=name)
    traces.save(first)
    traces.save(second)

    found = traces.list(agent=name)

    assert len(found) == 2
    for trace in found:
        assert [span.trace_id for span in trace.spans] == [trace.id]


def test_an_empty_listing_is_an_empty_list(traces: Any) -> None:
    """No traces for an agent is not an error, and must not try to fetch spans for nothing."""
    assert traces.list(agent=uuid.uuid4().hex) == []


def test_the_exporter_persists_a_finished_trace(traces: Any) -> None:
    """`PostgresExporter` is a `StoreExporter` over this store; it must actually write."""
    from runa.tracing.postgres import PostgresExporter

    trace = _trace()
    PostgresExporter(_DSN).export(trace)

    assert traces.get(trace.id) is not None


def test_the_shared_dsn_env_var_routes_traces_to_postgres(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point of `RUNA_DATABASE_URL`: no code change, and reads follow writes."""
    monkeypatch.setenv(db.DATABASE_URL_ENV, _DSN)
    trace = _trace()

    db.traces().save(trace)  # no path, no dsn: the env var decides

    assert db.traces().get(trace.id) is not None


def test_without_the_env_var_traces_stay_in_sqlite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The default is unchanged: an app that sets nothing still gets its local file."""
    monkeypatch.delenv(db.DATABASE_URL_ENV, raising=False)
    trace = _trace()

    db.traces(tmp_path).save(trace)

    assert (tmp_path / "db" / "runa.db").exists()
    assert db.traces(tmp_path).get(trace.id) is not None


def test_sessions_are_readable_from_the_shared_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """`runa sessions` and `runa ui`'s Sessions page must follow the variable too.

    They used to read `db/runa.db` with raw SQL no matter what, so a Postgres deployment showed
    an empty session list beside Traces and Evaluations pages that worked.
    """
    monkeypatch.setenv(db.DATABASE_URL_ENV, _DSN)
    session_id = uuid.uuid4().hex
    session = run(_write_session(session_id))

    store = db.sessions()
    assert session_id in [summary.id for summary in store.listing()]
    assert [message.text for message in store.messages(session_id)] == ["hello"]
    run(session.clear_session())


def test_a_shared_sessions_updated_at_is_rendered_like_a_local_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`"YYYY-MM-DD HH:MM:SS"`, no offset: the format `tests/test_session_store.py` pins.

    This side used to hand back a `TIMESTAMPTZ`'s offset-bearing ISO string, so one deployment's
    Sessions page printed a timestamp the other's never would.
    """
    monkeypatch.setenv(db.DATABASE_URL_ENV, _DSN)
    session_id = uuid.uuid4().hex
    session = run(_write_session(session_id))

    summary = next(s for s in db.sessions().listing() if s.id == session_id)

    assert len(summary.updated_at) == len("2026-10-04 10:08:03")
    assert "+" not in summary.updated_at
    run(session.clear_session())


async def _write_session(session_id: str) -> Any:
    """One session with one user message, written through the Postgres adapter."""
    from runa.session.postgres import PostgresSession

    session = PostgresSession(session_id, _DSN)
    await session.add_items([{"role": "user", "content": "hello"}])
    return session


def _report(agent_name: str, *, passed: bool = True) -> Any:
    from runa.eval.case import Case
    from runa.eval.evaluation.core import EvaluationResult, Status
    from runa.eval.report import CaseReport, Report
    from runa.eval.tracing.adapter import AgentRun

    run_row = AgentRun(input="in", final_output="out", trace=Trace(id="", name="t", start_time=0.0))
    case = CaseReport(
        index=0,
        case=Case(input="in"),
        run=run_row,
        results=[
            EvaluationResult(
                metric="m", status=Status.PASS if passed else Status.FAIL, reason="because"
            )
        ],
    )
    return Report(agent_name=agent_name, cases=[case], baseline=None)


def test_an_eval_run_round_trips_with_its_cases(evals: Any) -> None:
    """Eval history has to survive the trip, or the baseline comparison is meaningless."""
    agent = uuid.uuid4().hex
    run_id = evals.save(_report(agent))

    loaded = evals.get(run_id)

    assert loaded is not None
    assert loaded.agent_name == agent
    assert len(loaded.cases) == 1
    assert loaded.cases[0].input == "in"
    assert loaded.cases[0].passed is True


def test_the_baseline_is_the_latest_run_for_that_agent(evals: Any) -> None:
    """A shared baseline is why this exists: CI and a laptop must compare against the same run."""
    agent = uuid.uuid4().hex
    evals.save(_report(agent, passed=False))
    evals.save(_report(agent, passed=True))

    assert evals.baseline(agent) == {"in": True}


def test_the_baseline_can_look_before_a_given_run(evals: Any) -> None:
    """`before` is what a past run was compared against, for showing a regression in context."""
    agent = uuid.uuid4().hex
    evals.save(_report(agent, passed=False))
    second = evals.save(_report(agent, passed=True))

    assert evals.baseline(agent, before=second) == {"in": False}


def test_no_baseline_for_an_agent_that_has_never_run(evals: Any) -> None:
    """`None`, not an empty dict: "no baseline" and "everything failed" are different."""
    assert evals.baseline(uuid.uuid4().hex) is None


def test_eval_runs_are_listed_newest_first(evals: Any) -> None:
    """The UI's ordering contract, matching every other backend."""
    agent = uuid.uuid4().hex
    first = evals.save(_report(agent))
    second = evals.save(_report(agent))

    listed = [run.id for run in evals.list(limit=100)]

    assert listed.index(second) < listed.index(first)


def test_an_unknown_eval_run_id_is_none(evals: Any) -> None:
    """A miss is `None`, matching every other backend."""
    assert evals.get(2**40) is None
