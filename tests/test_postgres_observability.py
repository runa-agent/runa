"""Tests for `tracing/postgres.py` and `eval/postgres.py`, the shared-history backends.

Needs a live Postgres at `RUNA_TEST_POSTGRES_DSN` (defaults to `DEFAULT_POSTGRES_DSN`); skipped
wholesale when there isn't one, exactly like `test_postgres.py`, since CI provisions one as a
service container but a plain `make test` locally may not.

These exist because SQLite is per-process: without them, three replicas keep three disjoint trace
histories and a dashboard that can only ever show one. The contract under test is that the
Postgres backends are indistinguishable from the SQLite ones through the public API.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Coroutine
from typing import Any

import asyncpg
import pytest

import runa.db.postgres as postgres_module
from runa.db.postgres import DEFAULT_POSTGRES_DSN
from runa.tracing.spans import Span
from runa.tracing.traces import Trace

_DSN = os.environ.get("RUNA_TEST_POSTGRES_DSN", DEFAULT_POSTGRES_DSN)


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
    return postgres_module.run_sync(coro)


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


def test_a_trace_round_trips_with_its_spans() -> None:
    """What went in comes back out, spans and structured input/output included."""
    from runa.tracing import postgres

    trace = _trace()
    postgres.save_trace(trace, dsn=_DSN)

    loaded = postgres.get_trace(trace.id, dsn=_DSN)

    assert loaded is not None
    assert loaded.name == "Agent"
    assert len(loaded.spans) == 1
    assert loaded.spans[0].type == "llm"


def test_an_unknown_trace_id_is_none() -> None:
    """Same contract as the SQLite backend: a miss is `None`, not an exception."""
    from runa.tracing import postgres

    assert postgres.get_trace(uuid.uuid4().hex, dsn=_DSN) is None


def test_saving_the_same_trace_twice_replaces_it() -> None:
    """A trace is exported once per run, but a re-export must not duplicate rows."""
    from runa.tracing import postgres

    trace = _trace()
    postgres.save_trace(trace, dsn=_DSN)
    trace.name = "Renamed"
    postgres.save_trace(trace, dsn=_DSN)

    loaded = postgres.get_trace(trace.id, dsn=_DSN)

    assert loaded is not None
    assert loaded.name == "Renamed"
    assert len(loaded.spans) == 1


def test_traces_can_be_filtered_by_session() -> None:
    """`runa ui`'s session timeline needs this filter to behave as it does on SQLite."""
    from runa.tracing import postgres

    session_id = uuid.uuid4().hex
    postgres.save_trace(_trace(session_id=session_id), dsn=_DSN)
    postgres.save_trace(_trace(), dsn=_DSN)

    found = postgres.list_traces(session_id=session_id, dsn=_DSN)

    assert len(found) == 1
    assert found[0].session_id == session_id


def test_traces_can_be_filtered_by_error_status() -> None:
    """`runa traces errors` is a status filter; a trace is an error when a span is."""
    from runa.tracing import postgres

    name = uuid.uuid4().hex
    postgres.save_trace(_trace(name=name, error=True), dsn=_DSN)

    found = postgres.list_traces(agent=name, status="error", dsn=_DSN)

    assert len(found) == 1
    assert found[0].status == "error"


def test_listing_attaches_each_trace_its_own_spans() -> None:
    """The batched span fetch must not cross-assign spans between traces."""
    from runa.tracing import postgres

    name = uuid.uuid4().hex
    first, second = _trace(name=name), _trace(name=name)
    postgres.save_trace(first, dsn=_DSN)
    postgres.save_trace(second, dsn=_DSN)

    found = postgres.list_traces(agent=name, dsn=_DSN)

    assert len(found) == 2
    for trace in found:
        assert [span.trace_id for span in trace.spans] == [trace.id]


def test_an_empty_listing_is_an_empty_list() -> None:
    """No traces for an agent is not an error, and must not try to fetch spans for nothing."""
    from runa.tracing import postgres

    assert postgres.list_traces(agent=uuid.uuid4().hex, dsn=_DSN) == []


def test_the_exporter_persists_a_finished_trace() -> None:
    """`PostgresExporter` is what `_default_exporters` installs; it must actually write."""
    from runa.tracing.postgres import PostgresExporter, get_trace

    trace = _trace()
    PostgresExporter(_DSN).export(trace)

    assert get_trace(trace.id, dsn=_DSN) is not None


def test_the_shared_dsn_env_var_routes_traces_to_postgres(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point of `RUNA_POSTGRES_DSN`: no code change, and reads follow writes."""
    from runa.tracing import storage

    monkeypatch.setenv("RUNA_POSTGRES_DSN", _DSN)
    trace = _trace()

    storage.save_trace(trace)  # no db_path, no dsn: the env var decides

    assert storage.get_trace(trace.id) is not None


def test_without_the_env_var_traces_stay_in_sqlite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The default is unchanged: an app that sets nothing still gets its local file."""
    from runa.tracing import storage

    monkeypatch.delenv("RUNA_POSTGRES_DSN", raising=False)
    trace = _trace()
    db = tmp_path / "runa.db"

    storage.save_trace(trace, db_path=db)

    assert db.exists()
    assert storage.get_trace(trace.id, db_path=db) is not None


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


def test_an_eval_run_round_trips_with_its_cases() -> None:
    """Eval history has to survive the trip, or the baseline comparison is meaningless."""
    from runa.eval import postgres

    agent = uuid.uuid4().hex
    run_id = postgres.save_report(_report(agent), dsn=_DSN)

    loaded = postgres.get_eval_run(run_id, dsn=_DSN)

    assert loaded is not None
    assert loaded.agent_name == agent
    assert len(loaded.cases) == 1
    assert loaded.cases[0].input == "in"


def test_the_baseline_is_the_latest_run_for_that_agent() -> None:
    """A shared baseline is why this exists: CI and a laptop must compare against the same run."""
    from runa.eval import postgres

    agent = uuid.uuid4().hex
    postgres.save_report(_report(agent, passed=False), dsn=_DSN)
    postgres.save_report(_report(agent, passed=True), dsn=_DSN)

    baseline = postgres.load_baseline(agent, dsn=_DSN)

    assert baseline == {"in": True}


def test_the_baseline_can_look_before_a_given_run() -> None:
    """`before` is what a past run was compared against, for showing a regression in context."""
    from runa.eval import postgres

    agent = uuid.uuid4().hex
    postgres.save_report(_report(agent, passed=False), dsn=_DSN)
    second = postgres.save_report(_report(agent, passed=True), dsn=_DSN)

    assert postgres.load_baseline(agent, before=second, dsn=_DSN) == {"in": False}


def test_no_baseline_for_an_agent_that_has_never_run() -> None:
    """`None`, not an empty dict: "no baseline" and "everything failed" are different."""
    from runa.eval import postgres

    assert postgres.load_baseline(uuid.uuid4().hex, dsn=_DSN) is None


def test_eval_runs_are_listed_newest_first() -> None:
    """The UI's ordering contract, matching the SQLite backend."""
    from runa.eval import postgres

    agent = uuid.uuid4().hex
    first = postgres.save_report(_report(agent), dsn=_DSN)
    second = postgres.save_report(_report(agent), dsn=_DSN)

    listed = [run.id for run in postgres.list_eval_runs(limit=100, dsn=_DSN)]

    assert listed.index(second) < listed.index(first)


def test_an_unknown_eval_run_id_is_none() -> None:
    """A miss is `None`, matching `eval/storage.py`."""
    from runa.eval import postgres

    assert postgres.get_eval_run(2**40, dsn=_DSN) is None


def test_pruning_a_shared_postgres_removes_old_traces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retention has to work on the shared store too; that is where growth actually hurts."""
    from runa.db.prune import prune
    from runa.tracing import postgres

    monkeypatch.setenv("RUNA_POSTGRES_DSN", _DSN)
    old = _trace()
    old.start_time = 0.0  # 1970, comfortably past any cutoff
    old.spans[0].start_time = 0.0
    postgres.save_trace(old, dsn=_DSN)

    pruned = prune(older_than_days=30)

    assert pruned.traces >= 1
    assert pruned.spans >= 1
    assert postgres.get_trace(old.id, dsn=_DSN) is None


def test_pruning_a_shared_postgres_keeps_recent_traces(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cutoff means the same thing on both backends."""
    import time

    from runa.db.prune import prune
    from runa.tracing import postgres

    monkeypatch.setenv("RUNA_POSTGRES_DSN", _DSN)
    recent = _trace()
    recent.start_time = time.time()
    postgres.save_trace(recent, dsn=_DSN)

    prune(older_than_days=30)

    assert postgres.get_trace(recent.id, dsn=_DSN) is not None


def test_a_postgres_dry_run_deletes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--dry-run` is the same promise on Postgres: count, do not touch."""
    from runa.db.prune import prune
    from runa.tracing import postgres

    monkeypatch.setenv("RUNA_POSTGRES_DSN", _DSN)
    old = _trace()
    old.start_time = 0.0
    postgres.save_trace(old, dsn=_DSN)

    pruned = prune(older_than_days=30, dry_run=True)

    assert pruned.traces >= 1
    assert postgres.get_trace(old.id, dsn=_DSN) is not None
