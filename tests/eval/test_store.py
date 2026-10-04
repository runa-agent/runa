"""The `EvalStore` contract, run against every adapter that needs no server.

One set of assertions, parametrized over the backends `runa.db` can resolve without a live
Postgres, so a baseline a CI job grades against behaves the same as the one on a laptop.

`tests/test_postgres_observability.py` runs the same contract against `PostgresEvalStore`.
`test_adds_trace_id_to_an_older_local_file` is the one assertion that is deliberately
SQLite-only: a shared database is created by whichever replica connects first and has always had
the column, where a local file can predate it.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from runa import db
from runa.eval.case import Case
from runa.eval.evaluation.core import EvaluationResult, Status
from runa.eval.report import CaseReport, Report
from runa.eval.store import EvalStore
from runa.eval.tracing.adapter import AgentRun
from runa.tracing import Trace


@pytest.fixture(params=["sqlite", "ephemeral"])
def store(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> EvalStore:
    """An `EvalStore`, resolved by `runa.db` the way an app's would be."""
    if request.param == "ephemeral":
        monkeypatch.setenv(db.DATABASE_URL_ENV, "memory://")
        return db.evals()
    monkeypatch.delenv(db.DATABASE_URL_ENV, raising=False)
    return db.evals(tmp_path)


def _graded(input: str, status: Status, *, index: int = 0) -> CaseReport:
    return CaseReport(
        index=index,
        case=Case(input=input),
        run=AgentRun(input=input, final_output="ok"),
        results=[EvaluationResult(metric="task_completion", status=status, reason="r")],
    )


def test_save_persists_a_run_and_its_cases(store: EvalStore) -> None:
    """`save` records the run's score and every case's input, output and verdict."""
    case_report = CaseReport(
        index=0,
        case=Case(input="hi", expected="hello"),
        run=AgentRun(input="hi", final_output="hello"),
        results=[
            EvaluationResult(metric="task_completion", status=Status.PASS, reason="ok", score=1.0)
        ],
    )

    run_id = store.save(Report(agent_name="SupportAgent", cases=[case_report]))
    run = store.get(run_id)

    assert run is not None
    assert (run.agent_name, run.score, run.pass_rate) == ("SupportAgent", 1.0, 1.0)
    assert len(run.cases) == 1
    assert run.cases[0].input == "hi"
    assert run.cases[0].output == "hello"
    assert run.cases[0].passed is True
    assert run.cases[0].results[0]["metric"] == "task_completion"


def test_save_handles_an_empty_dataset(store: EvalStore) -> None:
    """Saving a report with no cases still records the run."""
    run_id = store.save(Report(agent_name="SupportAgent", cases=[]))

    run = store.get(run_id)
    assert run is not None
    assert run.cases == []


def test_list_returns_summaries_newest_first(store: EvalStore) -> None:
    """`list` returns every run without its cases, most recent first."""
    store.save(Report(agent_name="A", cases=[]))
    store.save(Report(agent_name="B", cases=[]))

    runs = store.list()

    assert [run.agent_name for run in runs] == ["B", "A"]
    assert runs[0].cases == []


def test_list_respects_limit(store: EvalStore) -> None:
    """`limit` caps the listing at the most recent runs."""
    for name in ("A", "B", "C"):
        store.save(Report(agent_name=name, cases=[]))

    assert [run.agent_name for run in store.list(limit=2)] == ["C", "B"]


def test_get_returns_none_for_an_unknown_id(store: EvalStore) -> None:
    """`get` returns `None` when this store has no such run."""
    assert store.get(999) is None


def test_baseline_maps_the_latest_runs_inputs_to_their_verdicts(store: EvalStore) -> None:
    """Only the agent's most recent run counts, keyed by input."""
    store.save(Report("A", [_graded("hi", Status.FAIL)]))
    store.save(Report("A", [_graded("hi", Status.PASS)]))
    store.save(Report("B", [_graded("yo", Status.FAIL)]))

    assert store.baseline("A") == {"hi": True}
    assert store.baseline("never_run") is None


def test_baseline_before_a_run_is_the_run_it_was_compared_against(store: EvalStore) -> None:
    """`before=run_id` skips that run and anything newer."""
    first = store.save(Report("A", [_graded("hi", Status.PASS)]))
    second = store.save(Report("A", [_graded("hi", Status.FAIL)]))

    assert store.baseline("A", before=second) == {"hi": True}
    assert store.baseline("A", before=first) is None


def test_save_links_each_case_to_its_runs_trace(store: EvalStore) -> None:
    """A case's `trace_id` is its run's trace, so a failing case opens straight onto its spans."""
    case = _graded("hi", Status.PASS)
    case.run.trace = Trace(id="trace_9", name="A", start_time=0.0)
    untraced = _graded("yo", Status.PASS, index=1)

    run_id = store.save(Report("A", [case, untraced]))

    run = store.get(run_id)
    assert run is not None
    assert [row.trace_id for row in run.cases] == ["trace_9", None]


def test_adds_trace_id_to_an_older_local_file(tmp_path: Path) -> None:
    """A `runa.db` whose `eval_cases` predates `trace_id` gets the column on next connect."""
    db_path = tmp_path / "db" / "runa.db"
    db_path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "CREATE TABLE eval_cases (run_id INTEGER NOT NULL, case_index INTEGER NOT NULL, "
            "input TEXT NOT NULL, output TEXT, passed INTEGER NOT NULL, "
            "results_json TEXT NOT NULL, PRIMARY KEY (run_id, case_index))"
        )

    store = db.evals(tmp_path)
    store.save(Report("A", [_graded("hi", Status.PASS)]))

    assert store.baseline("A") == {"hi": True}
