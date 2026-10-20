"""The `EvalStore` contract, run against every backend that needs no server.

The checks live in `tests/contracts/eval.py` and are driven here over the backends `runa.db` can
resolve without a live Postgres, so a baseline a CI job grades against behaves the same as the one
on a laptop. `tests/test_postgres_observability.py` runs them over a shared Postgres.

`test_adds_trace_id_to_an_older_local_file` stays here, outside the contract: a shared database is
created by whichever replica connects first and has always had the column, where a local file can
predate it.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from contracts.eval import CONTRACT, Check
from helpers import finished_run

from runa import db
from runa.eval.case import Case
from runa.eval.evaluation.core import EvalStatus, EvaluationResult
from runa.eval.report import CaseReport, Report
from runa.eval.store import EvalStore, baseline_clause
from runa.project import use


@pytest.fixture(params=["sqlite", "memory"])
def store(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> EvalStore:
    """An `EvalStore`, resolved by `runa.db` the way an app's would be."""
    if request.param == "memory":
        monkeypatch.setenv(db.DATABASE_URL_ENV, "memory://")
        return db.evals()
    monkeypatch.delenv(db.DATABASE_URL_ENV, raising=False)
    use(tmp_path)
    return db.evals()


@pytest.mark.parametrize("check", CONTRACT, ids=lambda check: check.__name__)
def test_eval_store_contract(store: EvalStore, check: Check) -> None:
    """Every local backend answers the `EvalStore` contract the same way."""
    check(store, "SupportAgent")


def test_baseline_clause_orders_caps_and_binds_before_exclusively() -> None:
    """One ordering, one cap, and `before` bound exclusively."""
    newest, params = baseline_clause("SupportAgent", None)
    older, older_params = baseline_clause("SupportAgent", 7)

    assert newest == "WHERE agent_name = $1 ORDER BY id DESC LIMIT 1"
    assert params == ("SupportAgent",)
    assert older == "WHERE agent_name = $1 AND id < $2 ORDER BY id DESC LIMIT 1"
    assert older_params == ("SupportAgent", 7)


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
    graded = CaseReport(
        index=0,
        case=Case(input="hi"),
        run=finished_run("ok"),
        results=[EvaluationResult(metric="task_completion", status=EvalStatus.PASS, reason="r")],
    )

    use(tmp_path)
    store = db.evals()
    store.save(Report("A", [graded]))

    assert store.baseline("A") == {"hi": True}
