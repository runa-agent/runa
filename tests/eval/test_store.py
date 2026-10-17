"""The `EvalStore` contract, run against every adapter that needs no server.

The checks live in `tests/contracts/eval.py` and are driven here over the backends `runa.db` can
resolve without a live Postgres, so a baseline a CI job grades against behaves the same as the one
on a laptop. `tests/test_postgres_observability.py` runs them against `PostgresEvalStore`.

`test_adds_trace_id_to_an_older_local_file` stays here, outside the contract: a shared database is
created by whichever replica connects first and has always had the column, where a local file can
predate it. `baseline_clause` is asserted per dialect for the same reason `listing_clause` is in
`tests/test_session_store.py`: both SQL adapters bind what it returns, and only one of them can be
run without a server. Its Python twin, `baseline_row`, needs no test of its own here -- the
contract runs over the ephemeral store, which is the adapter that calls it.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from contracts.eval import CONTRACT, Check
from helpers import finished_run

from runa import db
from runa.db.schema import POSTGRES, SQLITE, Dialect
from runa.eval.case import Case
from runa.eval.evaluation.core import EvalStatus, EvaluationResult
from runa.eval.report import CaseReport, Report
from runa.eval.store import EvalStore, baseline_clause
from runa.project import use


@pytest.fixture(params=["sqlite", "ephemeral"])
def store(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> EvalStore:
    """An `EvalStore`, resolved by `runa.db` the way an app's would be."""
    if request.param == "ephemeral":
        monkeypatch.setenv(db.DATABASE_URL_ENV, "memory://")
        return db.evals()
    monkeypatch.delenv(db.DATABASE_URL_ENV, raising=False)
    use(tmp_path)
    return db.evals()


@pytest.mark.parametrize("check", CONTRACT, ids=lambda check: check.__name__)
def test_eval_store_contract(store: EvalStore, check: Check) -> None:
    """Every local backend answers the `EvalStore` contract the same way."""
    check(store, "SupportAgent")


@pytest.mark.parametrize("dialect", [SQLITE, POSTGRES], ids=lambda dialect: dialect.name)
def test_baseline_clause_is_the_same_rule_in_either_dialect(dialect: Dialect) -> None:
    """One ordering, one cap, and `before` bound exclusively, whichever backend asks.

    The two adapters used to phrase this separately -- a `2**63 - 1` sentinel locally against two
    branches on Postgres -- so the only statement that they mean the same thing was prose.
    """
    newest, params = baseline_clause("SupportAgent", None, dialect)
    older, older_params = baseline_clause("SupportAgent", 7, dialect)

    assert newest == f"WHERE agent_name = {dialect.placeholder(1)} ORDER BY id DESC LIMIT 1"
    assert params == ("SupportAgent",)
    assert older == (
        f"WHERE agent_name = {dialect.placeholder(1)} AND id < {dialect.placeholder(2)} "
        "ORDER BY id DESC LIMIT 1"
    )
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
