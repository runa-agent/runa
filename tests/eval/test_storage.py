"""Tests for `runa.eval.storage`: `save_report` and reading runs back."""

import json
import sqlite3
from pathlib import Path

from runa.eval.case import Case
from runa.eval.evaluation.core import EvaluationResult, Status
from runa.eval.report import CaseReport, Report
from runa.eval.storage import get_eval_run, list_eval_runs, load_baseline, save_report
from runa.eval.tracing.adapter import AgentRun


def test_save_report_persists_a_run_and_its_cases(tmp_path: Path) -> None:
    """`save_report` writes one `eval_runs` row and one `eval_cases` row per case."""
    db_path = tmp_path / "runa.db"
    case_report = CaseReport(
        index=0,
        case=Case(input="hi", expected="hello"),
        run=AgentRun(input="hi", final_output="hello"),
        results=[
            EvaluationResult(metric="task_completion", status=Status.PASS, reason="ok", score=1.0)
        ],
    )
    report = Report(agent_name="SupportAgent", cases=[case_report])

    run_id = save_report(report, db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        run_row = conn.execute(
            "SELECT agent_name, score, pass_rate FROM eval_runs WHERE id = ?", (run_id,)
        ).fetchone()
        case_row = conn.execute(
            "SELECT input, output, passed, results_json FROM eval_cases WHERE run_id = ?",
            (run_id,),
        ).fetchone()

    assert run_row == ("SupportAgent", 1.0, 1.0)
    assert case_row[0] == "hi"
    assert case_row[1] == "hello"
    assert case_row[2] == 1
    assert json.loads(case_row[3])[0]["metric"] == "task_completion"


def test_save_report_handles_an_empty_dataset(tmp_path: Path) -> None:
    """Saving a report with no cases still records the run row."""
    db_path = tmp_path / "runa.db"
    report = Report(agent_name="SupportAgent", cases=[])

    run_id = save_report(report, db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM eval_cases WHERE run_id = ?", (run_id,)
        ).fetchone()[0]

    assert count == 0


def test_list_eval_runs_returns_summaries_newest_first(tmp_path: Path) -> None:
    """`list_eval_runs` returns every run without its cases, most recent id first."""
    db_path = tmp_path / "runa.db"
    save_report(Report(agent_name="A", cases=[]), db_path=db_path)
    save_report(Report(agent_name="B", cases=[]), db_path=db_path)

    runs = list_eval_runs(db_path=db_path)

    assert [run.agent_name for run in runs] == ["B", "A"]
    assert runs[0].cases == []


def test_get_eval_run_returns_none_for_an_unknown_id(tmp_path: Path) -> None:
    """`get_eval_run` returns `None` when `db_path` has no such `eval_runs.id`."""
    db_path = tmp_path / "runa.db"

    assert get_eval_run(999, db_path=db_path) is None


def test_get_eval_run_returns_the_run_with_its_cases(tmp_path: Path) -> None:
    """`get_eval_run` reconstructs each `EvalCaseRow`, including its parsed `results`."""
    db_path = tmp_path / "runa.db"
    case_report = CaseReport(
        index=0,
        case=Case(input="hi", expected="hello"),
        run=AgentRun(input="hi", final_output="hello"),
        results=[
            EvaluationResult(metric="task_completion", status=Status.PASS, reason="ok", score=1.0)
        ],
    )
    run_id = save_report(Report(agent_name="A", cases=[case_report]), db_path=db_path)

    run = get_eval_run(run_id, db_path=db_path)

    assert run is not None
    assert run.agent_name == "A"
    assert len(run.cases) == 1
    assert run.cases[0].input == "hi"
    assert run.cases[0].output == "hello"
    assert run.cases[0].passed is True
    assert run.cases[0].results[0]["metric"] == "task_completion"


def _graded(input: str, status: Status) -> CaseReport:
    return CaseReport(
        index=0,
        case=Case(input=input),
        run=AgentRun(input=input, final_output="ok"),
        results=[EvaluationResult(metric="task_completion", status=status, reason="r")],
    )


def test_load_baseline_maps_the_latest_run_s_inputs_to_their_verdicts(tmp_path: Path) -> None:
    """Only the agent's most recent run counts, keyed by input."""
    db_path = tmp_path / "runa.db"
    save_report(Report("A", [_graded("hi", Status.FAIL)]), db_path=db_path)
    save_report(Report("A", [_graded("hi", Status.PASS)]), db_path=db_path)
    save_report(Report("B", [_graded("yo", Status.FAIL)]), db_path=db_path)

    assert load_baseline("A", db_path=db_path) == {"hi": True}
    assert load_baseline("never_run", db_path=db_path) is None
