"""Tests for `runa.eval.report`: `CaseReport` and `Report` aggregation."""

from runa.eval.case import Case
from runa.eval.evaluation.core import EvaluationResult, Status
from runa.eval.report import CaseReport, Report
from runa.eval.tracing.adapter import AgentRun


def _case_report(index: int, results: list[EvaluationResult]) -> CaseReport:
    return CaseReport(
        index=index,
        case=Case(input=f"input {index}"),
        run=AgentRun(input=f"input {index}", final_output="ok"),
        results=results,
    )


def test_case_report_id_is_stable_and_index_based() -> None:
    """`CaseReport.id` is `case_<index>`, matching the failure-listing format."""
    assert _case_report(23, []).id == "case_23"


def test_case_report_passes_when_every_result_passes_or_is_skipped() -> None:
    """A case with only PASS/SKIPPED results passes."""
    report = _case_report(
        0,
        [
            EvaluationResult(metric="task_completion", status=Status.PASS, reason="ok", score=1.0),
            EvaluationResult(metric="faithfulness", status=Status.SKIPPED, reason="no context"),
        ],
    )

    assert report.passed
    assert report.failure_reason is None


def test_case_report_fails_when_any_result_fails() -> None:
    """A case with a FAIL result fails, and `failure_reason` names it."""
    report = _case_report(
        41,
        [
            EvaluationResult(metric="task_completion", status=Status.PASS, reason="ok", score=1.0),
            EvaluationResult(
                metric="answer_correctness", status=Status.FAIL, reason="unsupported claim"
            ),
        ],
    )

    assert not report.passed
    assert report.failure_reason == "answer_correctness: unsupported claim"


def test_case_report_fails_on_error_not_just_fail() -> None:
    """An ERROR result also fails the case, distinct from a graded FAIL."""
    report = _case_report(
        0, [EvaluationResult(metric="answer_relevance", status=Status.ERROR, reason="judge down")]
    )

    assert not report.passed


def test_report_aggregates_pass_rate_and_score() -> None:
    """`Report.pass_rate`/`score` reflect passed cases and the mean task-completion score."""
    passing = _case_report(
        0,
        [EvaluationResult(metric="task_completion", status=Status.PASS, reason="ok", score=1.0)],
    )
    failing = _case_report(
        1,
        [
            EvaluationResult(metric="task_completion", status=Status.PASS, reason="ok", score=0.6),
            EvaluationResult(metric="answer_correctness", status=Status.FAIL, reason="wrong"),
        ],
    )
    report = Report(agent_name="SupportAgent", cases=[passing, failing])

    assert report.pass_rate == 0.5
    assert report.passed == [passing]
    assert report.failed == report.failures == [failing]
    assert report.score == 0.8
    assert report.metrics["task_completion"] == 0.8


def test_report_handles_an_empty_dataset() -> None:
    """An empty dataset reports a pass rate and score of 0.0, with no failures."""
    report = Report(agent_name="SupportAgent", cases=[])

    assert report.pass_rate == 0.0
    assert report.score == 0.0
    assert report.passed == []
    assert report.failed == []


def test_report_str_lists_metrics_and_failures() -> None:
    """The rendered report names the agent, shows metric percentages, and lists failures."""
    failing = _case_report(
        41,
        [
            EvaluationResult(metric="task_completion", status=Status.PASS, reason="ok", score=0.9),
            EvaluationResult(
                metric="answer_correctness", status=Status.FAIL, reason="unsupported claim"
            ),
        ],
    )
    report = Report(agent_name="SupportAgent", cases=[failing])

    rendered = str(report)

    assert "SupportAgent Evaluation" in rendered
    assert "Task completion" in rendered
    assert "case_41  answer_correctness: unsupported claim" in rendered


def test_report_flags_a_failure_that_passed_last_run_as_a_regression() -> None:
    """A failed case whose input passed in `baseline` regressed, a new or still-failing one not."""
    fail = [EvaluationResult(metric="task_completion", status=Status.FAIL, reason="wrong")]
    regressed, still_failing, new = (_case_report(i, fail) for i in range(3))
    report = Report(
        agent_name="SupportAgent",
        cases=[regressed, still_failing, new],
        baseline={"input 0": True, "input 1": False},
    )

    assert report.regressions == [regressed]
    rendered = str(report)
    assert "1 regressed (last run: 1/2 passed)" in rendered
    assert "case_0  regressed  task_completion: wrong" in rendered
    assert "case_1  task_completion: wrong" in rendered
