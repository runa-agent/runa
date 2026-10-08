"""eval/evaluation/deterministic.py: checks that don't need a judge model.

Run before any semantic metric (see `eval/evaluate.py`): cheaper, and a
model call can't answer these more reliably than plain code can.
"""

from runa.eval.case import Case
from runa.eval.evaluation.core import EvaluationResult, Status
from runa.run import Run


def check_run_completed(run: Run) -> EvaluationResult:
    """Fail unless the run finished, rather than let a missing output confuse later metrics.

    `Run.status` is the question, not `Run.error`: a run paused on a human approval has no error
    and no output either, and grading it as if the agent had answered would score the agent on a
    `None` the judge can only mark down.
    """
    if run.status != "completed":
        return EvaluationResult(
            metric="run_completed",
            status=Status.FAIL,
            reason=run.error or f"run ended {run.status!r}",
        )
    return EvaluationResult(
        metric="run_completed", status=Status.PASS, reason="run completed successfully", score=1.0
    )


def check_expected_tool_called(case: Case, run: Run) -> EvaluationResult | None:
    """Assert `case.expected_tool` was called, or `None` if the case declares no expected tool."""
    if case.expected_tool is None:
        return None
    called = {tool_call.name for tool_call in run._tool_calls}
    if case.expected_tool in called:
        return EvaluationResult(
            metric="tool_correctness",
            status=Status.PASS,
            reason=f"called {case.expected_tool!r}",
            score=1.0,
        )
    return EvaluationResult(
        metric="tool_correctness",
        status=Status.FAIL,
        reason=f"expected {case.expected_tool!r} to be called, got {sorted(called)}",
        score=0.0,
    )
