"""eval/evaluation/deterministic.py: checks that don't need a judge model.

Run before any semantic metric (see `eval/evaluate.py`): cheaper, and a
model call can't answer these more reliably than plain code can.
"""

from runa.eval.case import Case
from runa.eval.evaluation.core import EvaluationResult, Status
from runa.eval.evaluation.metrics import RUN_COMPLETED, TOOL_CORRECTNESS
from runa.run import Run


def check_run_completed(run: Run) -> EvaluationResult:
    """Fail unless the run finished, rather than let a missing output confuse later metrics.

    `Run.status` is the question, not `Run.error`: a run paused on a human approval has no error
    and no output either, and grading it as if the agent had answered would score the agent on a
    `None` the judge can only mark down.
    """
    if run.status != "completed":
        return RUN_COMPLETED.result(
            Status.FAIL, run.error or f"run ended {run.status!r}", score=0.0
        )
    return RUN_COMPLETED.result(Status.PASS, "run completed successfully", score=1.0)


def check_expected_tool_called(case: Case, run: Run) -> EvaluationResult | None:
    """Assert `case.expected_tool` was called, or `None` if the case declares no expected tool."""
    if case.expected_tool is None:
        return None
    called = {tool_call.name for tool_call in run._tool_calls}
    if case.expected_tool in called:
        return TOOL_CORRECTNESS.result(Status.PASS, f"called {case.expected_tool!r}", score=1.0)
    return TOOL_CORRECTNESS.result(
        Status.FAIL,
        f"expected {case.expected_tool!r} to be called, got {sorted(called)}",
        score=0.0,
    )
