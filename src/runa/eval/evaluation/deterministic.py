"""eval/evaluation/deterministic.py: checks that don't need a judge model.

Run before any semantic metric (see `eval/evaluate.py`): cheaper, and a
model call can't answer these more reliably than plain code can.
"""

from runa.eval.case import Case
from runa.eval.evaluation.core import EvaluationResult, Status
from runa.eval.tracing.adapter import AgentRun


def check_run_completed(run: AgentRun) -> EvaluationResult:
    """Fail if the run itself raised, rather than let a missing output confuse later metrics."""
    if run.error is not None:
        return EvaluationResult(metric="run_completed", status=Status.FAIL, reason=run.error)
    return EvaluationResult(
        metric="run_completed", status=Status.PASS, reason="run completed successfully", score=1.0
    )


def check_expected_tool_called(case: Case, run: AgentRun) -> EvaluationResult | None:
    """Assert `case.expected_tool` was called, or `None` if the case declares no expected tool."""
    if case.expected_tool is None:
        return None
    called = {tool_call.name for tool_call in run.tool_calls}
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
