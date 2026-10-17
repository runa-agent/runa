"""`runa.eval`: `Case`/`Dataset` in, a `Report` out, see `Agent.evaluate()`.

Deterministic checks, judge-model-backed semantic metrics, and where a run's
history is kept are all implementation details behind `evaluate_agent()`; see
`eval/evaluation/`, `eval/judge.py`, and `eval/store.py`.

Reading that history back is `runa.db.evals()`, which hands over an `EvalStore`
already pointed at whichever backend this deployment has.
"""

from runa.eval.case import Case
from runa.eval.dataset import Dataset
from runa.eval.evaluate import evaluate_agent
from runa.eval.evaluation.core import EvalStatus, EvaluationResult
from runa.eval.evaluation.metrics import DEFAULT_THRESHOLDS, METRICS, Metric
from runa.eval.report import CaseReport, Report

__all__ = [
    "DEFAULT_THRESHOLDS",
    "METRICS",
    "Case",
    "CaseReport",
    "Dataset",
    "EvaluationResult",
    "Metric",
    "Report",
    "EvalStatus",
    "evaluate_agent",
]


def __getattr__(name: str) -> object:
    """Resolve `Status`, the deprecated spelling of `EvalStatus`, with a warning."""
    if name != "Status":
        raise AttributeError(f"module 'runa.eval' has no attribute {name!r}")
    import warnings

    warnings.warn(
        "runa.eval.Status is deprecated, use runa.eval.EvalStatus", DeprecationWarning, stacklevel=2
    )
    return EvalStatus
