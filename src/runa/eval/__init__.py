"""`runa.eval`: `Case`/`Dataset` in, a `Report` out, see `Agent.evaluate()`.

Deterministic checks, judge-model-backed semantic metrics, and SQLite
storage are all implementation details behind `evaluate_agent()`; see
`eval/evaluation/`, `eval/judge.py`, and `eval/storage.py`.
"""

from runa.eval.case import Case
from runa.eval.dataset import Dataset
from runa.eval.evaluate import evaluate_agent
from runa.eval.evaluation.core import EvaluationResult, Status
from runa.eval.evaluation.defaults import DEFAULT_THRESHOLDS
from runa.eval.report import CaseReport, Report

__all__ = [
    "DEFAULT_THRESHOLDS",
    "Case",
    "CaseReport",
    "Dataset",
    "EvaluationResult",
    "Report",
    "Status",
    "evaluate_agent",
]
