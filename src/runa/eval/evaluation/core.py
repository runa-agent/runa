"""eval/evaluation/core.py: `Status` and `EvaluationResult`, the shape every check returns.

Deterministic checks (`eval/evaluation/deterministic.py`) and judge-graded
semantic metrics (`eval/evaluation/semantic.py`) return unrelated native
shapes; every caller past this module deals only in `EvaluationResult`, so a
`Report` can aggregate them without caring which kind produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Status(StrEnum):
    """A metric's outcome for one case.

    `SKIPPED` means the metric didn't apply (e.g. faithfulness with no
    retrieval context); `ERROR` means the check itself failed to run (e.g.
    the judge model call raised). Neither is a score of 0 or 1: a `PASS`/
    `FAIL` verdict claims to have actually measured something.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


@dataclass
class EvaluationResult:
    """One metric's verdict on one case: deterministic or semantic, same shape either way."""

    metric: str
    status: Status
    reason: str
    score: float | None = None

    @property
    def passed(self) -> bool:
        """Whether this result counts toward a case passing (`SKIPPED` doesn't count against it)."""
        return self.status in (Status.PASS, Status.SKIPPED)
