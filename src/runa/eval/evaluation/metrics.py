"""eval/evaluation/metrics.py: every metric Runa grades, declared once, as data.

A metric used to be a bare string, and the closed set of them was spelled out four times: the
semantic runner produced the names and encoded each one's activation rule inline, a
`DEFAULT_THRESHOLDS` dict keyed thresholds by the same names, `eval/report.py` kept a display
order and a label map listing them again, and `Report.score` hard-coded one of the names as the
headline. Nothing tied the four together, so adding a metric was five edits in three files -- and
the easiest one to forget produced the worst symptom, because the report iterated its own order
list and dropped any metric missing from it, while that metric still decided whether cases passed.

So a metric is declared here, once: its name, how a report labels it, the score it must reach,
when it applies, and the pipeline that grades it. `evaluate_semantic()` below iterates this table
instead of listing metrics by hand, `eval/evaluation/deterministic.py` builds its results from the
two entries it owns, and `eval/report.py` reads labels and order from it, with no name of its own.
A result whose metric isn't declared here is a `KeyError` at the display path (see
`in_display_order`), the way a missing threshold has always been one.

What stays out of the table is the prompts: they're each metric's content, not its identity, and
they live with the pipelines in `eval/evaluation/semantic.py`.
"""

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass

from runa.eval.case import Case
from runa.eval.evaluation.core import EvaluationResult, Status
from runa.eval.evaluation.semantic import (
    grade_answer_correctness,
    grade_answer_relevance,
    grade_faithfulness,
    grade_task_completion,
)
from runa.eval.judge import JudgeModel, judge_model
from runa.run import Run

Grader = Callable[[JudgeModel, Case, Run], Awaitable[tuple[float, str]]]
"""A metric's judge pipeline: one case in, a `(score, reason)` pair out (see `semantic.py`)."""

Requirement = Callable[[Case], str | None]
"""What a metric needs of a case: `None` when the case supplies it, else why it doesn't."""


def _needs_nothing(case: Case) -> str | None:
    return None


@dataclass(frozen=True)
class Metric:
    """One metric's identity: what it's called, how it's shown, and when it has something to say.

    Declared once in `METRICS` below, never assembled by a caller: a metric a run can produce is
    a metric a report can name and a threshold can be keyed by, because all three read this.
    """

    name: str
    """The name every `EvaluationResult` for this metric carries, and its key in `thresholds`."""

    label: str
    """How a rendered `Report` names it."""

    threshold: float | None = None
    """The score a judged metric must reach to pass; `None` for a deterministic check, whose
    plain-code verdict is already a `PASS`/`FAIL` and scores 1.0/0.0 to match."""

    grade: Grader | None = None
    """The judge pipeline that scores it, or `None` when `deterministic.py` checks it instead."""

    missing: Requirement = _needs_nothing
    """What the case must supply for this metric to mean anything, answered as the reason it's
    `SKIPPED` without it: correctness needs a reference answer, faithfulness needs context."""

    @property
    def judged(self) -> bool:
        """Whether a judge model grades this metric, rather than `deterministic.py`'s own code."""
        return self.grade is not None

    def result(
        self, status: Status, reason: str, *, score: float | None = None
    ) -> EvaluationResult:
        """This metric's verdict on one case: the single place its `name` reaches a result."""
        return EvaluationResult(metric=self.name, status=status, reason=reason, score=score)

    async def evaluate(
        self, judge: JudgeModel, case: Case, run: Run, *, threshold: float
    ) -> EvaluationResult:
        """Grade one case with this metric's pipeline.

        `SKIPPED` when the case is `missing` what this metric needs, `ERROR` when the pipeline
        raises -- never a score of convenience, and never a raised exception reaching the dataset.
        """
        if self.grade is None:
            raise TypeError(f"{self.name!r} is graded by deterministic.py, not by a judge model")
        reason = self.missing(case)
        if reason is not None:
            return self.result(Status.SKIPPED, reason)
        try:
            score, reason = await self.grade(judge, case, run)
        except Exception as exc:
            return self.result(Status.ERROR, str(exc))
        status = Status.PASS if score >= threshold else Status.FAIL
        return self.result(status, reason, score=score)


RUN_COMPLETED = Metric(
    name="run_completed",
    label="Run completed",
    # Not judged and not threshold-able: either the run finished or it didn't. It scores anyway,
    # so a report that lists "run ended 'error'" failures also shows how many runs finished.
)

TASK_COMPLETION = Metric(
    name="task_completion",
    label="Task completion",
    threshold=0.90,
    grade=grade_task_completion,
)

ANSWER_CORRECTNESS = Metric(
    name="answer_correctness",
    label="Answer correctness",
    threshold=0.85,
    grade=grade_answer_correctness,
    missing=lambda case: None if case.expected is not None else "no expected answer",
)

ANSWER_RELEVANCE = Metric(
    name="answer_relevance",
    label="Answer relevance",
    threshold=0.85,
    grade=grade_answer_relevance,
)

FAITHFULNESS = Metric(
    name="faithfulness",
    label="Faithfulness",
    threshold=0.85,
    grade=grade_faithfulness,
    missing=lambda case: None if case.context else "no retrieval context",
)

TOOL_CORRECTNESS = Metric(
    name="tool_correctness",
    label="Tool correctness",
    # Deterministic, like `RUN_COMPLETED`: `case.expected_tool` was called or it wasn't.
)

METRICS: tuple[Metric, ...] = (
    RUN_COMPLETED,
    TASK_COMPLETION,
    ANSWER_CORRECTNESS,
    ANSWER_RELEVANCE,
    FAITHFULNESS,
    TOOL_CORRECTNESS,
)
"""Every metric Runa grades, in the order a report shows them."""

DEFAULT_THRESHOLDS: dict[str, float] = {
    metric.name: metric.threshold for metric in METRICS if metric.threshold is not None
}
"""Each judged metric's default pass threshold, read off `METRICS`.

Implementation defaults, not promises of universal correctness. Overridable via
`agent.evaluate(dataset, threshold=...)` (applies to every metric) or `thresholds={...}`
(applies to just the named ones).
"""

_BY_NAME = {metric.name: metric for metric in METRICS}


def in_display_order(names: Iterable[str]) -> list[Metric]:
    """The metrics `names` refers to, in `METRICS` order.

    A name no metric declares raises `KeyError` rather than being skipped: a result that reached a
    `Report` already decided whether its case passed, so dropping it would print metric lines that
    don't explain the failures listed under them.
    """
    chosen = {_BY_NAME[name] for name in names}
    return [metric for metric in METRICS if metric in chosen]


async def evaluate_semantic(
    case: Case, run: Run, *, model: str, thresholds: dict[str, float]
) -> list[EvaluationResult]:
    """Run every judged metric against `case`, recording the ones that don't apply as `SKIPPED`."""
    judge = judge_model(model)
    return [
        await metric.evaluate(judge, case, run, threshold=thresholds[metric.name])
        for metric in METRICS
        if metric.judged
    ]


__all__ = [
    "ANSWER_CORRECTNESS",
    "ANSWER_RELEVANCE",
    "DEFAULT_THRESHOLDS",
    "FAITHFULNESS",
    "METRICS",
    "RUN_COMPLETED",
    "TASK_COMPLETION",
    "TOOL_CORRECTNESS",
    "Metric",
    "evaluate_semantic",
    "in_display_order",
]
