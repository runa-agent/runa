"""eval/report.py: `CaseReport` and `Report`, the aggregated result of `agent.evaluate()`."""

from dataclasses import dataclass, field

from runa.eval.case import Case
from runa.eval.evaluation.core import EvaluationResult
from runa.eval.tracing.adapter import AgentRun

_METRIC_ORDER = [
    "task_completion",
    "answer_correctness",
    "answer_relevance",
    "faithfulness",
    "tool_correctness",
]
_METRIC_LABELS = {
    "task_completion": "Task completion",
    "answer_correctness": "Answer correctness",
    "answer_relevance": "Answer relevance",
    "faithfulness": "Faithfulness",
    "tool_correctness": "Tool correctness",
}


@dataclass
class CaseReport:
    """One `Case`'s run plus every `EvaluationResult` graded against it."""

    index: int
    case: Case
    run: AgentRun
    results: list[EvaluationResult] = field(default_factory=list)

    @property
    def id(self) -> str:
        """A short, stable identifier for this case, e.g. `case_23`."""
        return f"case_{self.index}"

    @property
    def passed(self) -> bool:
        """Whether every result passed (`SKIPPED` doesn't count against it)."""
        return all(result.passed for result in self.results)

    @property
    def failure_reason(self) -> str | None:
        """The first non-passing result's reason, or `None` if the case passed."""
        failing = next((result for result in self.results if not result.passed), None)
        return f"{failing.metric}: {failing.reason}" if failing else None


@dataclass
class Report:
    """The aggregated result of `agent.evaluate(dataset)`, under the agent's declared `name`."""

    agent_name: str
    cases: list[CaseReport]
    baseline: dict[str, bool] | None = None
    """Each input of the agent's previous run mapped to whether it passed, `None` if first run."""

    @property
    def passed(self) -> list[CaseReport]:
        """Every case that passed."""
        return [case for case in self.cases if case.passed]

    @property
    def failed(self) -> list[CaseReport]:
        """Every case that didn't pass."""
        return [case for case in self.cases if not case.passed]

    @property
    def failures(self) -> list[CaseReport]:
        """Alias for `failed`, matching the failure-listing use case."""
        return self.failed

    @property
    def regressions(self) -> list[CaseReport]:
        """Every failed case whose input passed in the previous run (see `baseline`)."""
        baseline = self.baseline or {}
        return [case for case in self.failed if baseline.get(case.run.input)]

    @property
    def pass_rate(self) -> float:
        """The fraction of cases that passed, or `0.0` for an empty dataset."""
        return len(self.passed) / len(self.cases) if self.cases else 0.0

    @property
    def metrics(self) -> dict[str, float]:
        """The mean score of each metric, across the cases where it wasn't skipped or errored."""
        scores: dict[str, list[float]] = {}
        for case in self.cases:
            for result in case.results:
                if result.score is not None:
                    scores.setdefault(result.metric, []).append(result.score)
        return {name: sum(values) / len(values) for name, values in scores.items()}

    @property
    def score(self) -> float:
        """The headline score: mean task completion, falling back to the overall pass rate."""
        metrics = self.metrics
        return metrics.get("task_completion", self.pass_rate)

    def __str__(self) -> str:
        """Render the report roughly as `report.score`/`report.metrics`/`report.failures` show."""
        rule = "─" * 32
        lines = [f"{self.agent_name} Evaluation", rule, ""]
        metrics = self.metrics
        for name in _METRIC_ORDER:
            if name in metrics:
                lines.append(f"{_METRIC_LABELS[name]:<20} {metrics[name]:.0%}")
        lines.append("")
        lines.append(f"{len(self.passed)} passed")
        lines.append(f"{len(self.failed)} failed")
        regressions = self.regressions
        if self.baseline:
            previous = f"{sum(self.baseline.values())}/{len(self.baseline)}"
            lines.append(f"{len(regressions)} regressed (last run: {previous} passed)")
        if self.failures:
            lines.append("")
            lines.append("Failures")
            lines.append(rule)
            lines.extend(
                f"{case.id}  {'regressed  ' if case in regressions else ''}{case.failure_reason}"
                for case in self.failures
            )
        return "\n".join(lines)
