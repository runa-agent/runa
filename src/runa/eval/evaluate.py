"""eval/evaluate.py: `evaluate_agent()`, the implementation behind `agent.evaluate(dataset)`.

Deterministic checks run first and cheaply; semantic metrics only run for a case whose run
actually completed, so a case that errors doesn't also burn a judge-model call. See
`eval/evaluation/deterministic.py` and `eval/evaluation/semantic.py` for what each layer covers.
"""

from collections.abc import Iterable

from runa.agent import Agent
from runa.eval.case import Case
from runa.eval.evaluation.defaults import DEFAULT_THRESHOLDS
from runa.eval.evaluation.deterministic import check_expected_tool_called, check_run_completed
from runa.eval.evaluation.semantic import evaluate_semantic
from runa.eval.report import CaseReport, Report
from runa.eval.storage import load_baseline, save_report
from runa.eval.tracing.adapter import run_agent_for_eval


def _resolve_thresholds(
    threshold: float | None, thresholds: dict[str, float] | None
) -> dict[str, float]:
    resolved = dict(DEFAULT_THRESHOLDS)
    if threshold is not None:
        resolved = dict.fromkeys(resolved, threshold)
    if thresholds:
        resolved.update(thresholds)
    return resolved


def _resolve_judge(judge: str | None, agent: Agent) -> str:
    if judge is not None:
        return judge
    if not isinstance(agent.model, str):
        raise TypeError(
            "agent.evaluate() needs judge=<model name> when the agent's own `model` isn't a "
            "plain model name string"
        )
    return agent.model


async def evaluate_agent(
    agent: Agent,
    dataset: Iterable[Case],
    *,
    judge: str | None = None,
    threshold: float | None = None,
    thresholds: dict[str, float] | None = None,
) -> Report:
    """Run every case in `dataset` through `agent` and grade it, with zero required configuration.

    `judge` overrides the model semantic metrics grade with; it defaults to `agent.model`, so
    grading needs no separate credentials. `threshold` overrides every metric's pass threshold at
    once; `thresholds` overrides just the named ones. Every case runs to completion even if an
    earlier one fails or errors. The finished `Report` is compared against the agent's previous
    run (see `Report.regressions`), then persisted to `runa.db` before it's returned (see
    `eval/storage.py`).
    """
    resolved_thresholds = _resolve_thresholds(threshold, thresholds)
    judge_model_name = _resolve_judge(judge, agent)

    case_reports = []
    for index, case in enumerate(dataset):
        run = await run_agent_for_eval(agent, case)
        results = [check_run_completed(run)]
        if run.error is None:
            tool_result = check_expected_tool_called(case, run)
            if tool_result is not None:
                results.append(tool_result)
            results.extend(
                await evaluate_semantic(
                    case, run, model=judge_model_name, thresholds=resolved_thresholds
                )
            )
        case_reports.append(CaseReport(index=index, case=case, run=run, results=results))

    agent_name = type(agent).__name__
    report = Report(agent_name=agent_name, cases=case_reports, baseline=load_baseline(agent_name))
    save_report(report)
    return report
