"""eval/evaluate.py: `evaluate_agent()`, the implementation behind `agent.evaluate(dataset)`.

Deterministic checks run first and cheaply; semantic metrics only run for a case whose run
actually completed, so a case that errors doesn't also burn a judge-model call. See
`eval/evaluation/deterministic.py` and `eval/evaluation/semantic.py` for what each layer covers.
"""

from __future__ import annotations

import asyncio
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


async def _evaluate_case(
    agent: Agent, index: int, case: Case, *, judge: str, thresholds: dict[str, float]
) -> CaseReport:
    run = await run_agent_for_eval(agent, case)
    results = [check_run_completed(run)]
    if run.error is None:
        tool_result = check_expected_tool_called(case, run)
        if tool_result is not None:
            results.append(tool_result)
        results.extend(await evaluate_semantic(case, run, model=judge, thresholds=thresholds))
    return CaseReport(index=index, case=case, run=run, results=results)


async def evaluate_agent(
    agent: Agent,
    dataset: Iterable[Case],
    *,
    judge: str | None = None,
    threshold: float | None = None,
    thresholds: dict[str, float] | None = None,
    concurrency: int = 8,
) -> Report:
    """Run every case in `dataset` through `agent` and grade it, with zero required configuration.

    `judge` overrides the model semantic metrics grade with; it defaults to `agent.model`, so
    grading needs no separate credentials. `threshold` overrides every metric's pass threshold at
    once; `thresholds` overrides just the named ones. Up to `concurrency` cases run at once, and
    `concurrency=1` runs them one by one, for an agent whose tools can't run in parallel. Every
    case runs to completion even if another fails or errors. The finished `Report` keeps dataset
    order, is compared against the agent's previous run (see `Report.regressions`), then persisted
    to `runa.db` before it's returned (see `eval/storage.py`).
    """
    resolved_thresholds = _resolve_thresholds(threshold, thresholds)
    judge_model_name = _resolve_judge(judge, agent)
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(index: int, case: Case) -> CaseReport:
        async with semaphore:
            return await _evaluate_case(
                agent, index, case, judge=judge_model_name, thresholds=resolved_thresholds
            )

    case_reports = await asyncio.gather(
        *(_bounded(index, case) for index, case in enumerate(dataset))
    )

    report = Report(
        agent_name=agent.name, cases=list(case_reports), baseline=load_baseline(agent.name)
    )
    save_report(report)
    return report
