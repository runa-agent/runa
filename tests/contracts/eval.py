"""contracts/eval.py: the one `EvalStore` contract, so every backend is held to it.

A baseline is what `agent.evaluate()` grades a run against, so a CI job and a laptop comparing
against "the latest run" have to mean the same thing by it whichever store they read.

Each check takes the store and one tag it uses as its agent name, so a listing or a baseline it
asserts on is its own, not whatever else a shared database already holds.
"""

from collections.abc import Callable, Coroutine
from typing import Any

from helpers import finished_run

from runa.eval.case import Case
from runa.eval.evaluation.core import EvalStatus, EvaluationResult
from runa.eval.report import CaseReport, Report
from runa.eval.store import EvalStore
from runa.tracing import Trace

Check = Callable[[EvalStore, str], Coroutine[Any, Any, None]]


def _graded(
    input: str,
    status: EvalStatus,
    *,
    index: int = 0,
    output: str = "ok",
    trace: Trace | None = None,
) -> CaseReport:
    """One graded case, the unit a `Report` is a list of."""
    return CaseReport(
        index=index,
        case=Case(input=input),
        run=finished_run(output, trace=trace),
        results=[EvaluationResult(metric="task_completion", status=status, reason="r")],
    )


def report(agent_name: str, *, passed: bool = True) -> Report:
    """One graded run for `agent_name`, for a driver that needs a report of its own."""
    status = EvalStatus.PASS if passed else EvalStatus.FAIL
    return Report(agent_name=agent_name, cases=[_graded("hi", status)])


async def check_save_persists_a_run_and_its_cases(store: EvalStore, tag: str) -> None:
    """`save` records the run's score and every case's input, output and verdict."""
    run_id = await store.save(
        Report(agent_name=tag, cases=[_graded("hi", EvalStatus.PASS, output="hello")])
    )

    run = await store.get(run_id)

    assert run is not None
    assert (run.agent_name, run.score, run.pass_rate) == (tag, 1.0, 1.0)
    assert len(run.cases) == 1
    assert run.cases[0].input == "hi"
    assert run.cases[0].output == "hello"
    assert run.cases[0].passed is True
    assert run.cases[0].results[0]["metric"] == "task_completion"


async def check_save_handles_an_empty_dataset(store: EvalStore, tag: str) -> None:
    """Saving a report with no cases still records the run."""
    run_id = await store.save(Report(agent_name=tag, cases=[]))

    run = await store.get(run_id)

    assert run is not None
    assert run.cases == []


async def check_list_returns_runs_newest_first_without_their_cases(
    store: EvalStore, tag: str
) -> None:
    """`list` is a listing: every run, most recent first, and no case rows attached."""
    first = await store.save(Report(agent_name=tag, cases=[_graded("hi", EvalStatus.PASS)]))
    second = await store.save(Report(agent_name=tag, cases=[_graded("hi", EvalStatus.PASS)]))

    listed = await store.list(limit=100)

    assert [run.id for run in listed if run.id in {first, second}] == [second, first]
    assert next(run for run in listed if run.id == second).cases == []


async def check_list_respects_limit(store: EvalStore, tag: str) -> None:
    """`limit` caps the listing."""
    await store.save(Report(agent_name=tag, cases=[]))
    await store.save(Report(agent_name=tag, cases=[]))

    assert len(await store.list(limit=1)) == 1


async def check_get_returns_none_for_an_unknown_id(store: EvalStore, tag: str) -> None:
    """`get` returns `None` when this store has no such run."""
    assert await store.get(2**40) is None


async def check_baseline_maps_the_latest_runs_inputs_to_their_verdicts(
    store: EvalStore, tag: str
) -> None:
    """Only the agent's most recent run counts, keyed by input rather than index."""
    await store.save(Report(tag, [_graded("hi", EvalStatus.FAIL)]))
    await store.save(Report(tag, [_graded("hi", EvalStatus.PASS)]))

    assert await store.baseline(tag) == {"hi": True}


async def check_baseline_before_a_run_is_the_run_it_was_compared_against(
    store: EvalStore, tag: str
) -> None:
    """`before=run_id` skips that run and anything newer, for showing a regression in context."""
    first = await store.save(Report(tag, [_graded("hi", EvalStatus.PASS)]))
    second = await store.save(Report(tag, [_graded("hi", EvalStatus.FAIL)]))

    assert await store.baseline(tag, before=second) == {"hi": True}
    assert await store.baseline(tag, before=first) is None


async def check_baseline_is_none_for_an_agent_that_has_never_run(
    store: EvalStore, tag: str
) -> None:
    """`None`, not an empty dict: "no baseline" and "everything failed" are different answers."""
    assert await store.baseline(f"{tag}-never-run") is None


async def check_save_links_each_case_to_its_runs_trace(store: EvalStore, tag: str) -> None:
    """A case's `trace_id` is its run's trace, so a failing case opens straight onto its spans."""
    first = _graded("hi", EvalStatus.PASS, trace=Trace(id=f"{tag}-first", name=tag, start_time=0.0))
    second = _graded(
        "yo", EvalStatus.PASS, index=1, trace=Trace(id=f"{tag}-second", name=tag, start_time=0.0)
    )

    run_id = await store.save(Report(tag, [first, second]))

    run = await store.get(run_id)
    assert run is not None
    assert [row.trace_id for row in run.cases] == [f"{tag}-first", f"{tag}-second"]


CONTRACT: list[Check] = [
    check_save_persists_a_run_and_its_cases,
    check_save_handles_an_empty_dataset,
    check_list_returns_runs_newest_first_without_their_cases,
    check_list_respects_limit,
    check_get_returns_none_for_an_unknown_id,
    check_baseline_maps_the_latest_runs_inputs_to_their_verdicts,
    check_baseline_before_a_run_is_the_run_it_was_compared_against,
    check_baseline_is_none_for_an_agent_that_has_never_run,
    check_save_links_each_case_to_its_runs_trace,
]
