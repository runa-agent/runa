"""Tests for `runa.eval.evaluate`: `evaluate_agent`, behind `agent.evaluate()`."""

import asyncio
from typing import Any

import pytest

from runa import db
from runa._types import Usage
from runa.agent import Agent
from runa.eval.case import Case
from runa.eval.evaluate import evaluate_agent
from runa.eval.evaluation.core import EvaluationResult, Status
from runa.run import Run
from runa.tracing import Trace


class _TestAgent(Agent):
    """A minimal `Agent` for `evaluate_agent` to run."""

    name = "UnderTest"
    model = "gpt-5.4-nano"


_AGENT = _TestAgent()


def _run(output: Any) -> Run:
    """The `Run` a faked `Agent.run` hands `_evaluate_case`."""
    return Run(
        output=output,
        trace=Trace(id="t", name="t", start_time=0.0, spans=[]),
        usage=Usage(),
    )


def _patch_run_and_storage(monkeypatch: pytest.MonkeyPatch, outputs: dict[str, Any]) -> None:
    """Fake the model call, and send eval history to the in-process store.

    `memory://` is a real `EvalStore`, so `evaluate_agent`'s own save and baseline lookup run
    exactly as they do in an app. They used to be two monkeypatched module attributes, which
    meant nothing here exercised the path from a finished `Report` to stored history.
    """

    async def fake_run(self: Agent, message: Any, *args: Any, **kwargs: Any) -> Run:
        if message in outputs and isinstance(outputs[message], Exception):
            exc = outputs[message]
            return Run(output=None, trace=None, usage=Usage(), status="error", error=str(exc))
        return _run(outputs.get(message, message))

    monkeypatch.setattr(Agent, "run", fake_run)
    monkeypatch.setenv(db.DATABASE_URL_ENV, "memory://")


def _stub_semantic(monkeypatch: pytest.MonkeyPatch, status: Status = Status.PASS) -> None:
    async def fake_evaluate_semantic(case: Any, run: Any, **kwargs: Any) -> list[EvaluationResult]:
        return [EvaluationResult(metric="task_completion", status=status, reason="stub", score=1.0)]

    monkeypatch.setattr("runa.eval.evaluate.evaluate_semantic", fake_evaluate_semantic)


def test_evaluate_agent_runs_every_case_in_a_mixed_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dataset mixing a passing, a failing, and an erroring case evaluates every one."""
    from runa.exceptions import MaxTurnsExceeded

    _patch_run_and_storage(
        monkeypatch,
        {"good": "good", "bad": MaxTurnsExceeded("too many turns")},
    )

    async def fake_evaluate_semantic(case: Any, run: Any, **kwargs: Any) -> list[EvaluationResult]:
        return [
            EvaluationResult(metric="task_completion", status=Status.PASS, reason="ok", score=1.0)
        ]

    monkeypatch.setattr("runa.eval.evaluate.evaluate_semantic", fake_evaluate_semantic)

    dataset = [Case(input="good"), Case(input="bad")]
    report = asyncio.run(evaluate_agent(_AGENT, dataset))

    assert len(report.cases) == 2
    assert report.cases[0].passed
    assert not report.cases[1].passed
    assert report.cases[1].run.error is not None


def test_evaluate_agent_runs_each_case_on_its_own_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each case gets a `_fresh()` copy, so one case's history never reaches the next.

    `evaluate_agent` runs up to `concurrency` cases at once over one `Agent`, which is exactly
    what `Agent._exclusive` refuses for session-less runs and what would otherwise let two cases
    overwrite each other's conversation.
    """
    _patch_run_and_storage(monkeypatch, {})
    _stub_semantic(monkeypatch)
    ran_on: list[Agent] = []

    async def fake_run(self: Agent, message: Any, *args: Any, **kwargs: Any) -> Run:
        ran_on.append(self)
        return _run(message)

    monkeypatch.setattr(Agent, "run", fake_run)

    asyncio.run(evaluate_agent(_AGENT, [Case(input="one"), Case(input="two")]))

    assert len(ran_on) == 2
    assert ran_on[0] is not ran_on[1]
    assert _AGENT not in ran_on


def test_evaluate_agent_skips_semantic_metrics_after_a_deterministic_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A case whose run errors never reaches semantic evaluation (no wasted judge calls)."""
    from runa.exceptions import MaxTurnsExceeded

    _patch_run_and_storage(monkeypatch, {"bad": MaxTurnsExceeded("boom")})

    called = False

    async def fake_evaluate_semantic(case: Any, run: Any, **kwargs: Any) -> list[EvaluationResult]:
        nonlocal called
        called = True
        return []

    monkeypatch.setattr("runa.eval.evaluate.evaluate_semantic", fake_evaluate_semantic)

    report = asyncio.run(evaluate_agent(_AGENT, [Case(input="bad")]))

    assert not called
    assert not report.cases[0].passed


def test_evaluate_agent_handles_an_empty_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty dataset produces a `Report` with no cases, rather than raising."""
    _patch_run_and_storage(monkeypatch, {})
    _stub_semantic(monkeypatch)

    report = asyncio.run(evaluate_agent(_AGENT, []))

    assert report.cases == []
    assert report.pass_rate == 0.0


def test_evaluate_agent_defaults_the_judge_to_the_agent_s_own_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no `judge` override, semantic metrics grade with `agent.model`."""
    _patch_run_and_storage(monkeypatch, {"hi": "hi"})
    captured: dict[str, Any] = {}

    async def fake_evaluate_semantic(case: Any, run: Any, **kwargs: Any) -> list[EvaluationResult]:
        captured["model"] = kwargs["model"]
        return []

    monkeypatch.setattr("runa.eval.evaluate.evaluate_semantic", fake_evaluate_semantic)

    asyncio.run(evaluate_agent(_AGENT, [Case(input="hi")]))

    assert captured["model"] == "gpt-5.4-nano"


def test_evaluate_agent_lets_judge_override_the_grading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit `judge=...` overrides the default model semantic metrics grade with."""
    _patch_run_and_storage(monkeypatch, {"hi": "hi"})
    captured: dict[str, Any] = {}

    async def fake_evaluate_semantic(case: Any, run: Any, **kwargs: Any) -> list[EvaluationResult]:
        captured["model"] = kwargs["model"]
        return []

    monkeypatch.setattr("runa.eval.evaluate.evaluate_semantic", fake_evaluate_semantic)

    asyncio.run(evaluate_agent(_AGENT, [Case(input="hi")], judge="gpt-5.4"))

    assert captured["model"] == "gpt-5.4"


def test_evaluate_agent_applies_a_single_threshold_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`threshold=...` overrides every metric's default threshold at once."""
    _patch_run_and_storage(monkeypatch, {"hi": "hi"})
    captured: dict[str, Any] = {}

    async def fake_evaluate_semantic(case: Any, run: Any, **kwargs: Any) -> list[EvaluationResult]:
        captured["thresholds"] = kwargs["thresholds"]
        return []

    monkeypatch.setattr("runa.eval.evaluate.evaluate_semantic", fake_evaluate_semantic)

    asyncio.run(evaluate_agent(_AGENT, [Case(input="hi")], threshold=0.5))

    assert set(captured["thresholds"].values()) == {0.5}


def test_evaluate_agent_applies_a_per_metric_threshold_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`thresholds={...}` overrides just the named metrics, leaving the rest at their default."""
    _patch_run_and_storage(monkeypatch, {"hi": "hi"})
    captured: dict[str, Any] = {}

    async def fake_evaluate_semantic(case: Any, run: Any, **kwargs: Any) -> list[EvaluationResult]:
        captured["thresholds"] = kwargs["thresholds"]
        return []

    monkeypatch.setattr("runa.eval.evaluate.evaluate_semantic", fake_evaluate_semantic)

    asyncio.run(evaluate_agent(_AGENT, [Case(input="hi")], thresholds={"faithfulness": 0.99}))

    assert captured["thresholds"]["faithfulness"] == 0.99
    assert captured["thresholds"]["task_completion"] == 0.90


def test_evaluate_agent_persists_the_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """`evaluate_agent` writes the finished `Report` to the store before returning it."""
    _patch_run_and_storage(monkeypatch, {"hi": "hi"})
    _stub_semantic(monkeypatch)

    report = asyncio.run(evaluate_agent(_AGENT, [Case(input="hi")]))

    stored = db.evals().list()
    assert [run.agent_name for run in stored] == [report.agent_name]
    saved = db.evals().get(stored[0].id)
    assert saved is not None
    assert [case.input for case in saved.cases] == ["hi"]


def test_evaluate_agent_grades_against_its_previous_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """The run just saved is the baseline the next one compares against.

    One store, resolved once: the baseline read and the history write used to be separate
    questions to `runa.db`, each able to answer with a different backend.
    """
    _patch_run_and_storage(monkeypatch, {"hi": "hi"})
    _stub_semantic(monkeypatch, status=Status.FAIL)
    asyncio.run(evaluate_agent(_AGENT, [Case(input="hi")]))

    _stub_semantic(monkeypatch, status=Status.PASS)
    report = asyncio.run(evaluate_agent(_AGENT, [Case(input="hi")]))

    assert report.baseline == {"hi": False}


@pytest.mark.parametrize(("concurrency", "expected_peak"), [(8, 3), (2, 2), (1, 1)])
def test_evaluate_agent_runs_cases_concurrently_in_dataset_order(
    monkeypatch: pytest.MonkeyPatch, concurrency: int, expected_peak: int
) -> None:
    """Up to `concurrency` cases run at once, and the report keeps the dataset's order."""
    _patch_run_and_storage(monkeypatch, {})
    _stub_semantic(monkeypatch)
    running, peak = 0, 0

    async def slow_run(self: Agent, message: Any, *args: Any, **kwargs: Any) -> Run:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01 if message == "a" else 0)
        running -= 1
        return _run(message)

    monkeypatch.setattr(Agent, "run", slow_run)
    dataset = [Case(input="a"), Case(input="b"), Case(input="c")]

    report = asyncio.run(evaluate_agent(_AGENT, dataset, concurrency=concurrency))

    assert peak == expected_peak
    assert [case.case.input for case in report.cases] == ["a", "b", "c"]
