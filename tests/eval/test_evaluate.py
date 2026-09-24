"""Tests for `runa.eval.evaluate`: `evaluate_agent`, behind `agent.evaluate()`."""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from runa.agent import Agent
from runa.eval.case import Case
from runa.eval.evaluate import evaluate_agent
from runa.eval.evaluation.core import EvaluationResult, Status
from runa.tracing import Trace


class _TestAgent(Agent):
    """A minimal `Agent` for `evaluate_agent` to run."""

    name = "UnderTest"
    model = "gpt-5.4-nano"


_AGENT = _TestAgent()


@dataclass
class _FakeResult:
    final_output: Any = "ok"
    new_items: list[Any] = field(default_factory=list)
    trace: Trace = field(default_factory=lambda: Trace(id="t", name="t", start_time=0.0, spans=[]))


def _patch_run_and_storage(monkeypatch: pytest.MonkeyPatch, outputs: dict[str, Any]) -> None:
    async def fake_run(agent: Any, input: Any, **kwargs: Any) -> Any:
        if input in outputs and isinstance(outputs[input], Exception):
            raise outputs[input]
        return _FakeResult(final_output=outputs.get(input, input))

    monkeypatch.setattr("runa.eval.tracing.adapter.Runner.run", staticmethod(fake_run))
    monkeypatch.setattr("runa.eval.evaluate.save_report", lambda report: 1)
    monkeypatch.setattr("runa.eval.evaluate.load_baseline", lambda agent_name: None)


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
    """`evaluate_agent` writes the finished `Report` to storage before returning it."""
    _patch_run_and_storage(monkeypatch, {"hi": "hi"})
    _stub_semantic(monkeypatch)
    saved: list[Any] = []
    monkeypatch.setattr("runa.eval.evaluate.save_report", saved.append)

    report = asyncio.run(evaluate_agent(_AGENT, [Case(input="hi")]))

    assert saved == [report]
