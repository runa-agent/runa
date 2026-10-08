"""Tests for `runa.eval.evaluation.semantic`: the judge-model-backed metrics."""

import asyncio
from typing import Any

import pytest
from helpers import finished_run

from runa.eval.case import Case
from runa.eval.evaluation.core import Status
from runa.eval.evaluation.defaults import DEFAULT_THRESHOLDS
from runa.eval.evaluation.semantic import (
    _grade_answer_correctness,
    _grade_answer_relevance,
    _grade_faithfulness,
    _grade_task_completion,
    evaluate_semantic,
)


class _ScriptedJudge:
    """A fake judge that replies based on which JSON shape a prompt is asking for.

    Each entry pairs a substring found only in the prompt(s) it should answer with the reply to
    return for it, so a test can script a whole multi-call pipeline without caring about call
    order (faithfulness's truths/claims calls run concurrently via `asyncio.gather`).
    """

    def __init__(self, replies: list[tuple[str, str]]) -> None:
        self._replies = replies
        self.calls: list[str] = []

    async def ask(self, prompt: str) -> str:
        self.calls.append(prompt)
        for marker, reply in self._replies:
            if marker in prompt:
                return reply
        raise AssertionError(f"no scripted reply matches prompt: {prompt[:200]!r}")


def test_grade_task_completion_extracts_then_scores() -> None:
    """Task completion runs two calls: extract task/outcome, then verdict the outcome."""
    judge = _ScriptedJudge(
        [
            (
                '"task": "...", "outcome"',
                '{"task": "cancel order 123", "outcome": "order 123 was cancelled"}',
            ),
            ('"verdict": <float 0-1>', '{"verdict": 0.9, "reason": "fully achieved"}'),
        ]
    )
    case = Case(input="cancel order 123")
    run = finished_run("Order 123 is cancelled.")

    score, reason = asyncio.run(_grade_task_completion(judge, case, run))

    assert score == 0.9
    assert reason == "fully achieved"
    assert len(judge.calls) == 2


def test_grade_answer_relevance_scores_the_fraction_not_irrelevant() -> None:
    """Score is `count(verdict != 'no') / total`, matching DeepEval's own formula."""
    judge = _ScriptedJudge(
        [
            ('"statements":', '{"statements": ["a", "b", "c"]}'),
            (
                '"verdicts":',
                '{"verdicts": ['
                '{"verdict": "yes"}, {"verdict": "no", "reason": "off topic"}, '
                '{"verdict": "idk", "reason": "ambiguous"}'
                "]}",
            ),
            ('{"reason": "..."}', '{"reason": "mostly relevant, one aside"}'),
        ]
    )
    case = Case(input="what's the weather?")
    run = finished_run("sunny, also I like cats, maybe rain later")

    score, reason = asyncio.run(_grade_answer_relevance(judge, case, run))

    assert score == pytest.approx(2 / 3)
    assert reason == "mostly relevant, one aside"


def test_grade_answer_relevance_treats_no_statements_as_perfect() -> None:
    """An empty output makes no statements, matching DeepEval's `number_of_verdicts == 0 -> 1`."""
    judge = _ScriptedJudge([('"statements":', '{"statements": []}')])
    case = Case(input="hi")
    run = finished_run("")

    score, reason = asyncio.run(_grade_answer_relevance(judge, case, run))

    assert score == 1.0
    assert len(judge.calls) == 1


def test_grade_answer_correctness_generates_steps_then_scores_out_of_ten() -> None:
    """GEval's mechanic: steps from the criteria once, then a 0-10 score normalized to 0-1."""
    judge = _ScriptedJudge(
        [
            ('"steps":', '{"steps": ["compare key claims", "check for contradictions"]}'),
            ('"score": <int 0-10>', '{"score": 8, "reason": "matches the key claim"}'),
        ]
    )
    case = Case(input="who won?", expected="the home team won")
    run = finished_run("the home team won 3-1")

    score, reason = asyncio.run(_grade_answer_correctness(judge, case, run))

    assert score == pytest.approx(0.8)
    assert reason == "matches the key claim"


def test_grade_faithfulness_scores_the_fraction_not_contradicted() -> None:
    """Truths and claims are extracted concurrently, then each claim is verdicted against them."""
    judge = _ScriptedJudge(
        [
            ('"truths":', '{"truths": ["refunds are processed within 30 days"]}'),
            ('"claims":', '{"claims": ["refunds are processed within 30 days", "no fee applies"]}'),
            (
                '"verdicts":',
                '{"verdicts": [{"verdict": "yes"}, '
                '{"verdict": "no", "reason": "context never mentions fees"}]}',
            ),
            ('{"reason": "..."}', '{"reason": "one unsupported claim about fees"}'),
        ]
    )
    case = Case(input="what's the refund policy?", context=["refunds are processed within 30 days"])
    run = finished_run("refunds within 30 days, no fee applies")

    score, reason = asyncio.run(_grade_faithfulness(judge, case, run))

    assert score == pytest.approx(0.5)
    assert reason == "one unsupported claim about fees"


def test_grade_faithfulness_treats_no_claims_as_perfect() -> None:
    """An output with no factual claims can't contradict the context, matching DeepEval."""
    judge = _ScriptedJudge(
        [
            ('"truths":', '{"truths": ["x"]}'),
            ('"claims":', '{"claims": []}'),
        ]
    )
    case = Case(input="hi", context=["x"])
    run = finished_run("ok")

    score, reason = asyncio.run(_grade_faithfulness(judge, case, run))

    assert score == 1.0


def test_evaluate_semantic_skips_answer_correctness_without_an_expected_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No `case.expected` means `answer_correctness` is `SKIPPED`, never graded."""
    judge = _ScriptedJudge(
        [
            ('"task": "...", "outcome"', '{"task": "t", "outcome": "o"}'),
            ('"verdict": <float 0-1>', '{"verdict": 1.0, "reason": "ok"}'),
            ('"statements":', '{"statements": []}'),
        ]
    )
    monkeypatch.setattr("runa.eval.evaluation.semantic.judge_model", lambda model: judge)

    case = Case(input="hi")
    run = finished_run("ok")
    results = asyncio.run(
        evaluate_semantic(case, run, model="gpt-5.4-nano", thresholds=DEFAULT_THRESHOLDS)
    )

    by_metric = {r.metric: r for r in results}
    assert by_metric["answer_correctness"].status == Status.SKIPPED
    assert by_metric["faithfulness"].status == Status.SKIPPED


def test_evaluate_semantic_maps_a_raised_exception_to_error_not_a_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A judge call that can't be answered becomes `ERROR`, never a `PASS`/`FAIL` of convenience."""

    class _BrokenJudge:
        async def ask(self, prompt: str) -> Any:
            raise RuntimeError("judge call failed")

    monkeypatch.setattr("runa.eval.evaluation.semantic.judge_model", lambda model: _BrokenJudge())

    case = Case(input="hi")
    run = finished_run("ok")
    results = asyncio.run(
        evaluate_semantic(case, run, model="gpt-5.4-nano", thresholds=DEFAULT_THRESHOLDS)
    )

    by_metric = {r.metric: r for r in results}
    assert by_metric["task_completion"].status == Status.ERROR
    assert by_metric["task_completion"].score is None
    assert "judge call failed" in by_metric["task_completion"].reason
