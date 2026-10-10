"""Tests for `runa.eval.evaluation.metrics`: the one table every metric is declared in."""

import asyncio

import pytest
from helpers import finished_run

from runa.eval.case import Case
from runa.eval.evaluation.metrics import (
    ANSWER_CORRECTNESS,
    DEFAULT_THRESHOLDS,
    FAITHFULNESS,
    METRICS,
    RUN_COMPLETED,
    TASK_COMPLETION,
    in_display_order,
)


class _NoJudge:
    """A judge a deterministic metric must never reach for."""

    async def ask(self, prompt: str) -> str:
        raise AssertionError("a deterministic metric asked a judge model")


_NO_JUDGE = _NoJudge()


def test_every_metric_is_declared_once_with_a_label() -> None:
    """One entry per metric, each with the label a report shows it under."""
    names = [metric.name for metric in METRICS]

    assert len(set(names)) == len(names)
    assert all(metric.label for metric in METRICS)


def test_default_thresholds_are_read_off_the_table() -> None:
    """`DEFAULT_THRESHOLDS` isn't a second list of metrics, it's derived from `METRICS`."""
    declared = {metric.name: metric.threshold for metric in METRICS if metric.judged}

    assert declared == DEFAULT_THRESHOLDS
    assert DEFAULT_THRESHOLDS[TASK_COMPLETION.name] == 0.90


def test_a_threshold_is_exactly_what_a_judged_metric_has() -> None:
    """A deterministic check scores 1.0/0.0 and has nothing to threshold, so it declares none."""
    assert all(metric.judged == (metric.threshold is not None) for metric in METRICS)
    assert RUN_COMPLETED.name not in DEFAULT_THRESHOLDS


def test_what_a_metric_needs_is_declared_data_not_a_rule_inline_in_the_runner() -> None:
    """Each metric says what a case must supply, as the reason it's skipped when it doesn't."""
    assert ANSWER_CORRECTNESS.missing(Case(input="hi")) == "no expected answer"
    assert ANSWER_CORRECTNESS.missing(Case(input="hi", expected="hello")) is None
    assert FAITHFULNESS.missing(Case(input="hi")) == "no retrieval context"
    assert FAITHFULNESS.missing(Case(input="hi", context=["x"])) is None
    assert TASK_COMPLETION.missing(Case(input="hi")) is None


def test_in_display_order_follows_the_declaration_not_the_caller() -> None:
    """A report shows metrics in `METRICS` order, whatever order the results came back in."""
    assert in_display_order(["faithfulness", "task_completion"]) == [
        TASK_COMPLETION,
        FAITHFULNESS,
    ]


def test_in_display_order_refuses_a_metric_the_table_doesnt_declare() -> None:
    """An undeclared metric raises rather than being dropped from the display it decided."""
    with pytest.raises(KeyError):
        in_display_order(["task_completion", "vibes"])


def test_a_deterministic_metric_cant_be_asked_for_a_judged_verdict() -> None:
    """`Metric.evaluate` is the judged path; `deterministic.py` owns the other two entries."""
    run = finished_run("ok")

    with pytest.raises(TypeError):
        asyncio.run(RUN_COMPLETED.evaluate(_NO_JUDGE, Case(input="hi"), run, threshold=1.0))
