"""Tests for `runa.eval.evaluation.deterministic`: checks that don't need a judge model."""

from runa.eval.case import Case
from runa.eval.evaluation.core import Status
from runa.eval.evaluation.deterministic import check_expected_tool_called, check_run_completed
from runa.eval.tracing.adapter import AgentRun
from runa.tool import ToolCall


def test_check_run_completed_passes_when_there_is_no_error() -> None:
    """A run with no `error` passes, with a score of 1.0."""
    result = check_run_completed(AgentRun(input="hi", final_output="ok"))

    assert result.status == Status.PASS
    assert result.score == 1.0


def test_check_run_completed_fails_and_reports_the_error() -> None:
    """A run with `error` set fails, carrying the error message as the reason."""
    result = check_run_completed(AgentRun(input="hi", final_output=None, error="boom"))

    assert result.status == Status.FAIL
    assert result.reason == "boom"


def test_check_expected_tool_called_returns_none_when_no_tool_is_expected() -> None:
    """A case with no `expected_tool` has nothing for this check to assert, so it returns `None`."""
    result = check_expected_tool_called(Case(input="hi"), AgentRun(input="hi", final_output="ok"))

    assert result is None


def test_check_expected_tool_called_passes_when_the_tool_was_called() -> None:
    """Passes when a tool call in the run matches `case.expected_tool`."""
    case = Case(input="cancel order 123", expected_tool="cancel_order")
    run = AgentRun(
        input=case.input,
        final_output="done",
        tool_calls=[ToolCall(name="cancel_order", arguments="{}", output="done")],
    )

    result = check_expected_tool_called(case, run)

    assert result is not None
    assert result.status == Status.PASS


def test_check_expected_tool_called_fails_when_the_tool_was_not_called() -> None:
    """Fails, naming what was actually called, when `expected_tool` never ran."""
    case = Case(input="cancel order 123", expected_tool="cancel_order")
    run = AgentRun(
        input=case.input,
        final_output="done",
        tool_calls=[ToolCall(name="search", arguments="{}", output="nothing")],
    )

    result = check_expected_tool_called(case, run)

    assert result is not None
    assert result.status == Status.FAIL
    assert "cancel_order" in result.reason
    assert "search" in result.reason
