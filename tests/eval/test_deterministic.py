"""Tests for `runa.eval.evaluation.deterministic`: checks that don't need a judge model."""

from helpers import finished_run, trace_of

from runa.eval.case import Case
from runa.eval.evaluation.core import Status
from runa.eval.evaluation.deterministic import check_expected_tool_called, check_run_completed
from runa.tool import ToolCall
from runa.tracing import Span, Trace


def test_check_run_completed_passes_for_a_completed_run() -> None:
    """A run whose `status` is `"completed"` passes, with a score of 1.0."""
    result = check_run_completed(finished_run("ok"))

    assert result.status == Status.PASS
    assert result.score == 1.0


def test_check_run_completed_fails_and_reports_the_error() -> None:
    """An errored run fails, carrying the error message as the reason and a score of 0.0.

    It scores rather than leaving the metric unscored so a report's "Run completed" line is the
    fraction of runs that finished, not a constant 100% among the ones that did.
    """
    result = check_run_completed(finished_run(None, status="error", error="boom"))

    assert result.status == Status.FAIL
    assert result.reason == "boom"
    assert result.score == 0.0


def test_check_run_completed_fails_a_run_paused_for_approval() -> None:
    """A run that paused never answered, so grading it as an answer would score a `None`."""
    result = check_run_completed(finished_run(None, status="paused"))

    assert result.status == Status.FAIL
    assert "paused" in result.reason


def test_check_expected_tool_called_returns_none_when_no_tool_is_expected() -> None:
    """A case with no `expected_tool` has nothing for this check to assert, so it returns `None`."""
    result = check_expected_tool_called(Case(input="hi"), finished_run("ok"))

    assert result is None


def test_check_expected_tool_called_passes_when_the_tool_was_called() -> None:
    """Passes when a tool call in the run matches `case.expected_tool`."""
    case = Case(input="cancel order 123", expected_tool="cancel_order")
    run = finished_run(
        "done", _tool_calls=[ToolCall(name="cancel_order", arguments="{}", output="done")]
    )

    result = check_expected_tool_called(case, run)

    assert result is not None
    assert result.status == Status.PASS


def test_check_expected_tool_called_fails_when_the_tool_was_not_called() -> None:
    """Fails, naming what was actually called, when `expected_tool` never ran."""
    case = Case(input="cancel order 123", expected_tool="cancel_order")
    run = finished_run(
        "done", _tool_calls=[ToolCall(name="search", arguments="{}", output="nothing")]
    )

    result = check_expected_tool_called(case, run)

    assert result is not None
    assert result.status == Status.FAIL
    assert "cancel_order" in result.reason
    assert "search" in result.reason


def test_check_expected_tool_called_grades_the_runs_own_record_not_its_trace_spans() -> None:
    """Graded evidence is `Run._tool_calls`, which no tracing privacy policy has filtered.

    A `"tool"` span's input/output goes through `runa.tracing.observe` first, so reading the
    judge's evidence off the trace let an observability setting decide an eval score. The trace is
    still on the `Run` the report carries, for `runa ui` to display.
    """
    case = Case(input="cancel order 123", expected_tool="cancel_order")
    redacted = Span(
        id="s1",
        trace_id="t1",
        parent_id=None,
        name="cancel_order",
        type="tool",
        start_time=0.0,
        end_time=0.0,
        input=None,
        output=None,
    )
    call = ToolCall(name="cancel_order", arguments='{"order_id": "123"}', output="cancelled")
    run = finished_run(
        "done",
        trace=Trace(id="t1", name="UnderTest", start_time=0.0, end_time=0.0, spans=[redacted]),
        _tool_calls=[call],
    )

    result = check_expected_tool_called(case, run)

    assert result is not None
    assert result.status == Status.PASS
    assert any(span.type == "tool" for span in trace_of(run).spans)
