"""Tests for `runa.eval.tracing.adapter`: `run_agent_for_eval`.

`run_agent_for_eval` goes through `Agent.run`, so these fake that one method rather than the turn
loop under it: what the adapter is responsible for is reading a `Run` into an `AgentRun`, and
giving each case an agent of its own. Whether a `RunaError` becomes `status="error"` in the first
place is `Agent.run`'s contract, covered in `tests/test_agent.py`.
"""

import asyncio
from typing import Any

import pytest

from runa._types import Usage
from runa.agent import Agent
from runa.eval.case import Case
from runa.eval.tracing.adapter import run_agent_for_eval
from runa.run import Run
from runa.tool import ToolCall
from runa.tracing import Span, Trace


class _TestAgent(Agent):
    """A minimal `Agent` for `run_agent_for_eval` to run."""

    name = "UnderTest"


_AGENT = _TestAgent()


def _run(output: Any, trace: Trace | None, **overrides: Any) -> Run:
    return Run(output=output, trace=trace, usage=Usage(), **overrides)


def test_run_agent_for_eval_captures_final_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful run's output and latency are captured, with no error."""
    trace = Trace(id="t1", name="UnderTest", start_time=0.0, end_time=0.0, spans=[])

    async def fake_run(self: Agent, message: Any, *args: Any, **kwargs: Any) -> Run:
        return _run("the answer", trace)

    monkeypatch.setattr(Agent, "run", fake_run)

    run = asyncio.run(run_agent_for_eval(_AGENT, Case(input="hi")))

    assert run.final_output == "the answer"
    assert run.error is None
    assert run.latency >= 0.0


def test_run_agent_for_eval_reads_tool_calls_off_the_run_not_its_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graded evidence is the run's own `ToolCall` record, whatever the trace ended up holding.

    A `"tool"` span's input/output has been through the tracing privacy policy, so reading the
    judge's evidence off it let an observability setting decide an eval score. The trace is still
    carried on the `AgentRun`, for `runa ui` to display.
    """
    redacted_span = Span(
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
    trace = Trace(id="t1", name="UnderTest", start_time=0.0, end_time=0.0, spans=[redacted_span])
    call = ToolCall(name="cancel_order", arguments='{"order_id": "123"}', output="cancelled")

    async def fake_run(self: Agent, message: Any, *args: Any, **kwargs: Any) -> Run:
        return _run("done", trace, _tool_calls=[call])

    monkeypatch.setattr(Agent, "run", fake_run)

    run = asyncio.run(run_agent_for_eval(_AGENT, Case(input="cancel order 123")))

    assert run.tool_calls == [call]
    assert any(span.type == "tool" for span in run.trace.spans)


def test_run_agent_for_eval_captures_a_failed_run_as_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run `Agent.run` reported as `status="error"` becomes an `AgentRun` with `error` set."""

    async def fake_run(self: Agent, message: Any, *args: Any, **kwargs: Any) -> Run:
        return _run(None, None, status="error", error="too many turns")

    monkeypatch.setattr(Agent, "run", fake_run)

    run = asyncio.run(run_agent_for_eval(_AGENT, Case(input="hi")))

    assert run.final_output is None
    assert run.error is not None and "too many turns" in run.error


def test_run_agent_for_eval_falls_back_to_an_empty_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run that never got as far as a `Trace` still yields a readable `AgentRun`."""

    async def fake_run(self: Agent, message: Any, *args: Any, **kwargs: Any) -> Run:
        return _run(None, None, status="error", error="no trace")

    monkeypatch.setattr(Agent, "run", fake_run)

    run = asyncio.run(run_agent_for_eval(_AGENT, Case(input="hi")))

    assert run.trace.spans == []


def test_run_agent_for_eval_runs_each_case_on_its_own_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each case gets a `_fresh()` copy, so one case's history never reaches the next.

    `evaluate_agent` runs up to `concurrency` cases at once over one `Agent`, which is exactly
    what `Agent._exclusive` refuses for session-less runs and what would otherwise let two cases
    overwrite each other's conversation.
    """
    ran_on: list[Agent] = []

    async def fake_run(self: Agent, message: Any, *args: Any, **kwargs: Any) -> Run:
        ran_on.append(self)
        return _run("ok", None)

    monkeypatch.setattr(Agent, "run", fake_run)

    async def _both() -> None:
        await asyncio.gather(
            run_agent_for_eval(_AGENT, Case(input="one")),
            run_agent_for_eval(_AGENT, Case(input="two")),
        )

    asyncio.run(_both())

    assert len(ran_on) == 2
    assert ran_on[0] is not ran_on[1]
    assert _AGENT not in ran_on
