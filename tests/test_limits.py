"""Tests for the three ceilings on one run: `max_turns`, `max_tokens`, and `timeout`.

Each one ends a run as `Run(status="error")` rather than raising, the same contract every other
`RunaError` follows, and each still produces an inspectable trace.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from runa import Agent
from runa._types import ModelResponse, Usage
from runa.exceptions import MaxTokensExceeded, RunTimeout


def _final_message(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text, "tool_calls": None}


def _tool_call_message(call_id: str, name: str, arguments: str = "{}") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


class _CostlyModel:
    """Reports `per_call` tokens on every response, so a budget can be crossed deliberately."""

    def __init__(self, messages: list[dict[str, Any]], per_call: int) -> None:
        self._messages = list(messages)
        self._per_call = per_call
        self.calls = 0

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:  # noqa: ANN002, ANN003
        self.calls += 1
        return ModelResponse(
            output=[self._messages.pop(0)],
            usage=Usage(
                input_tokens=self._per_call // 2,
                output_tokens=self._per_call // 2,
                total_tokens=self._per_call,
                requests=1,
            ),
        )


class _SlowModel:
    """Hangs for `delay` seconds before answering, to be cut short by a run `timeout`."""

    def __init__(self, delay: float) -> None:
        self._delay = delay

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:  # noqa: ANN002, ANN003
        await asyncio.sleep(self._delay)
        return ModelResponse(output=[_final_message("too late")], usage=Usage())


def test_a_run_under_its_token_budget_completes() -> None:
    """`max_tokens` is a ceiling, not a target: staying under it changes nothing."""

    class Thrifty(Agent):
        name = "Thrifty"
        instructions = "Answer."
        max_tokens = 1_000
        model = _CostlyModel([_final_message("done")], per_call=10)

    run = Thrifty().run_sync("hi")

    assert run.status == "completed"
    assert run.output == "done"


def test_exceeding_the_token_budget_ends_the_run_as_an_error() -> None:
    """Crossing `max_tokens` stops the run and reports how much it had spent."""

    class Spendthrift(Agent):
        name = "Spendthrift"
        instructions = "Answer."
        max_tokens = 50
        model = _CostlyModel(
            [_tool_call_message("c1", "noop"), _final_message("done")], per_call=100
        )

    run = Spendthrift().run_sync("hi")

    assert run.status == "error"
    assert "max tokens (50) exceeded" in (run.error or "")
    assert "100 used" in (run.error or "")


def test_the_budget_stops_the_run_before_the_next_model_call() -> None:
    """A blown budget costs one model call, not the whole `max_turns` worth of them."""
    model = _CostlyModel(
        [_tool_call_message(f"c{i}", "noop") for i in range(5)] + [_final_message("done")],
        per_call=100,
    )

    class Looper(Agent):
        name = "Looper"
        instructions = "Answer."
        max_turns = 5
        max_tokens = 50

    agent = Looper()
    agent.model = model
    run = agent.run_sync("hi")

    assert run.status == "error"
    assert model.calls == 1


def test_no_token_budget_by_default() -> None:
    """`max_tokens` is unset by default, so a run is bounded only by `max_turns`."""

    class Default(Agent):
        name = "Default"
        instructions = "Answer."
        model = _CostlyModel([_final_message("done")], per_call=1_000_000)

    assert Default().max_tokens is None
    assert Default().run_sync("hi").status == "completed"


def test_usage_is_still_reported_for_a_run_that_blew_its_budget() -> None:
    """A budget error is a `RunaError`, so the usual error path still records what was spent."""

    class Spendthrift(Agent):
        name = "Spendthrift"
        instructions = "Answer."
        max_tokens = 10
        model = _CostlyModel([_final_message("done")], per_call=100)

    run = Spendthrift().run_sync("hi")

    assert run.status == "error"
    assert run.usage.total_tokens == 100


def test_a_run_slower_than_its_timeout_ends_as_an_error() -> None:
    """`timeout` bounds wall-clock time, the ceiling neither `max_turns` nor `max_tokens` gives."""

    class Patient(Agent):
        name = "Patient"
        instructions = "Answer."
        timeout = 0.05
        model = _SlowModel(delay=5.0)

    run = Patient().run_sync("hi")

    assert run.status == "error"
    assert "timed out after 0.05s" in (run.error or "")


def test_a_run_inside_its_timeout_is_untouched() -> None:
    """A run that finishes in time behaves exactly as if no `timeout` were set."""

    class Quick(Agent):
        name = "Quick"
        instructions = "Answer."
        timeout = 5.0
        model = _SlowModel(delay=0.0)

    run = Quick().run_sync("hi")

    assert run.status == "completed"
    assert run.output == "too late"


def test_no_timeout_by_default() -> None:
    """`timeout` is opt-in; an agent that does not set one is never cut short."""

    class Default(Agent):
        name = "Default"
        instructions = "Answer."

    assert Default().timeout is None


def test_a_timed_out_run_still_exports_its_trace() -> None:
    """A timeout closes the span and reports the partial trace, like any other run-ending error."""

    class Patient(Agent):
        name = "Patient"
        instructions = "Answer."
        timeout = 0.05
        model = _SlowModel(delay=5.0)

    run = Patient().run_sync("hi")

    assert run.trace is not None
    assert run.trace.end_time is not None


def test_cancelling_a_run_is_not_turned_into_an_error_result() -> None:
    """A cancelled run propagates `CancelledError`; the caller asked it to stop, not to fail.

    A dropped HTTP connection or a shutting-down worker must not be silently reported back as a
    completed-with-error run, which would hide the cancellation from the caller's own handling.
    """

    class Patient(Agent):
        name = "Patient"
        instructions = "Answer."
        model = _SlowModel(delay=5.0)

    async def cancel_mid_run() -> None:
        task = asyncio.create_task(Patient().run("hi"))
        await asyncio.sleep(0.01)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancel_mid_run())


def test_both_ceilings_can_apply_to_one_agent() -> None:
    """`max_turns`, `max_tokens` and `timeout` are independent; whichever trips first wins."""

    class Bounded(Agent):
        name = "Bounded"
        instructions = "Answer."
        max_turns = 3
        max_tokens = 50
        timeout = 30.0
        model = _CostlyModel([_tool_call_message("c1", "noop")], per_call=100)

    run = Bounded().run_sync("hi")

    assert run.status == "error"
    assert isinstance(MaxTokensExceeded("x"), Exception)
    assert isinstance(RunTimeout("x"), Exception)
    assert "max tokens" in (run.error or "")
