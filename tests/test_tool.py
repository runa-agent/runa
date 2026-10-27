"""Tests for the `@tool` decorator: its schema and its `guardrails=` wiring."""

from collections.abc import Awaitable
from typing import Any, cast

import pytest
from helpers import run as run_awaitable

from runa import guardrail, tool
from runa._types import RunContext
from runa.exceptions import UserError
from runa.guardrail import BoundGuardrail, GuardrailVerdict, Phase
from runa.tool import FunctionTool


def _run(g: BoundGuardrail, value: Any) -> GuardrailVerdict:
    """Call a wrapped tool guardrail's function directly, awaiting its always-async wrapper."""
    coro = cast(Awaitable[GuardrailVerdict], g.guardrail_function(value))
    return run_awaitable(coro)


@guardrail
def block_args(args: dict) -> bool:
    """Trip when the tool is called with any arguments."""
    return bool(args)


@guardrail
def block_long(output: str) -> bool:
    """Trip when the output is too long."""
    return len(output) > 100


@tool
def bare() -> str:
    """Return a constant string, with no guardrails."""
    return "ok"


def test_bare_tool_has_no_guardrails_or_approval() -> None:
    """`@tool` with no args produces a plain `FunctionTool`."""
    assert isinstance(bare, FunctionTool)
    assert bare.guardrails[Phase.TOOL_INPUT] == []
    assert bare.guardrails[Phase.TOOL_OUTPUT] == []
    assert not run_awaitable(bare.needs_approval(RunContext(), {}, "call_1"))


def test_guardrail_list_splits_by_binding() -> None:
    """`.input`/`.output`-bound entries land in their matching tool phase."""

    @tool(guardrails=[block_args.input, block_long.output])
    def now() -> str:
        """Return a constant string."""
        return "now"

    assert [g.get_name() for g in now.guardrails[Phase.TOOL_INPUT]] == ["block_args"]
    assert [g.get_name() for g in now.guardrails[Phase.TOOL_OUTPUT]] == ["block_long"]


def test_bare_guardrail_wires_both_sides() -> None:
    """A bare `@guardrail` predicate in the list is wired as both input and output."""

    @tool(guardrails=[block_args])
    def now() -> str:
        """Return a constant string."""
        return "now"

    assert [g.get_name() for g in now.guardrails[Phase.TOOL_INPUT]] == ["block_args"]
    assert [g.get_name() for g in now.guardrails[Phase.TOOL_OUTPUT]] == ["block_args"]


def test_invalid_guardrail_entry_raises() -> None:
    """A `guardrail` entry not bound via `.input`/`.output` is rejected."""
    with pytest.raises(UserError, match="guardrails entries must be"):

        @tool(guardrails=cast(Any, [lambda value: False]))
        def now() -> str:
            """Return a constant string."""
            return "now"


def test_needs_approval_true_asks_for_every_call() -> None:
    """`needs_approval=True` reaches `FunctionTool` as a predicate answering yes."""

    @tool(needs_approval=True)
    def now() -> str:
        """Return a constant string."""
        return "now"

    assert run_awaitable(now.needs_approval(RunContext(), {}, "call_1"))


def test_tool_input_guardrail_sees_the_arguments_dict() -> None:
    """An `.input`-bound predicate, reused in a tool context, sees the call's arguments dict."""

    @tool(guardrails=[block_args.input])
    def now() -> str:
        """Return a constant string."""
        return "now"

    (bound,) = now.guardrails[Phase.TOOL_INPUT]
    assert _run(bound, {"x": 1}).tripped
    assert not _run(bound, {}).tripped


def test_tool_output_guardrail_sees_return_value() -> None:
    """An `.output`-bound predicate, reused in a tool context, sees the tool's return value."""

    @tool(guardrails=[block_long.output])
    def now() -> str:
        """Return a constant string."""
        return "now"

    (bound,) = now.guardrails[Phase.TOOL_OUTPUT]
    assert _run(bound, "x" * 101).tripped
    assert not _run(bound, "short").tripped


@tool
def get_weather(city: str, units: str = "c") -> str:
    """Return the current weather for a city.

    city: city name
    Note: units default to Celsius.
    """
    return city


def test_param_lines_describe_parameters_not_the_tool() -> None:
    """A `<param>: <text>` line moves to that parameter's schema; other lines stay."""
    assert get_weather.description == (
        "Return the current weather for a city.\n\nNote: units default to Celsius."
    )
    properties = get_weather.params_json_schema["properties"]
    assert properties["city"] == {"type": "string", "description": "city name"}
    assert properties["units"] == {"type": "string"}
