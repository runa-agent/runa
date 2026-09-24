"""Tests for the `@tool` decorator's `guardrails=` wiring."""

from collections.abc import Awaitable
from typing import Any, cast

import pytest
from helpers import run as run_awaitable

from runa import guardrail, tool
from runa.guardrail import ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolOutputGuardrail
from runa.tool import FunctionTool


def _run(
    g: ToolInputGuardrail[Any] | ToolOutputGuardrail[Any], data: Any
) -> ToolGuardrailFunctionOutput:
    """Call a wrapped tool guardrail's function directly, awaiting its always-async wrapper."""
    coro = cast(Awaitable[ToolGuardrailFunctionOutput], g.guardrail_function(data))
    return run_awaitable(coro)


def _tripped(output: ToolGuardrailFunctionOutput) -> bool:
    """Whether a `ToolGuardrailFunctionOutput` halts execution rather than allowing it."""
    return output.behavior["type"] == "raise_exception"


class _Data:
    """A stand-in for `ToolInputGuardrailData`/`ToolOutputGuardrailData`."""

    def __init__(self, tool_arguments: str, output: Any = None) -> None:
        self.context = cast(Any, type("Ctx", (), {"tool_arguments": tool_arguments})())
        self.output = output


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
    assert bare.tool_input_guardrails is None
    assert bare.tool_output_guardrails is None
    assert bare.needs_approval is False


def test_guardrail_list_splits_by_binding() -> None:
    """`.input`/`.output`-bound entries land in their matching SDK tool-guardrail list."""

    @tool(guardrails=[block_args.input, block_long.output])
    def now() -> str:
        """Return a constant string."""
        return "now"

    assert [g.get_name() for g in now.tool_input_guardrails or []] == ["block_args"]
    assert [g.get_name() for g in now.tool_output_guardrails or []] == ["block_long"]


def test_bare_guardrail_wires_both_sides() -> None:
    """A bare `@guardrail` predicate in the list is wired as both input and output."""

    @tool(guardrails=[block_args])
    def now() -> str:
        """Return a constant string."""
        return "now"

    assert [g.get_name() for g in now.tool_input_guardrails or []] == ["block_args"]
    assert [g.get_name() for g in now.tool_output_guardrails or []] == ["block_args"]


def test_dict_guardrails_wire_by_key() -> None:
    """A `{"input": [...], "output": [...]}` dict binds each bare entry by its key."""

    @tool(guardrails={"input": [block_args], "output": [block_long]})
    def now() -> str:
        """Return a constant string."""
        return "now"

    assert [g.get_name() for g in now.tool_input_guardrails or []] == ["block_args"]
    assert [g.get_name() for g in now.tool_output_guardrails or []] == ["block_long"]


def test_invalid_guardrail_entry_raises() -> None:
    """A `guardrail` entry not bound via `.input`/`.output` is rejected."""
    with pytest.raises(TypeError, match="guardrail entries must be"):

        @tool(guardrails=cast(Any, [lambda value: False]))
        def now() -> str:
            """Return a constant string."""
            return "now"


def test_needs_approval_passes_through_natively() -> None:
    """`needs_approval=True` forwards straight through to `FunctionTool`, unmodified."""

    @tool(needs_approval=True)
    def now() -> str:
        """Return a constant string."""
        return "now"

    assert now.needs_approval is True


def test_tool_input_guardrail_sees_parsed_arguments() -> None:
    """An `.input`-bound predicate, reused in a tool context, sees parsed call arguments."""

    @tool(guardrails=[block_args.input])
    def now() -> str:
        """Return a constant string."""
        return "now"

    (bound,) = now.tool_input_guardrails or []
    assert _tripped(_run(bound, _Data(tool_arguments='{"x": 1}')))
    assert not _tripped(_run(bound, _Data(tool_arguments="{}")))


def test_tool_output_guardrail_sees_return_value() -> None:
    """An `.output`-bound predicate, reused in a tool context, sees the tool's return value."""

    @tool(guardrails=[block_long.output])
    def now() -> str:
        """Return a constant string."""
        return "now"

    (bound,) = now.tool_output_guardrails or []
    assert _tripped(_run(bound, _Data(tool_arguments="{}", output="x" * 101)))
    assert not _tripped(_run(bound, _Data(tool_arguments="{}", output="short")))
