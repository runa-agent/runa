"""Tests for the `@guardrail` predicate wiring in `Agent`."""

from collections.abc import Awaitable
from typing import Any, cast

import pytest
from helpers import run as run_awaitable

from runa import Agent, guardrail
from runa.guardrail import GuardrailFunctionOutput, InputGuardrail, OutputGuardrail

_CTX = cast(Any, None)
_AGENT = cast(Any, None)


def _run(g: InputGuardrail[Any] | OutputGuardrail[Any], value: Any) -> GuardrailFunctionOutput:
    """Call a wrapped guardrail's function directly, awaiting its always-async wrapper."""
    coro = cast(Awaitable[GuardrailFunctionOutput], g.guardrail_function(_CTX, _AGENT, value))
    return run_awaitable(coro)


@guardrail
def block_empty(input: str) -> bool:
    """Trip when the input is empty."""
    return not input


@guardrail
def block_long(output: str) -> bool:
    """Trip when the output is too long."""
    return len(output) > 100


def test_guardrails_split_into_input_and_output() -> None:
    """A `guardrails` list is sorted into `input_guardrails`/`output_guardrails` by binding."""
    bound_input, bound_output = block_empty.input, block_long.output

    class Support(Agent):
        name = "Support"
        instructions = "support"
        guardrails = [bound_input, bound_output]

    agent = Support()

    assert agent.input_guardrails == [bound_input]
    assert agent.output_guardrails == [bound_output]


def test_no_guardrails_leaves_both_lists_empty() -> None:
    """An agent with no `guardrails` attribute wires up cleanly."""

    class Support(Agent):
        name = "Support"
        instructions = "support"

    agent = Support()

    assert agent.input_guardrails == []
    assert agent.output_guardrails == []


def test_invalid_guardrail_entry_raises() -> None:
    """A `guardrails` entry not bound via `.input`/`.output` is rejected."""

    class Support(Agent):
        name = "Support"
        instructions = "support"
        guardrails = [lambda value: False]

    with pytest.raises(TypeError, match="guardrails entries must be"):
        Support()


def test_bare_guardrail_wires_both_input_and_output() -> None:
    """A `@guardrail` predicate listed without `.input`/`.output` is wired as both."""

    class Support(Agent):
        name = "Support"
        instructions = "support"
        guardrails = [block_empty]

    agent = Support()

    assert [g.name for g in agent.input_guardrails] == ["block_empty"]
    assert [g.name for g in agent.output_guardrails] == ["block_empty"]


def test_dict_guardrails_wire_by_key() -> None:
    """A `{"input": [...], "output": [...]}` dict binds each bare entry by its key."""

    class Support(Agent):
        name = "Support"
        instructions = "support"
        guardrails = {"input": [block_empty], "output": [block_long]}

    agent = Support()

    assert [g.name for g in agent.input_guardrails] == ["block_empty"]
    assert [g.name for g in agent.output_guardrails] == ["block_long"]


def test_input_predicate_reduces_item_list_to_latest_text() -> None:
    """`.input` hands the predicate plain text, even when the SDK passes an item list."""
    turn_input = [{"role": "user", "content": ""}]

    result = _run(block_empty.input, turn_input)

    assert result.tripwire_triggered is True
    assert result.output_info == "Trip when the input is empty."


def test_output_predicate_false_does_not_trip() -> None:
    """A predicate returning `False` leaves the guardrail untripped."""
    result = _run(block_long.output, "short")

    assert result.tripwire_triggered is False


def test_i_and_o_are_shorthand_for_input_and_output() -> None:
    """`.i`/`.o` bind exactly like `.input`/`.output`, same guardrail, same name, same side."""

    class Support(Agent):
        name = "Support"
        instructions = "support"
        guardrails = [block_empty.i, block_long.o]

    agent = Support()

    assert [g.name for g in agent.input_guardrails] == ["block_empty"]
    assert [g.name for g in agent.output_guardrails] == ["block_long"]


def test_async_predicate_is_awaited() -> None:
    """An async predicate is awaited and still produces a `GuardrailFunctionOutput`."""

    @guardrail
    async def block_async(input: str) -> bool:
        """Trip always, asynchronously."""
        return True

    result = _run(block_async.input, "x")

    assert result.tripwire_triggered is True
