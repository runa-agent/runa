"""Tests for `agent_as_tool` and the shared agent-graph walk.

`agent_as_tool`'s sharing with its caller (sticky approvals, usage, `context`), and the walk
every module needing "reachable from here" shares (`reachable`/`subagent_edges`).
"""

from typing import Any

from helpers import run as run_awaitable

from runa._governance import _Governance
from runa._types import ModelResponse, RunContext, Usage
from runa.agent import Agent
from runa.handoff import agent_as_tool, reachable, subagent_edges
from runa.tool import tool


def _final_message(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text, "tool_calls": None}


def _tool_call_message(name: str, arguments: str, call_id: str = "call_1") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


class _ScriptedModel:
    """A `Model` stand-in returning pre-scripted messages, run through the real `Runner`."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = list(messages)

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:  # noqa: ANN002, ANN003
        return ModelResponse(
            output=[self._messages.pop(0)],
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, requests=1),
        )


def test_agent_as_tool_shares_governance_so_sticky_approvals_apply_to_the_delegate() -> None:
    """A sticky `always=True` approval already on the caller's ledger lets a delegate's tool run.

    Without sharing, the nested run would start a brand-new `_Governance` with an empty approval
    ledger, so its `dangerous` call would pause instead and the tool would raise `DelegatePaused`
    rather than return. Sharing is what makes `result == "delegate done"` possible here.
    """

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Needs approval."""
        return "handled"

    class Delegate(Agent):
        name = "Delegate"
        tools = [dangerous]
        model = _ScriptedModel(
            [_tool_call_message("dangerous", "{}"), _final_message("delegate done")]
        )

    tool_fn = agent_as_tool(Delegate(), None, None)
    governance = _Governance()
    governance.approval_ledger.record("dangerous", approved=True)

    result = run_awaitable(
        tool_fn.on_invoke_tool(RunContext(), governance, '{"input": "go"}', "call_1")
    )

    assert result == "delegate done"


def test_agent_as_tool_merges_the_delegates_usage_into_the_callers_context() -> None:
    """The delegate's token usage is added onto the caller's `RunContext.usage`."""

    class Delegate(Agent):
        name = "Delegate"
        model = _ScriptedModel([_final_message("delegate done")])

    tool_fn = agent_as_tool(Delegate(), None, None)
    ctx = RunContext()
    assert ctx.usage.total_tokens == 0

    run_awaitable(tool_fn.on_invoke_tool(ctx, _Governance(), '{"input": "go"}', "call_1"))

    assert ctx.usage.total_tokens == 2


def test_agent_as_tool_shares_the_same_context_object_with_the_delegate() -> None:
    """The delegate's tools see the exact same `context` object as the caller's, not a copy."""
    seen: list[Any] = []

    @tool
    def record(ctx: RunContext) -> str:
        """Record the context object it was called with."""
        seen.append(ctx.context)
        return "recorded"

    class Delegate(Agent):
        name = "Delegate"
        tools = [record]
        model = _ScriptedModel(
            [_tool_call_message("record", "{}"), _final_message("delegate done")]
        )

    marker = object()
    tool_fn = agent_as_tool(Delegate(), None, None)
    ctx = RunContext(context=marker)

    run_awaitable(tool_fn.on_invoke_tool(ctx, _Governance(), '{"input": "go"}', "call_1"))

    assert seen == [marker]


def test_reachable_names_every_agent_once_even_through_a_cycle() -> None:
    """Two agents handing off to each other terminate the walk, each named exactly once."""

    class First(Agent):
        name = "first"
        instructions = "First."

    class Second(Agent):
        name = "second"
        instructions = "Second."

    first, second = First(), Second()
    first.handoffs.append(second)
    second.handoffs.append(first)  # a back-edge the walk must not follow forever

    assert [agent.name for agent in reachable(first)] == ["first", "second"]


def test_subagent_edges_reports_an_auto_subagent_under_both_modes() -> None:
    """A bare (auto) subagent is wired up as a handoff *and* a delegate: an edge of each."""

    class Helper(Agent):
        name = "helper"
        instructions = "Helper."

    class Lead(Agent):
        name = "lead"
        instructions = "Lead."
        subagents = [Helper]

    edges = list(subagent_edges(Lead()))

    assert [mode for mode, _ in edges] == ["handoff", "delegate"]
    assert {agent.name for _, agent in edges} == {"helper"}
