"""Tests for `agent_as_tool`'s context-forking: sticky approvals, usage, and `context` sharing."""

from typing import Any

from helpers import run as run_awaitable

from runa._types import ModelResponse, RunContextWrapper, Usage
from runa.agent import Agent
from runa.handoff import agent_as_tool
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


def test_agent_as_tool_forks_context_so_sticky_approvals_apply_to_the_delegate() -> None:
    """A sticky `always=True` approval already on the context lets a delegate's gated tool run.

    Without forking, `on_invoke_tool` builds the delegate a brand-new `RunContextWrapper` with an
    empty approval ledger, so its `dangerous` call would pause instead -- and since `Run.status`
    has no "paused" state, a paused delegate run's `output` comes back as `None` rather than the
    tool's real result. Forking is what makes `result == "delegate done"` possible here.
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
    ctx = RunContextWrapper(context=None)
    ctx.approval_ledger["dangerous"] = True

    result = run_awaitable(tool_fn.on_invoke_tool(ctx, '{"input": "go"}', "call_1"))

    assert result == "delegate done"


def test_agent_as_tool_merges_the_delegates_usage_into_the_callers_context() -> None:
    """The delegate's token usage is added onto the caller's `context_wrapper.usage`."""

    class Delegate(Agent):
        name = "Delegate"
        model = _ScriptedModel([_final_message("delegate done")])

    tool_fn = agent_as_tool(Delegate(), None, None)
    ctx = RunContextWrapper(context=None)
    assert ctx.usage.total_tokens == 0

    run_awaitable(tool_fn.on_invoke_tool(ctx, '{"input": "go"}', "call_1"))

    assert ctx.usage.total_tokens == 2


def test_agent_as_tool_shares_the_same_context_object_with_the_delegate() -> None:
    """The delegate's tools see the exact same `context` object as the caller's, not a copy."""
    seen: list[Any] = []

    @tool
    def record(ctx: RunContextWrapper) -> str:
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
    ctx = RunContextWrapper(context=marker)

    run_awaitable(tool_fn.on_invoke_tool(ctx, '{"input": "go"}', "call_1"))

    assert seen == [marker]
