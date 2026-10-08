"""Tests for `RunState`'s durable JSON serialization: `to_json`/`to_string`/`from_json`.

These let a paused run survive a process restart -- see `RunState`'s docstring for what's
preserved (agent identity by name, the sticky approval ledger, usage) and what isn't
(guardrail results, trace spans, a dataclass context's original type).
"""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from helpers import context_of

from runa._types import ModelResponse, Usage
from runa.exceptions import UserError
from runa.handoff import Handoff
from runa.run_internal.agent_shape import AgentShape
from runa.run_internal.run_config import RunConfig
from runa.run_internal.run_loop import _run_async
from runa.run_state import RunState
from runa.tool import tool


def _agent(**overrides: Any) -> Any:
    """A stand-in agent for the loop: an `AgentShape` with everything else defaulted."""
    return AgentShape(**{"name": "TestAgent", "instructions": "be helpful", **overrides})


def _text_response(text: str) -> ModelResponse:
    return ModelResponse(
        output=[{"role": "assistant", "content": text, "tool_calls": None}],
        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, requests=1),
    )


def _tool_call_response(name: str, arguments: str, call_id: str = "call_1") -> ModelResponse:
    return ModelResponse(
        output=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            }
        ],
        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, requests=1),
    )


class _ScriptedModel:
    """A `Model` stand-in that returns pre-scripted `ModelResponse`s in order."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:  # noqa: ANN002, ANN003
        return self._responses.pop(0)

    async def stream_response(self, *args: Any, **kwargs: Any):  # noqa: ANN002, ANN003
        raise NotImplementedError


def _run_config() -> RunConfig:
    return RunConfig(workflow_name="TestAgent")


def _paused_result_and_agent() -> tuple[Any, Any]:
    """Run an agent to a pause on `needs_approval`, returning `(result, agent)`."""

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        return "done"

    agent = _agent(
        tools=[dangerous],
        model=_ScriptedModel([_tool_call_response("dangerous", "{}"), _text_response("all done")]),
    )
    result = asyncio.run(_run_async(agent, "do it", run_config=_run_config()))
    return result, agent


def _fresh_agent_like(agent: Any) -> Any:
    """A structurally equivalent but distinct agent object, standing in for a fresh instance.

    Carries `model` over too -- a real fresh instance of the same `Agent` subclass would have
    the same `model` config; `from_json` re-resolving `.agent` by name is the behavior under
    test, not needing to independently reconstruct the model too.
    """
    return _agent(
        name=agent.name, tools=list(agent.tools), handoffs=dict(agent.handoffs), model=agent.model
    )


def test_to_json_then_from_json_round_trips_a_paused_run() -> None:
    """A paused run serialized with `to_json` and rebuilt with `from_json` resumes cleanly."""
    result, agent = _paused_result_and_agent()
    state = result.to_state()
    state.approve(state.pending[0])

    blob = state.to_json()
    restored = RunState.from_json(_fresh_agent_like(agent), blob)

    resumed = asyncio.run(_run_async(agent, restored, run_config=_run_config()))
    assert resumed.output == "all done"


def test_to_string_then_from_string_round_trips_the_same_way() -> None:
    """`to_string`/`from_string` behave like `to_json`/`from_json`, just JSON-text-shaped."""
    result, agent = _paused_result_and_agent()
    state = result.to_state()
    state.approve(state.pending[0])

    blob = state.to_string()
    assert isinstance(blob, str)
    restored = RunState.from_string(_fresh_agent_like(agent), blob)

    resumed = asyncio.run(_run_async(agent, restored, run_config=_run_config()))
    assert resumed.output == "all done"


def test_from_json_rejects_an_unknown_schema_version() -> None:
    """An unrecognized `schema_version` raises `UserError`, not a raw Pydantic error."""
    result, agent = _paused_result_and_agent()
    state = result.to_state()
    state.approve(state.pending[0])
    blob = state.to_json()
    blob["schema_version"] = 999

    with pytest.raises(UserError):
        RunState.from_json(_fresh_agent_like(agent), blob)


def test_from_json_rejects_a_malformed_field() -> None:
    """A field with the wrong shape also surfaces as `UserError`, not a raw Pydantic error."""
    result, agent = _paused_result_and_agent()
    state = result.to_state()
    state.approve(state.pending[0])
    blob = state.to_json()
    blob["approvals"] = "not-a-dict"

    with pytest.raises(UserError):
        RunState.from_json(_fresh_agent_like(agent), blob)


def test_to_json_records_the_current_agent_after_a_handoff_occurred_before_the_pause() -> None:
    """The serialized `agent_name` reflects a handoff that switched agents before the pause."""

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        return "done"

    target = _agent(
        name="Target",
        tools=[dangerous],
        model=_ScriptedModel([_tool_call_response("dangerous", "{}"), _text_response("done")]),
    )
    handoff = Handoff.from_agent(target)
    main = _agent(
        name="Main",
        handoffs=[handoff],
        model=_ScriptedModel([_tool_call_response(handoff.tool_name, "{}")]),
    )

    result = asyncio.run(_run_async(main, "please transfer", run_config=_run_config()))
    state = result.to_state()
    state.approve(state.pending[0])

    blob = state.to_json()
    assert blob["agent_name"] == "Target"

    fresh_main = _agent(name="Main", handoffs=[Handoff.from_agent(target)])
    restored = RunState.from_json(fresh_main, blob)
    assert restored.agent.name == "Target"

    resumed = asyncio.run(_run_async(target, restored, run_config=_run_config()))
    assert resumed.output == "done"


def test_from_json_resolves_the_paused_interruptions_agent_from_its_name() -> None:
    """A pending interruption's `agent` is re-resolved by name against the fresh agent."""
    result, agent = _paused_result_and_agent()
    state = result.to_state()

    blob = state.to_json()
    fresh_agent = _fresh_agent_like(agent)
    restored = RunState.from_json(fresh_agent, blob)

    assert restored.pending[0].agent is fresh_agent
    assert restored.pending[0].name == "dangerous"


def test_from_json_never_lists_an_agents_mcp_server_tools() -> None:
    """Reading a blob resolves no tools, so an agent's MCP servers are left alone entirely.

    The tool a pending call names is looked up by the resumed run, off the shape it builds
    anyway -- which is what keeps `from_json` a synchronous, I/O-free read even when the paused
    call was an MCP server's tool and the fresh agent declares nothing locally.
    """

    class _ExplodingServer:
        async def list_tools(self) -> list[Any]:
            raise AssertionError("from_json must not connect to an MCP server")

    result, agent = _paused_result_and_agent()
    blob = result.to_state().to_json()
    fresh_agent = SimpleNamespace(
        name=agent.name, tools=[], handoffs={}, mcp_servers=[_ExplodingServer()]
    )

    restored = RunState.from_json(fresh_agent, blob)

    assert [item.name for item in restored.pending] == ["dangerous"]


def test_to_json_carries_the_sticky_approval_ledger_and_executed_call_ids() -> None:
    """The sticky approval ledger and replay-guard set survive a JSON round trip."""
    result, agent = _paused_result_and_agent()
    state = result.to_state()
    state.approve(state.pending[0], always=True)

    blob = state.to_json()
    assert blob["approval_ledger"] == {"dangerous": True}

    restored = RunState.from_json(_fresh_agent_like(agent), blob)
    assert restored.context_wrapper.approval_ledger.sticky == {"dangerous": True}

    resumed = asyncio.run(_run_async(agent, restored, run_config=_run_config()))
    assert resumed.output == "all done"
    assert "call_1" in context_of(resumed).approval_ledger.executed
