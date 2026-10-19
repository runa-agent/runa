"""Tests for `RunState`'s durable JSON serialization: `to_json`/`to_string`/`from_json`.

These let a paused run survive a process restart -- see `RunState`'s docstring for what's
preserved (everything the run carries, agents by name) and what isn't (guardrail results, trace
spans, a dataclass context's original type).
"""

from dataclasses import fields
from types import SimpleNamespace
from typing import Any

import pytest
from helpers import ScriptedModel, agent_class, text_response, tool_call_response

from runa.agent import Agent
from runa.exceptions import UserError
from runa.run import Run
from runa.run_state import RunState, _RunRecord, _RunStateSchema
from runa.session.ephemeral import EphemeralSession
from runa.tool import tool


def _paused_run() -> tuple[Run, type[Agent]]:
    """Run an agent to a pause on `needs_approval`, returning the paused `Run` and its class.

    The class is what `RunState.from_json` is handed a fresh instance of, as after a restart; it
    declares its scripted model, so every instance answers from the same script.
    """

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        return "done"

    gated = agent_class(
        "TestAgent",
        ScriptedModel([tool_call_response("dangerous", "{}"), text_response("all done")]),
        tools=[dangerous],
    )
    return gated().run_sync("do it"), gated


_SERIALIZED_AS = {"_record": "run", "trace": "trace", "_approvals": "approvals"}
"""Each `RunState` field and the `_RunStateSchema` field it is written to."""

_NOT_SERIALIZED = {
    "guardrail_results": "every guardrail that ran is already a span in the run's trace",
}
"""`RunState` fields deliberately left out of the JSON envelope, and why.

The reason is also written where a reader meets the field -- `_RunStateSchema`'s docstring -- and
this is the list that makes leaving a *second* field out a deliberate act rather than an omission
nothing notices.
"""


def test_every_run_state_field_is_serialized_or_named_as_not() -> None:
    """A field carried through a pause, in memory but not through a restart, by accident."""
    state_fields = {field.name for field in fields(RunState)}

    assert state_fields == set(_SERIALIZED_AS) | set(_NOT_SERIALIZED)
    assert set(_RunStateSchema.model_fields) == set(_SERIALIZED_AS.values()) | {"schema_version"}


def test_every_run_record_field_is_serialized() -> None:
    """The run itself has no field left out: a fact added to it survives a restart unasked."""
    result, _ = _paused_run()

    assert set(result.to_state().to_json()["run"]) == set(_RunRecord.model_fields)


def test_to_json_then_from_json_round_trips_a_paused_run() -> None:
    """A paused run serialized with `to_json` and rebuilt with `from_json` resumes cleanly."""
    result, gated = _paused_run()
    state = result.to_state()
    state.approve(state.pending[0])

    blob = state.to_json()
    fresh = gated()
    restored = RunState.from_json(fresh, blob)

    resumed = fresh.run_sync(restored)
    assert resumed.output == "all done"


def test_to_string_then_from_string_round_trips_the_same_way() -> None:
    """`to_string`/`from_string` behave like `to_json`/`from_json`, just JSON-text-shaped."""
    result, gated = _paused_run()
    state = result.to_state()
    state.approve(state.pending[0])

    blob = state.to_string()
    assert isinstance(blob, str)
    fresh = gated()
    restored = RunState.from_string(fresh, blob)

    resumed = fresh.run_sync(restored)
    assert resumed.output == "all done"


def test_from_json_rejects_an_unknown_schema_version() -> None:
    """An unrecognized `schema_version` raises `UserError`, not a raw Pydantic error."""
    result, gated = _paused_run()
    state = result.to_state()
    state.approve(state.pending[0])
    blob = state.to_json()
    blob["schema_version"] = 999

    with pytest.raises(UserError):
        RunState.from_json(gated(), blob)


def test_from_json_rejects_a_malformed_field() -> None:
    """A field with the wrong shape also surfaces as `UserError`, not a raw Pydantic error."""
    result, gated = _paused_run()
    state = result.to_state()
    state.approve(state.pending[0])
    blob = state.to_json()
    blob["approvals"]["calls"] = "not-a-dict"

    with pytest.raises(UserError):
        RunState.from_json(gated(), blob)


def test_to_json_records_the_current_agent_after_a_handoff_occurred_before_the_pause() -> None:
    """The serialized `agent` reflects a handoff that switched agents before the pause."""

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        return "done"

    target = agent_class(
        "Target",
        ScriptedModel([tool_call_response("dangerous", "{}"), text_response("done")]),
        tools=[dangerous],
    )
    main = agent_class(
        "Main",
        ScriptedModel([tool_call_response("transfer_to_target", "{}")]),
        subagents=[target.handoff],
    )

    result = main().run_sync("please transfer")
    state = result.to_state()
    state.approve(state.pending[0])

    blob = state.to_json()
    assert blob["run"]["agent"] == "Target"

    fresh_main = main()
    restored = RunState.from_json(fresh_main, blob)
    assert restored.agent.name == "Target"

    resumed = fresh_main.run_sync(restored)
    assert resumed.output == "done"


def test_from_json_resolves_the_paused_interruptions_agent_from_its_name() -> None:
    """A pending interruption's `agent` is re-resolved by name against the fresh agent."""
    result, gated = _paused_run()
    state = result.to_state()

    blob = state.to_json()
    fresh_agent = gated()
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

    result, gated = _paused_run()
    blob = result.to_state().to_json()
    fresh_agent: Any = SimpleNamespace(
        name="TestAgent", tools=[], handoffs=[], mcp_servers=[_ExplodingServer()]
    )

    restored = RunState.from_json(fresh_agent, blob)

    assert [item.name for item in restored.pending] == ["dangerous"]


def test_to_json_carries_the_sticky_approval_ledger_and_executed_call_ids() -> None:
    """The sticky approval ledger and replay-guard set survive a JSON round trip."""
    result, gated = _paused_run()
    state = result.to_state()
    state.approve(state.pending[0], always=True)

    blob = state.to_json()
    assert blob["approvals"]["sticky"] == {"dangerous": True}

    fresh = gated()
    restored = RunState.from_json(fresh, blob)
    assert restored._approvals.sticky == {"dangerous": True}

    resumed = fresh.run_sync(restored)
    assert resumed.output == "all done"
    assert "call_1" in restored._approvals.executed


def _paused_after_a_safe_call(**settings: Any) -> tuple[Run, type[Agent]]:
    """Pause on `dangerous` in a message whose `safe` call already ran, for `settings`."""

    @tool
    def safe() -> str:
        """Do something harmless."""
        return "safe done"

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        return "done"

    gated = agent_class(
        "TestAgent",
        ScriptedModel(
            [
                tool_call_response("safe", "{}"),
                tool_call_response("dangerous", "{}", "call_2"),
                text_response("all done"),
            ]
        ),
        tools=[safe, dangerous],
        **settings,
    )
    return gated().run_sync("do it", session=EphemeralSession("s", user_id="u1")), gated


def test_a_restored_run_keeps_its_user_and_the_tool_calls_made_before_the_pause() -> None:
    """Who the run is for and what it already did both come back, with no session to ask."""
    result, gated = _paused_after_a_safe_call()
    state = result.to_state()
    state.approve(state.pending[0])

    fresh = gated()
    restored = RunState.from_json(fresh, state.to_json())

    assert restored.run_context.user_id == "u1"
    resumed = fresh.run_sync(restored, session=EphemeralSession("s", user_id="u1"))
    assert [call.name for call in resumed.tool_calls] == ["safe", "dangerous"]


def test_a_restored_trace_keeps_its_session_grouping() -> None:
    """The trace a resumed run continues is still grouped under the session it started on."""
    result, gated = _paused_after_a_safe_call()

    restored = RunState.from_json(gated(), result.to_state().to_json())

    assert restored.trace.session_id == "s"
    assert restored.trace.metadata == {"group_id": "s"}


def test_a_resumed_run_spends_the_timeout_left_over_from_before_the_pause() -> None:
    """The deadline is the run's, not each call's: time spent running before a pause counts."""
    result, gated = _paused_after_a_safe_call(timeout=5.0)
    state = result.to_state()
    state.approve(state.pending[0])
    blob = state.to_json()
    assert blob["run"]["elapsed"] > 0
    blob["run"]["elapsed"] = 5.0

    resumed = gated().run_sync(RunState.from_json(gated(), blob))

    assert resumed.status == "error"
    assert resumed.error == "run timed out after 5.0s"
