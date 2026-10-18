"""Tests for `RunState`'s durable JSON serialization: `to_json`/`to_string`/`from_json`.

These let a paused run survive a process restart -- see `RunState`'s docstring for what's
preserved (agent identity by name, the sticky approval ledger, usage) and what isn't
(guardrail results, trace spans, a dataclass context's original type).
"""

from dataclasses import fields
from types import SimpleNamespace
from typing import Any

import pytest
from helpers import ScriptedModel, agent_class, text_response, tool_call_response

from runa.agent import Agent
from runa.exceptions import UserError
from runa.run import Run
from runa.run_state import RunState, _RunStateSchema
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


_SERIALIZED_AS = {
    "agent": ("agent_name",),
    "transcript": ("transcript",),
    "ready_results": ("ready_results",),
    "pending": ("pending",),
    "approvals": ("approvals",),
    "rejection_messages": ("rejection_messages",),
    "run_context": ("context", "usage"),
    "_governance": (
        "approval_ledger",
        "approval_ledger_messages",
        "executed_call_ids",
        "delegates",
        "delegate_owners",
    ),
    "trace": ("trace_id", "trace_name", "trace_start_time"),
}
"""Each `RunState` field and the `_RunStateSchema` field(s) `_to_schema` writes it to.

Mostly one to one, under the same name. The three that aren't are the three holding an object
rather than data: the `RunContext` flattens into its `context` and `usage`, the `_Governance` into
the contents of two of the protocol objects it holds (`ApprovalLedger`, `PausedDelegates`), and
a `Trace` into the three fields that identify it, its spans being exported separately.
"""

_NOT_SERIALIZED = {
    "guardrail_results": "every guardrail that ran is already a span in the run's trace",
}
"""`RunState` fields deliberately left out of the JSON envelope, and why.

The reason is also written where a reader meets the field -- `_RunStateSchema`'s docstring -- and
this is the list that makes leaving a *second* field out a deliberate act rather than an omission
nothing notices. (`Transcript.tool_calls` is left out one level down; see `test_transcript.py`.)
"""


def test_every_run_state_field_is_serialized_or_named_as_not() -> None:
    """A field carried through a pause, in memory but not through a restart, by accident.

    `_to_schema` and `_from_schema` map `RunState` to its JSON envelope field by field, and that
    is the drift this pins down from the dataclass's side: a field added to `RunState` is either
    mapped to the schema or named above as one that isn't. The other side is covered twice over
    -- `_RunStateSchema` requires every field, so pyright rejects a `_to_schema` that skips one,
    and the next test asserts no schema field is written from nothing.
    """
    state_fields = {field.name for field in fields(RunState)}

    assert state_fields == set(_SERIALIZED_AS) | set(_NOT_SERIALIZED)


def test_every_schema_field_is_written_from_a_run_state_field() -> None:
    """The envelope's side: a field read back by `_from_schema` that nothing ever writes.

    `schema_version` is the exception, and the only one: it describes the envelope itself rather
    than anything the paused run holds.
    """
    written = {name for names in _SERIALIZED_AS.values() for name in names}

    assert set(_RunStateSchema.model_fields) == written | {"schema_version"}


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
    blob["approvals"] = "not-a-dict"

    with pytest.raises(UserError):
        RunState.from_json(gated(), blob)


def test_to_json_records_the_current_agent_after_a_handoff_occurred_before_the_pause() -> None:
    """The serialized `agent_name` reflects a handoff that switched agents before the pause."""

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
    assert blob["agent_name"] == "Target"

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
    assert blob["approval_ledger"] == {"dangerous": True}

    fresh = gated()
    restored = RunState.from_json(fresh, blob)
    assert restored._governance.approval_ledger.sticky == {"dangerous": True}

    resumed = fresh.run_sync(restored)
    assert resumed.output == "all done"
    assert "call_1" in restored._governance.approval_ledger.executed
