"""Tests for `RunState`'s durability: a pause kept in `runa.db`, loaded back, resumed once.

A paused run is kept the moment it pauses, under its `run_id`; `RunState.load` is how another
process gets it back -- see `RunState`'s docstring for what's preserved (everything the run
carries, agents by name) and what isn't (guardrail results, trace spans, a dataclass context's
original type).
"""

import asyncio
from dataclasses import fields
from types import SimpleNamespace
from typing import Any

import pytest
from helpers import ScriptedModel, agent_class, text_response, tool_call_response

from runa import _loop, db
from runa.agent import Agent
from runa.exceptions import UserError
from runa.lifecycle import Hooks
from runa.run import Run
from runa.run_internal.record import Pause, PauseSchema, RunRecord, RunTree, resuming
from runa.run_state import RunState
from runa.tool import tool


def _paused_run() -> tuple[Run, type[Agent]]:
    """Run an agent to a pause on `needs_approval`, returning the paused `Run` and its class.

    The class is what `RunState.load` is handed a fresh instance of, as after a restart; it
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


def _stored(run: Run, agent: str = "TestAgent") -> dict[str, Any]:
    """The row `run`'s pause is kept as."""
    assert run.run_id is not None
    blob = _loop.run_sync(db.paused_runs().load(agent, run.run_id))
    assert blob is not None
    return blob


def _store(blob: dict[str, Any], agent: str = "TestAgent") -> str:
    """Keep `blob` as a pause of its own, as a row edited by hand would be, and return its id."""
    return _loop.run_sync(db.paused_runs().save(agent, blob))


def _load(agent: Agent, run_id: str | None) -> RunState:
    """`RunState.load_sync`, for a pause the test knows is there."""
    assert run_id is not None
    state = RunState.load_sync(agent, run_id)
    assert state is not None
    return state


_SERIALIZED_AS = {"record": "run", "tree": "tree"}
"""Each `Pause` field and the `PauseSchema` field it is written to."""

_NOT_SERIALIZED = {
    "run_id": "the row's key, not its content",
    "session": "a live object; the record carries its id",
}
"""`Pause` fields deliberately left out of the stored blob, and why.

The reason is also written where a reader meets the field -- `PauseSchema`'s docstring -- and
this is the list that makes leaving a *second* field out a deliberate act rather than an omission
nothing notices.
"""


def test_every_pause_field_is_serialized_or_named_as_not() -> None:
    """A field carried through a pause, in memory but not through a restart, by accident."""
    state_fields = {field.name for field in fields(Pause)}

    assert state_fields == set(_SERIALIZED_AS) | set(_NOT_SERIALIZED)
    assert set(PauseSchema.model_fields) == set(_SERIALIZED_AS.values()) | {"schema_version"}


def test_every_run_record_field_is_serialized() -> None:
    """The run itself has no field left out: a fact added to it survives a restart unasked."""
    result, _ = _paused_run()

    assert set(_stored(result)["run"]) == set(RunRecord.model_fields)


def test_every_run_tree_field_but_the_audit_trail_is_serialized() -> None:
    """What the tree shares survives a restart, but for guardrails already kept as spans."""
    result, _ = _paused_run()

    assert set(_stored(result)["tree"]) == set(RunTree.model_fields) - {"guardrail_results"}


def test_a_paused_run_is_kept_under_its_run_id() -> None:
    """Nothing to save by hand: the pause is in `runa.db` by the time the `Run` comes back."""
    result, _ = _paused_run()

    assert result.run_id == result.to_state().run_id
    assert result.to_payload()["run_id"] == result.run_id
    assert _stored(result)["run"]["pending"][0]["name"] == "dangerous"


def test_a_completed_run_has_no_run_id() -> None:
    """Only a pause is kept, so only a pause has a handle."""
    result, gated = _paused_run()
    state = result.to_state()
    state.approve(state.pending[0])

    assert gated().run_sync(state).run_id is None


def test_a_loaded_pause_resumes_in_a_fresh_agent() -> None:
    """Another process loads the pause by its `run_id`, decides it, and resumes it."""
    result, gated = _paused_run()

    fresh = gated()
    restored = _load(fresh, result.run_id)
    restored.approve(restored.pending[0])

    resumed = fresh.run_sync(restored)
    assert resumed.output == "all done"


def test_load_is_awaitable_for_an_async_caller() -> None:
    """`load` is the async spelling `load_sync` wraps, for `await agent.run(...)`."""
    result, gated = _paused_run()
    fresh = gated()
    run_id = result.run_id
    assert run_id is not None

    async def resume() -> Run:
        restored = await RunState.load(fresh, run_id)
        assert restored is not None
        restored.approve(restored.pending[0])
        return await fresh.run(restored)

    assert asyncio.run(resume()).output == "all done"


def test_a_pause_resumes_once_across_every_copy_of_it() -> None:
    """Two loads of one pause, as two processes would make: the second resume runs nothing."""
    ran: list[str] = []

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        ran.append("dangerous")
        return "done"

    gated = agent_class(
        "TestAgent",
        ScriptedModel([tool_call_response("dangerous", "{}"), text_response("all done")]),
        tools=[dangerous],
    )
    result = gated().run_sync("do it")
    first, second = gated(), gated()
    first_state, second_state = _load(first, result.run_id), _load(second, result.run_id)
    for state in (first_state, second_state):
        state.approve(state.pending[0])

    assert first.run_sync(first_state).status == "completed"
    replayed = second.run_sync(second_state)

    assert replayed.status == "error"
    assert replayed.error == f"paused run {result.run_id!r} was already resumed"
    assert ran == ["dangerous"]


def test_a_resume_that_fails_before_running_a_call_leaves_the_pause_to_resume_again() -> None:
    """A failed resume that ran nothing it decided hasn't spent the human's decision."""
    ran: list[str] = []

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        ran.append("dangerous")
        return "done"

    class _BrokenHooks(Hooks[Any]):
        async def on_agent_start(self, context: Any, agent: Any) -> None:
            raise RuntimeError("the approval log is down")

    gated = agent_class(
        "TestAgent",
        ScriptedModel([tool_call_response("dangerous", "{}"), text_response("all done")]),
        tools=[dangerous],
    )
    result = gated().run_sync("do it")
    state = _load(gated(), result.run_id)
    state.approve(state.pending[0])

    failed = gated().run_sync(state, hooks=_BrokenHooks())
    retried = _load(gated(), result.run_id)
    retried.approve(retried.pending[0])

    assert failed.status == "error"
    assert ran == []
    assert gated().run_sync(retried).output == "all done"
    assert ran == ["dangerous"]


def test_a_resumed_pause_is_no_longer_found() -> None:
    """A pause already resumed loads as `None`, like one that never existed."""
    result, gated = _paused_run()
    state = result.to_state()
    state.approve(state.pending[0])
    gated().run_sync(state)

    assert result.run_id is not None
    assert RunState.load_sync(gated(), result.run_id) is None
    assert RunState.load_sync(gated(), "no-such-run") is None


def test_a_pause_is_found_only_through_the_agent_it_started_with() -> None:
    """A `run_id` under another agent's name is somebody else's pause, so it isn't there."""
    result, _ = _paused_run()
    other = agent_class("OtherAgent", ScriptedModel([text_response("hi")]))
    assert result.run_id is not None

    assert RunState.load_sync(other(), result.run_id) is None


def test_load_rejects_an_unknown_schema_version() -> None:
    """An unrecognized `schema_version` raises `UserError`, not a raw Pydantic error."""
    result, gated = _paused_run()
    blob = _stored(result)
    blob["schema_version"] = 999

    with pytest.raises(UserError):
        RunState.load_sync(gated(), _store(blob))


def test_load_rejects_a_malformed_field() -> None:
    """A field with the wrong shape also surfaces as `UserError`, not a raw Pydantic error."""
    result, gated = _paused_run()
    blob = _stored(result)
    blob["tree"]["approvals"]["calls"] = "not-a-dict"

    with pytest.raises(UserError):
        RunState.load_sync(gated(), _store(blob))


def test_a_pause_keeps_the_current_agent_after_a_handoff_occurred_before_it() -> None:
    """The stored `agent` reflects a handoff; the pause is still found through the first agent."""

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
    assert _stored(result, "Main")["run"]["agent"] == "Target"

    fresh_main = main()
    restored = _load(fresh_main, result.run_id)
    assert restored.agent.name == "Target"
    restored.approve(restored.pending[0])

    resumed = fresh_main.run_sync(restored)
    assert resumed.output == "done"


def test_a_handed_off_run_that_pauses_again_is_kept_under_the_first_agent() -> None:
    """A resumed run's next pause is found the way its first one was: through `Main`."""

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        return "done"

    target = agent_class(
        "Target",
        ScriptedModel(
            [
                tool_call_response("dangerous", "{}"),
                tool_call_response("dangerous", "{}", "call_2"),
                text_response("done"),
            ]
        ),
        tools=[dangerous],
    )
    main = agent_class(
        "Main",
        ScriptedModel([tool_call_response("transfer_to_target", "{}")]),
        subagents=[target.handoff],
    )
    first = main()
    state = first.run_sync("please transfer").to_state()
    state.approve(state.pending[0])

    again = first.run_sync(state)

    assert again.status == "paused"
    assert _load(main(), again.run_id).pending[0].call_id == "call_2"


def test_load_resolves_the_paused_interruptions_agent_from_its_name() -> None:
    """A pending interruption's `agent` is re-resolved by name against the fresh agent."""
    result, gated = _paused_run()

    fresh_agent = gated()
    restored = _load(fresh_agent, result.run_id)

    assert restored.pending[0].agent is fresh_agent
    assert restored.pending[0].name == "dangerous"


def test_load_never_lists_an_agents_mcp_server_tools() -> None:
    """Loading a pause resolves no tools, so an agent's MCP servers are left alone entirely.

    The tool a pending call names is looked up by the resumed run, off the shape it builds
    anyway -- even when the paused call was an MCP server's tool and the fresh agent declares
    nothing locally.
    """

    class _ExplodingServer:
        async def list_tools(self) -> list[Any]:
            raise AssertionError("load must not connect to an MCP server")

    result, _ = _paused_run()
    fresh_agent: Any = SimpleNamespace(
        name="TestAgent", tools=[], handoffs=[], mcp_servers=[_ExplodingServer()]
    )

    restored = _load(fresh_agent, result.run_id)

    assert [item.name for item in restored.pending] == ["dangerous"]


def test_a_loaded_pause_carries_the_sticky_approval_ledger() -> None:
    """A sticky decision made on a loaded pause reaches the run that resumes it."""
    result, gated = _paused_run()
    fresh = gated()
    restored = _load(fresh, result.run_id)
    restored.approve(restored.pending[0], always=True)

    resumed = fresh.run_sync(restored)

    assert resumed.output == "all done"
    assert resuming(restored).tree.approvals.sticky == {"dangerous": True}
    assert "call_1" in resuming(restored).tree.approvals.executed


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
    return gated().run_sync("do it", session=db.session("s"), user_id="u1"), gated


def test_a_restored_run_keeps_its_user_and_the_tool_calls_made_before_the_pause() -> None:
    """Who the run is for and what it already did both come back, with no session to ask."""
    result, gated = _paused_after_a_safe_call()

    fresh = gated()
    restored = _load(fresh, result.run_id)
    restored.approve(restored.pending[0])

    assert restored.run_context.user_id == "u1"
    resumed = fresh.run_sync(restored)
    assert [call.name for call in resumed.tool_calls] == ["safe", "dangerous"]


def test_a_restored_run_saves_its_turn_to_the_session_it_paused_in() -> None:
    """A worker resuming by `run_id` names no session, and the conversation still gets the turn."""
    result, gated = _paused_after_a_safe_call()

    fresh = gated()
    restored = _load(fresh, result.run_id)
    restored.approve(restored.pending[0])
    resumed = fresh.run_sync(restored)

    assert resumed.status == "completed"
    assert restored.session_id == "s"
    items = _loop.run_sync(db.session("s").get_items())
    assert items[0] == {"role": "user", "content": "do it"}
    assert items[-1]["content"] == "all done"
    assert fresh.history == []


@pytest.mark.parametrize("session", ["s", "other", "object of other"])
def test_a_session_beside_a_restored_run_is_refused(session: str) -> None:
    """The run's session comes back with it: an id, or another conversation's object, is refused."""
    result, gated = _paused_after_a_safe_call()
    passed = db.session("other") if session == "object of other" else session

    fresh = gated()
    restored = _load(fresh, result.run_id)

    with pytest.raises(UserError, match="already carries the session"):
        fresh.run_sync(restored, session=passed)


def test_a_restored_trace_keeps_its_session_grouping() -> None:
    """The trace a resumed run continues is still grouped under the session it started on."""
    result, gated = _paused_after_a_safe_call()

    restored = _load(gated(), result.run_id)

    assert restored.trace.session_id == "s"
    assert restored.trace.metadata == {"group_id": "s"}


def test_a_resumed_run_spends_the_timeout_left_over_from_before_the_pause() -> None:
    """The deadline is the run's, not each call's: time spent running before a pause counts."""
    result, gated = _paused_after_a_safe_call(timeout=5.0)
    blob = _stored(result)
    assert blob["run"]["elapsed"] > 0
    blob["run"]["elapsed"] = 5.0

    fresh = gated()
    restored = _load(fresh, _store(blob))
    restored.approve(restored.pending[0])
    resumed = fresh.run_sync(restored)

    assert resumed.status == "error"
    assert resumed.error == "run timed out after 5.0s"


def test_a_resumed_run_spends_the_turns_left_over_from_before_the_pause() -> None:
    """`max_turns` counts the run's model calls, not each resume's: two before the pause count."""
    result, gated = _paused_after_a_safe_call(max_turns=2)
    assert _stored(result)["run"]["turns"] == 2

    fresh = gated()
    restored = _load(fresh, result.run_id)
    restored.approve(restored.pending[0])
    resumed = fresh.run_sync(restored, session=db.session("s"))

    assert resumed.status == "error"
    assert resumed.error == "max turns (2) exceeded"


def test_a_handed_off_run_keeps_the_starting_agents_ceilings_across_a_pause() -> None:
    """After a handoff and a pause, the run is still held to the agent it started with."""

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
        max_turns=2,
    )
    paused = main().run_sync("please transfer")
    assert _stored(paused, "Main")["run"]["start"] == "Main"

    fresh = main()
    restored = _load(fresh, paused.run_id)
    restored.approve(restored.pending[0])
    resumed = fresh.run_sync(restored)

    assert resumed.status == "error"
    assert resumed.error == "max turns (2) exceeded"
