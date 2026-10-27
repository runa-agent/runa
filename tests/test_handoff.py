"""Tests for `Delegate` and the shared agent-graph walk.

What a delegate's run shares with its caller's (sticky approvals, usage, `context`, hooks, the
trace), what the caller's `Run` records of it, and the walk every module needing "reachable from
here" shares (`reachable`/`subagent_edges`).
"""

from collections.abc import Sequence
from typing import Any

import pytest

from runa import Trace
from runa._types import ModelResponse, RunContext, Usage
from runa.agent import Agent, Subagent
from runa.exceptions import UserError
from runa.guardrail import guardrail
from runa.lifecycle import Hooks
from runa.run_internal.agent_shape import reachable, subagent_edges
from runa.tool import ToolCall, tool
from runa.tracing import add_exporter


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

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN002, ANN003
        """Answers unstreamed only."""
        raise NotImplementedError


def _lead(delegate: type[Agent] | Subagent, script: list[dict[str, Any]], **settings: Any) -> Agent:
    """A caller whose model follows `script`, with `delegate` among its tools.

    A class is wired as a plain `.delegate`; pass `Cls.delegate(...)` to give it a policy.
    """

    class Lead(Agent):
        name = "Lead"
        model = _ScriptedModel(script)

    entry = delegate if isinstance(delegate, Subagent) else delegate.delegate
    return Lead(subagents=[entry], **settings)


def test_a_sticky_approval_on_the_caller_covers_the_delegates_matching_tool() -> None:
    """`always=True` on the caller's side lets the same tool run inside a delegate unasked.

    The delegate's run shares its caller's governance, so the second `dangerous` call -- made by
    the delegate -- finds the caller's sticky decision instead of pausing the run again.
    """
    ran: list[str] = []

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Needs approval."""
        ran.append("dangerous")
        return "handled"

    class Delegate(Agent):
        name = "Delegate"
        tools = [dangerous]
        model = _ScriptedModel(
            [_tool_call_message("dangerous", "{}", "d1"), _final_message("delegate done")]
        )

    lead = _lead(
        Delegate,
        [
            _tool_call_message("dangerous", "{}", "c1"),
            _tool_call_message("delegate", '{"input": "go"}', "c2"),
            _final_message("all done"),
        ],
        tools=[dangerous],
    )

    paused = lead.run_sync("go")
    state = paused.to_state()
    state.approve(paused.interruptions[0], always=True)
    run = lead.run_sync(state)

    assert run.status == "completed"
    assert run.output == "all done"
    assert ran == ["dangerous", "dangerous"]


def test_a_delegates_usage_is_merged_into_its_callers() -> None:
    """The caller's `usage` counts the delegate's model calls beside its own."""

    class Delegate(Agent):
        name = "Delegate"
        model = _ScriptedModel([_final_message("delegate done")])

    lead = _lead(
        Delegate,
        [_tool_call_message("delegate", '{"input": "go"}'), _final_message("all done")],
    )

    run = lead.run_sync("go")

    assert run.usage.total_tokens == 6  # two calls of the caller's, one of the delegate's


def test_a_delegates_token_ceiling_counts_its_spend_before_a_pause() -> None:
    """Resuming a delegate resumes its spend too, so its `max_tokens` isn't reset by a pause."""

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Needs approval."""
        return "handled"

    class Delegate(Agent):
        name = "Delegate"
        tools = [dangerous]
        max_tokens = 3
        model = _ScriptedModel(
            [_tool_call_message("dangerous", "{}", "d1"), _final_message("delegate done")]
        )

    lead = _lead(
        Delegate,
        [_tool_call_message("delegate", '{"input": "go"}'), _final_message("all done")],
    )

    paused = lead.run_sync("go")
    state = paused.to_state()
    state.approve(paused.interruptions[0])
    run = lead.run_sync(state)

    [call] = run.tool_calls
    assert call.error is not None and "max tokens (3) exceeded: 4 used" in call.error
    assert run.usage.total_tokens == 8  # two calls of the caller's, two of the delegate's


def test_a_delegate_sees_the_same_context_object_as_its_caller() -> None:
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
            [_tool_call_message("record", "{}", "d1"), _final_message("delegate done")]
        )

    lead = _lead(
        Delegate,
        [_tool_call_message("delegate", '{"input": "go"}'), _final_message("all done")],
    )
    marker = object()

    lead.run_sync("go", context=marker)

    assert seen == [marker]


def test_the_runs_hooks_fire_for_the_delegate_too() -> None:
    """Hooks passed to `run` see every agent in it, the one reached by delegation included."""
    log: list[tuple[str, str]] = []

    class Recording(Hooks[Any]):
        async def on_agent_start(self, context: Any, agent: Any) -> None:
            log.append(("start", agent.name))

        async def on_llm_start(self, context: Any, agent: Any, prompt: Any, items: Any) -> None:
            log.append(("llm", agent.name))

        async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
            log.append(("end", agent.name))

    class Delegate(Agent):
        name = "Delegate"
        model = _ScriptedModel([_final_message("delegate done")])

    lead = _lead(
        Delegate,
        [_tool_call_message("delegate", '{"input": "go"}'), _final_message("all done")],
    )

    lead.run_sync("go", hooks=Recording())

    assert log == [
        ("start", "Lead"),
        ("llm", "Lead"),
        ("start", "Delegate"),
        ("llm", "Delegate"),
        ("end", "Delegate"),
        ("llm", "Lead"),
        ("end", "Lead"),
    ]


def test_a_delegates_spans_are_in_its_callers_trace_under_the_call() -> None:
    """One run is one exported trace: the delegate's agent span hangs under its delegate span."""
    exported: list[Trace] = []

    class Recording:
        def export(self, trace: Trace) -> None:
            exported.append(trace)

    class Delegate(Agent):
        name = "Delegate"
        model = _ScriptedModel([_final_message("delegate done")])

    lead = _lead(
        Delegate,
        [_tool_call_message("delegate", '{"input": "go"}'), _final_message("all done")],
    )

    add_exporter(Recording())
    run = lead.run_sync("go")

    assert [trace.id for trace in exported] == [run.trace.id]
    spans = {span.name: span for span in run.trace.spans if span.type in ("agent", "delegate")}
    assert spans["Lead"].parent_id is None
    assert spans["delegate"].parent_id == spans["Lead"].id
    assert spans["Delegate"].parent_id == spans["delegate"].id
    assert spans["Delegate"].end_time is not None


def test_a_delegate_call_is_on_its_callers_tool_calls_with_the_delegates_own_beneath() -> None:
    """The caller records the delegate call and its output; the delegate's own are its children."""

    @tool
    def lookup(order: str) -> str:
        """Look an order up."""
        return f"{order} shipped"

    class Delegate(Agent):
        name = "Delegate"
        tools = [lookup]
        model = _ScriptedModel(
            [_tool_call_message("lookup", '{"order": "4821"}', "d1"), _final_message("shipped")]
        )

    lead = _lead(
        Delegate,
        [_tool_call_message("delegate", '{"input": "where is 4821?"}'), _final_message("done")],
    )

    run = lead.run_sync("go")

    assert run.status == "completed", run.error
    (call,) = run.tool_calls
    assert (call.name, call.arguments, call.output) == (
        "delegate",
        '{"input": "where is 4821?"}',
        "shipped",
    )
    assert call.children == (ToolCall("lookup", '{"order": "4821"}', "4821 shipped"),)
    assert [each.name for each in call.walk()] == ["delegate", "lookup"]


def test_a_delegate_that_errors_keeps_the_calls_it_made_as_its_children() -> None:
    """A delegate failing after a tool call fails the delegate call, with that call beneath."""

    @tool
    def lookup(order: str) -> str:
        """Look an order up."""
        return f"{order} shipped"

    class Delegate(Agent):
        name = "Delegate"
        tools = [lookup]
        max_turns = 1
        model = _ScriptedModel([_tool_call_message("lookup", '{"order": "4821"}', "d1")])

    lead = _lead(
        Delegate,
        [_tool_call_message("delegate", '{"input": "where is 4821?"}'), _final_message("done")],
    )

    run = lead.run_sync("go")

    assert run.status == "completed", run.error
    (call,) = run.tool_calls
    assert call.error == "max turns (1) exceeded"
    assert call.children == (ToolCall("lookup", '{"order": "4821"}', "4821 shipped"),)


def test_a_delegate_called_with_malformed_arguments_answers_an_error_and_runs_nothing() -> None:
    """Arguments the model got wrong go back to it as the call's result, as for any tool."""

    class Delegate(Agent):
        name = "Delegate"
        model = _ScriptedModel([])  # never called: the delegate doesn't start

    lead = _lead(Delegate, [_tool_call_message("delegate", "not json"), _final_message("ok")])

    run = lead.run_sync("go")

    assert run.status == "completed", run.error
    (call,) = run.tool_calls
    assert (call.name, call.children) == ("delegate", ())
    assert call.error is not None
    assert call.output == f"error: {call.error}"
    assert not [span for span in run.trace.spans if span.type == "delegate"]


class _ToolEvents(Hooks[Any]):
    """Records each `on_tool_start`/`on_tool_end` as `(event, tool name)`."""

    def __init__(self) -> None:
        self.log: list[tuple[str, str]] = []

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        self.log.append(("start", tool.name))

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: object) -> None:
        self.log.append(("end", tool.name))


def test_a_delegate_call_fires_the_tool_hooks_like_any_tool_call() -> None:
    """A run-wide `on_tool_*` sees the delegate call, then the calls the delegate makes."""

    @tool
    def lookup() -> str:
        """Look something up."""
        return "found"

    class Delegate(Agent):
        name = "Delegate"
        tools = [lookup]
        model = _ScriptedModel([_tool_call_message("lookup", "{}", "d1"), _final_message("ok")])

    lead = _lead(
        Delegate,
        [_tool_call_message("delegate", '{"input": "go"}'), _final_message("all done")],
    )
    hooks = _ToolEvents()

    lead.run_sync("go", hooks=hooks)

    assert hooks.log == [
        ("start", "delegate"),
        ("start", "lookup"),
        ("end", "lookup"),
        ("end", "delegate"),
    ]


def test_a_delegate_that_pauses_starts_and_ends_once_across_the_resume() -> None:
    """The delegate call fires `on_tool_start` when it starts and `on_tool_end` when it finishes."""

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Needs approval."""
        return "handled"

    class Delegate(Agent):
        name = "Delegate"
        tools = [dangerous]
        model = _ScriptedModel(
            [_tool_call_message("dangerous", "{}", "d1"), _final_message("delegate done")]
        )

    lead = _lead(
        Delegate,
        [_tool_call_message("delegate", '{"input": "go"}'), _final_message("all done")],
    )
    hooks = _ToolEvents()

    paused = lead.run_sync("go", hooks=hooks)
    state = paused.to_state()
    state.approve(paused.interruptions[0])
    run = lead.run_sync(state, hooks=hooks)

    assert run.status == "completed", run.error
    assert hooks.log == [
        ("start", "delegate"),
        ("start", "dangerous"),
        ("end", "dangerous"),
        ("end", "delegate"),
    ]


def test_a_delegate_that_needs_approval_pauses_before_it_starts() -> None:
    """`.delegate(needs_approval=True)` gates the call the way `@tool(needs_approval=True)` does."""
    started: list[str] = []

    class Delegate(Agent):
        name = "Delegate"
        model = _ScriptedModel([_final_message("delegate done")])

    class Recording(Hooks[Any]):
        async def on_agent_start(self, context: Any, agent: Any) -> None:
            started.append(agent.name)

    lead = _lead(
        Delegate.delegate(needs_approval=True),
        [_tool_call_message("delegate", '{"input": "go"}'), _final_message("all done")],
    )

    paused = lead.run_sync("go", hooks=Recording())

    assert paused.status == "paused"
    assert [item.name for item in paused.interruptions] == ["delegate"]
    assert started == ["Lead"]

    state = paused.to_state()
    state.approve(paused.interruptions[0])
    run = lead.run_sync(state, hooks=Recording())

    assert run.status == "completed", run.error
    assert [call.output for call in run.tool_calls] == ["delegate done"]
    assert "Delegate" in started


def test_a_rejected_delegate_never_starts() -> None:
    """A rejection answers the model in place of the delegate's output, as for a tool."""

    class Delegate(Agent):
        name = "Delegate"
        model = _ScriptedModel([])  # never called: the delegate doesn't start

    lead = _lead(
        Delegate.delegate(needs_approval=True),
        [_tool_call_message("delegate", '{"input": "go"}'), _final_message("all done")],
    )

    paused = lead.run_sync("go")
    state = paused.to_state()
    state.reject(paused.interruptions[0], rejection_message="not now")
    run = lead.run_sync(state)

    assert run.status == "completed", run.error
    assert run.tool_calls == ()
    assert not [span for span in run.trace.spans if span.type == "delegate"]


def test_a_delegates_input_guardrail_trips_before_it_starts() -> None:
    """`.delegate(guardrails=[...])` checks the call's arguments, as a tool guardrail does."""

    @guardrail
    def no_secrets(args: Any) -> bool:
        """Trip on a request for secrets."""
        return "secret" in args["input"]

    class Delegate(Agent):
        name = "Delegate"
        model = _ScriptedModel([])  # never called: the guardrail trips first

    lead = _lead(
        Delegate.delegate(guardrails=[no_secrets.input]),
        [_tool_call_message("delegate", '{"input": "the secret"}'), _final_message("all done")],
    )

    run = lead.run_sync("go")

    assert run.status == "error"
    assert [result.tripped for result in run.tool_input_guardrail_results] == [True]


def test_a_handoff_takes_no_approval_or_guardrails() -> None:
    """A handoff is a switch, not a call, so a policy on one is refused where it's written."""

    class Target(Agent):
        name = "Target"

    with pytest.raises(UserError, match="delegate"):
        Target.handoff(needs_approval=True)


def test_reachable_names_every_agent_once_even_through_a_cycle() -> None:
    """Two agents handing off to each other terminate the walk, each named exactly once."""

    class First(Agent):
        name = "first"
        instructions = "First."
        subagents: Sequence[type[Agent] | Subagent] = ()  # filled in below, once `Second` exists

    class Second(Agent):
        name = "second"
        instructions = "Second."
        subagents = [First.handoff]  # a back-edge the walk must not follow

    First.subagents = [Second.handoff]
    first = First()

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
