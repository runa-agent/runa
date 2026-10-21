"""Tests for `run_internal.run_loop`: the turn loop behind every `Agent.run`.

These go through `Agent.run_sync`/`run_streamed`, over a real `Agent` answering with a scripted
`Model` (`helpers.scripted_agent`), so a test pins what an application can observe rather than
the loop's own signature. The one that drives `_run_async` directly does so because the input it
needs, a list that doesn't end on the user's message, is not one `Agent.run` builds.
"""

import asyncio
import re
from collections.abc import AsyncGenerator
from typing import Any, cast

from helpers import (
    ScriptedModel,
    agent_class,
    scripted_agent,
    stand_in_agent,
    text_response,
    tool_call_response,
)

from runa._models import ModelRequest, StreamDelta
from runa._types import ModelResponse, ModelSettings, Usage
from runa.compact import default_compactor
from runa.exceptions import ServiceUnavailable, UserError
from runa.guardrail import BoundGuardrail, GuardrailVerdict, Phase, guardrail
from runa.lifecycle import Hooks
from runa.run import Run
from runa.run_internal.run_loop import _run_async
from runa.tool import ToolCall, tool
from runa.tracing.util import gen_trace_id


class _ScriptedStreamingModel:
    """A `Model` stand-in that yields pre-scripted `StreamDelta`s in order."""

    def __init__(self, deltas: list[StreamDelta]) -> None:
        self._deltas = deltas

    async def get_response(self, *args: Any, **kwargs: Any):  # noqa: ANN001, ANN002, ANN003
        raise NotImplementedError

    async def stream_response(self, request):  # noqa: ANN001, ARG002
        for delta in self._deltas:
            yield delta


class _SequentialStreamingModel:
    """A streaming `Model` stand-in that yields a different pre-scripted delta list per call.

    Unlike `_ScriptedStreamingModel` (which replays the same deltas forever), this pops one
    delta list per `stream_response` call -- needed for multi-turn streaming tests where each
    turn must look different (e.g. a tool call, then a final text reply).
    """

    def __init__(self, delta_lists: list[list[StreamDelta]]) -> None:
        self._delta_lists = list(delta_lists)

    async def get_response(self, *args: Any, **kwargs: Any):  # noqa: ANN001, ANN002, ANN003
        raise NotImplementedError

    async def stream_response(self, *args: Any, **kwargs: Any):  # noqa: ANN001, ANN002, ANN003
        for delta in self._delta_lists.pop(0):
            yield delta


def test_plain_text_turn_returns_final_output() -> None:
    """A model reply with no tool calls becomes the run's final output directly."""
    agent = scripted_agent(model=ScriptedModel([text_response("hello there")]))

    result = agent.run_sync("hi")

    assert result.output == "hello there"
    assert result.interruptions == []
    assert result.usage.input_tokens == 1


def test_run_sync_matches_run() -> None:
    """`run_sync` is a synchronous wrapper with identical behavior to `run`."""
    sync = scripted_agent(model=ScriptedModel([text_response("ok")])).run_sync("hi")
    unsync = asyncio.run(scripted_agent(model=ScriptedModel([text_response("ok")])).run("hi"))

    assert (sync.status, sync.output) == (unsync.status, unsync.output) == ("completed", "ok")


def test_tool_call_then_final_text() -> None:
    """A tool call is executed and its result fed back before the model gives a final answer."""

    @tool
    def now() -> str:
        """Return a fixed time."""
        return "2024-01-01"

    agent = scripted_agent(
        tools=[now],
        model=ScriptedModel([tool_call_response("now", "{}"), text_response("it is 2024-01-01")]),
    )

    result = agent.run_sync("what time is it?")

    assert result.output == "it is 2024-01-01"
    assert result.tool_calls == (ToolCall(name="now", arguments="{}", output="2024-01-01"),)
    tool_spans = [s for s in result.trace.spans if s.type == "tool"]
    assert len(tool_spans) == 1
    assert tool_spans[0].name == "now"
    assert tool_spans[0].status == "ok"
    agent_spans = [s for s in result.trace.spans if s.type == "agent"]
    assert agent_spans[0].name == "TestAgent"


def test_tool_error_is_fed_back_and_run_continues() -> None:
    """A tool that raises doesn't abort the run; its span records the error, the run completes."""

    @tool
    def boom() -> str:
        """Raise unconditionally."""
        raise ValueError("boom")

    agent = scripted_agent(
        tools=[boom],
        model=ScriptedModel([tool_call_response("boom", "{}"), text_response("handled it")]),
    )

    result = agent.run_sync("go")

    assert result.output == "handled it"
    (tool_span,) = [s for s in result.trace.spans if s.type == "tool"]
    assert tool_span.status == "error"
    assert "boom" in (tool_span.error or "")


def test_malformed_tool_arguments_are_fed_back_and_run_continues() -> None:
    """Invalid JSON arguments from the model become a tool error, not a run-ending crash."""

    @tool
    def lookup(city: str) -> str:
        """Look up a city."""
        return city

    model = ScriptedModel([tool_call_response("lookup", "{not json"), text_response("sorry")])
    agent = scripted_agent(tools=[lookup], model=model)

    result = agent.run_sync("go")

    assert result.output == "sorry"
    assert model.calls[1][-1]["content"].startswith("error: invalid JSON arguments")


def test_non_object_tool_arguments_are_fed_back_as_an_error() -> None:
    """Valid JSON that isn't an object (e.g. a list) is rejected the same way."""

    @tool
    def lookup(city: str) -> str:
        """Look up a city."""
        return city

    model = ScriptedModel([tool_call_response("lookup", "[1]"), text_response("sorry")])
    agent = scripted_agent(tools=[lookup], model=model)

    agent.run_sync("go")

    assert model.calls[1][-1]["content"] == "error: tool arguments must be a JSON object"


def test_handoff_switches_current_agent() -> None:
    """Calling a handoff's tool switches to the target agent for the rest of the run."""
    target = agent_class("Target", ScriptedModel([text_response("handled by target")]))
    main = scripted_agent(
        name="Main",
        subagents=[target.handoff],
        model=ScriptedModel([tool_call_response("transfer_to_target", "{}")]),
    )

    result = main.run_sync("please transfer")

    assert result.output == "handled by target"
    handoff_spans = [s for s in result.trace.spans if s.type == "handoff"]
    assert len(handoff_spans) == 1


def test_delegate_call_is_traced_as_a_delegate_span_not_a_tool_span() -> None:
    """A `.delegate` call traces as `type="delegate"`, unlike a plain tool call.

    A delegate call runs a whole nested run -- its own agent span and LLM call, under the
    delegate span -- so it gets its own span type, the same way a handoff does. It is still
    recorded as one of the caller's tool calls, with what the delegate answered.
    """
    researcher = agent_class("researcher", ScriptedModel([text_response("the policy is 30 days")]))
    caller = scripted_agent(
        subagents=[researcher.delegate],
        model=ScriptedModel(
            [
                tool_call_response("researcher", '{"input": "what is the policy?"}'),
                text_response("it's 30 days"),
            ]
        ),
    )

    result = caller.run_sync("what's the policy?")

    assert result.output == "it's 30 days"
    (delegate_span,) = [s for s in result.trace.spans if s.type == "delegate"]
    assert delegate_span.name == "researcher"
    (nested,) = [s for s in result.trace.spans if s.parent_id == delegate_span.id]
    assert (nested.type, nested.name) == ("agent", "researcher")
    assert not [s for s in result.trace.spans if s.type == "tool"]
    assert result.tool_calls == (
        ToolCall("researcher", '{"input": "what is the policy?"}', "the policy is 30 days"),
    )


def test_a_handoff_reaches_the_model_under_the_tool_name_it_was_given() -> None:
    """`.handoff(tool_name=..., tool_description=...)` is what the model is offered."""
    target = agent_class("Target", ScriptedModel([text_response("handled by target")]))
    model = ScriptedModel([text_response("hi")])
    escalate = target.handoff(tool_name="escalate", tool_description="Escalate to a human.")
    main = scripted_agent(name="Main", subagents=[escalate], model=model)

    main.run_sync("hello")

    (received,) = model.received_tools
    assert (received.name, received.description) == ("escalate", "Escalate to a human.")
    assert received.parameters == {"type": "object", "properties": {}}


def test_input_guardrail_tripwire_halts_the_run() -> None:
    """A tripped input guardrail ends the run before the model is ever called."""

    @guardrail
    def block_all(message: str) -> bool:
        """Trip on every message."""
        return True

    model = ScriptedModel([text_response("should not be reached")])
    agent = scripted_agent(guardrails=[block_all.input], model=model)

    result = agent.run_sync("hi")

    assert model.calls == []

    assert result.status == "error"
    assert "block_all" in str(result.error)


def test_output_guardrail_tripwire_halts_the_run() -> None:
    """A tripped output guardrail ends the run after the model responds, before returning."""

    @guardrail
    def block_long(answer: str) -> bool:
        """Trip on an answer longer than three characters."""
        return len(answer) > 3

    agent = scripted_agent(
        guardrails=[block_long.output], model=ScriptedModel([text_response("way too long")])
    )

    result = agent.run_sync("hi")

    assert result.status == "error"
    assert result.output is None
    assert "block_long" in str(result.error)


def test_tool_input_guardrail_tripwire_halts_the_run() -> None:
    """A tripped tool input guardrail aborts the whole run, unlike a plain tool exception."""

    @guardrail
    def block_args(args: dict[str, Any]) -> bool:
        """Trip on any arguments."""
        return bool(args)

    @tool(guardrails=[block_args.input])
    def search(query: str) -> str:
        """Search for something."""
        return "results"

    agent = scripted_agent(
        tools=[search],
        model=ScriptedModel([tool_call_response("search", '{"query": "x"}')]),
    )

    result = agent.run_sync("search for x")

    assert result.status == "error"
    assert "block_args" in str(result.error)


def test_max_turns_exceeded() -> None:
    """A model that keeps calling tools forever ends the run with `MaxTurnsExceeded`'s message."""

    @tool
    def loop_tool() -> str:
        """Always return the same thing."""
        return "again"

    responses = [tool_call_response("loop_tool", "{}", call_id=f"call_{i}") for i in range(5)]
    agent = scripted_agent(tools=[loop_tool], model=ScriptedModel(responses), max_turns=3)

    result = agent.run_sync("go")

    assert result.status == "error"
    assert result.error == "max turns (3) exceeded"


def test_a_failed_run_comes_back_as_an_error_run_with_its_trace() -> None:
    """A failure is the loop's third `Run`, not an exception carrying what the run had.

    The spans are what report what it produced before failing: the run reached the model twice
    and ran a tool each time, and the trace says so even though nothing was answered.
    """

    @tool
    def loop_tool() -> str:
        """Always return the same thing."""
        return "again"

    responses = [tool_call_response("loop_tool", "{}", call_id=f"call_{i}") for i in range(2)]
    agent = scripted_agent(tools=[loop_tool], model=ScriptedModel(responses), max_turns=2)

    result = agent.run_sync("go")

    assert result.status == "error"
    assert result.output is None
    assert len([s for s in result.trace.spans if s.type == "llm"]) == 2
    assert len([s for s in result.trace.spans if s.type == "tool"]) == 2
    (agent_span,) = [s for s in result.trace.spans if s.type == "agent"]
    assert agent_span.status == "error"
    assert agent_span.error == "max turns (2) exceeded"


def test_a_provider_error_closes_the_llm_span_with_the_error() -> None:
    """The model call that failed is the span worth inspecting, so it isn't left open."""

    class _Down(ScriptedModel):
        async def get_response(self, request):  # noqa: ANN001, ARG002
            raise ConnectionError("provider is down")

    result = scripted_agent(model=_Down([])).run_sync("hi")

    assert result.status == "error"
    (llm_span,) = [s for s in result.trace.spans if s.type == "llm"]
    assert llm_span.end_time is not None
    assert llm_span.status == "error"
    assert llm_span.error == "ConnectionError: provider is down"


def test_an_exceeded_token_budget_closes_the_llm_span_with_its_usage() -> None:
    """The call that went over budget still says what it spent, and that it ended the run."""
    usage = Usage(input_tokens=5, output_tokens=5, total_tokens=10, requests=1)
    agent = scripted_agent(model=ScriptedModel([text_response("hi", usage)]), max_tokens=3)

    result = agent.run_sync("hi")

    assert result.status == "error"
    (llm_span,) = [s for s in result.trace.spans if s.type == "llm"]
    assert llm_span.end_time is not None
    assert llm_span.status == "error"
    assert "max tokens (3) exceeded" in (llm_span.error or "")
    assert llm_span.output["usage"]["total_tokens"] == 10


def test_a_guardrail_that_raises_closes_its_span_with_the_error() -> None:
    """A predicate that raises is a failed check, not one that never finished."""

    @guardrail
    def broken(message: str) -> bool:
        """Fail to decide."""
        raise ValueError("bad predicate")

    agent = scripted_agent(
        guardrails=[broken.input], model=ScriptedModel([text_response("should not be reached")])
    )

    result = agent.run_sync("hi")

    assert result.status == "error"
    (guardrail_span,) = [s for s in result.trace.spans if s.type == "guardrail"]
    assert guardrail_span.end_time is not None
    assert guardrail_span.error == "ValueError: bad predicate"


def test_needs_approval_pauses_then_resumes_on_approve() -> None:
    """A tool requiring approval pauses the run with an `Interruption`; approving resumes it."""

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        return "done"

    agent = scripted_agent(
        tools=[dangerous],
        model=ScriptedModel([tool_call_response("dangerous", "{}"), text_response("all done")]),
    )

    result = agent.run_sync("do it")

    assert result.output is None
    assert len(result.interruptions) == 1
    interruption = result.interruptions[0]
    assert interruption.name == "dangerous"

    state = result.to_state()
    state.approve(interruption)
    resumed = agent.run_sync(state)

    assert resumed.output == "all done"


def test_needs_approval_rejected_feeds_back_and_continues() -> None:
    """Rejecting a paused tool call resumes with a rejection message, not the tool's result."""

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        return "should not run"

    agent = scripted_agent(
        tools=[dangerous],
        model=ScriptedModel(
            [tool_call_response("dangerous", "{}"), text_response("okay, skipped it")]
        ),
    )

    result = agent.run_sync("do it")
    state = result.to_state()
    state.reject(result.interruptions[0])
    resumed = agent.run_sync(state)

    assert resumed.output == "okay, skipped it"


def test_needs_approval_always_approve_skips_future_prompts_for_the_same_tool() -> None:
    """Approving with `always=True` sticks; a later call to the same tool doesn't re-prompt."""

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        return "done"

    agent = scripted_agent(
        tools=[dangerous],
        model=ScriptedModel(
            [
                tool_call_response("dangerous", "{}", call_id="call_1"),
                tool_call_response("dangerous", "{}", call_id="call_2"),
                text_response("all done"),
            ]
        ),
    )

    result = agent.run_sync("do it")
    state = result.to_state()
    state.approve(result.interruptions[0], always=True)

    resumed = agent.run_sync(state)

    assert resumed.interruptions == []
    assert resumed.output == "all done"


def test_reject_with_a_custom_rejection_message_feeds_the_custom_text_back() -> None:
    """A custom `rejection_message` is fed back to the model instead of the default text."""

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        return "should not run"

    agent = scripted_agent(
        tools=[dangerous],
        model=ScriptedModel(
            [tool_call_response("dangerous", "{}"), text_response("okay, skipped it")]
        ),
    )

    result = agent.run_sync("do it")
    state = result.to_state()
    state.reject(result.interruptions[0], rejection_message="not allowed today")
    agent.run_sync(state)

    tool_messages = [item["content"] for item in agent.history if item.get("role") == "tool"]
    assert "not allowed today" in tool_messages


def test_needs_approval_always_reject_feeds_back_the_sticky_message_for_later_calls() -> None:
    """Rejecting with `always=True` + a message sticks; a later call reuses that message too."""

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        return "should not run"

    agent = scripted_agent(
        tools=[dangerous],
        model=ScriptedModel(
            [
                tool_call_response("dangerous", "{}", call_id="call_1"),
                tool_call_response("dangerous", "{}", call_id="call_2"),
                text_response("noted twice"),
            ]
        ),
    )

    result = agent.run_sync("do it")
    state = result.to_state()
    state.reject(result.interruptions[0], always=True, rejection_message="not allowed, ever")

    resumed = agent.run_sync(state)

    assert resumed.interruptions == []
    tool_messages = [item["content"] for item in agent.history if item.get("role") == "tool"]
    assert tool_messages.count("not allowed, ever") == 2


def _safe_and_gated_calls_response() -> ModelResponse:
    """One assistant message calling an ungated `safe` tool and an approval-gated `gated` one."""
    return ModelResponse(
        output=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "safe", "arguments": "{}"},
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "gated", "arguments": "{}"},
                    },
                ],
            }
        ],
        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, requests=1),
    )


def _safe_and_gated_tools(ran: list[str]) -> list[Any]:
    @tool
    def safe() -> str:
        """Run without approval."""
        ran.append("safe")
        return "safe done"

    @tool(needs_approval=True)
    def gated() -> str:
        """Needs approval."""
        ran.append("gated")
        return "gated done"

    return [safe, gated]


def test_resume_runs_the_approved_call_and_reuses_the_ready_result_from_the_same_message() -> None:
    """A message mixing an ungated and a gated call resumes with both results, each run once."""
    ran: list[str] = []
    model = ScriptedModel([_safe_and_gated_calls_response(), text_response("all done")])
    agent = scripted_agent(tools=_safe_and_gated_tools(ran), model=model)

    result = agent.run_sync("go")
    state = result.to_state()
    state.approve(result.interruptions[0])
    resumed = agent.run_sync(state)

    assert ran == ["safe", "gated"]
    assert resumed.output == "all done"
    assert [item["role"] for item in model.calls[1]] == ["user", "assistant", "tool", "tool"]
    assert [item["content"] for item in model.calls[1][2:]] == ["safe done", "gated done"]
    assert agent.history == [
        {"role": "user", "content": "go"},
        _safe_and_gated_calls_response().output[0],
        {"role": "tool", "tool_call_id": "c1", "content": "safe done"},
        {"role": "tool", "tool_call_id": "c2", "content": "gated done"},
        {"role": "assistant", "content": "all done", "tool_calls": None},
    ]


def test_a_resumed_run_reports_the_calls_that_ran_before_the_pause_too() -> None:
    """One turn, one record: an approval in the middle of it doesn't split what the run did.

    The paused `RunState` carries the calls already executed, so the `Run` that finally comes back
    reports the whole turn -- what `runa.eval` grades, and what `run.interruptions` suggested the
    operator was deciding about in the first place.
    """
    ran: list[str] = []
    agent = scripted_agent(
        tools=_safe_and_gated_tools(ran),
        model=ScriptedModel([_safe_and_gated_calls_response(), text_response("all done")]),
    )

    result = agent.run_sync("go")
    assert result.tool_calls == (ToolCall(name="safe", arguments="{}", output="safe done"),)

    state = result.to_state()
    state.approve(result.interruptions[0])
    resumed = agent.run_sync(state)

    assert resumed.tool_calls == (
        ToolCall(name="safe", arguments="{}", output="safe done"),
        ToolCall(name="gated", arguments="{}", output="gated done"),
    )
    # The resumed run continued from a copy: the paused run still reads as it was paused.
    assert result.tool_calls == (ToolCall(name="safe", arguments="{}", output="safe done"),)
    assert state.paused_message == _safe_and_gated_calls_response().output[0]


def test_resume_persists_the_whole_turn_to_the_session(tmp_path: Any) -> None:
    """A paused session-backed run saves nothing; resuming saves the full turn once."""
    from runa.session import SQLiteSession

    session = SQLiteSession("s1", db_path=tmp_path / "runa.db")
    ran: list[str] = []
    agent = scripted_agent(
        tools=_safe_and_gated_tools(ran),
        model=ScriptedModel([_safe_and_gated_calls_response(), text_response("all done")]),
    )

    result = agent.run_sync("go", session=session)
    assert asyncio.run(session.get_items()) == []

    state = result.to_state()
    state.approve(result.interruptions[0])
    agent.run_sync(state, session=session)

    assert asyncio.run(session.get_items()) == [
        {"role": "user", "content": "go"},
        _safe_and_gated_calls_response().output[0],
        {"role": "tool", "tool_call_id": "c1", "content": "safe done"},
        {"role": "tool", "tool_call_id": "c2", "content": "gated done"},
        {"role": "assistant", "content": "all done", "tool_calls": None},
    ]


def test_a_loaded_pause_keeps_the_session_turn(tmp_path: Any) -> None:
    """The transcript's `session_input`/`generated` survive the stored pause and a restart."""
    from runa.run_state import RunState
    from runa.session import SQLiteSession

    session = SQLiteSession("s1", db_path=tmp_path / "runa.db")
    ran: list[str] = []
    agent = scripted_agent(
        tools=_safe_and_gated_tools(ran),
        model=ScriptedModel([_safe_and_gated_calls_response(), text_response("all done")]),
    )

    result = agent.run_sync("go", session=session)
    assert result.run_id is not None
    state = RunState.load_sync(agent, result.run_id)
    assert state is not None
    state.approve(state.pending[0])
    agent.run_sync(state, session=session)

    assert ran == ["safe", "gated"]
    items = asyncio.run(session.get_items())
    assert items[0] == {"role": "user", "content": "go"}
    assert items[-1] == {"role": "assistant", "content": "all done", "tool_calls": None}
    assert len(items) == 5


def _two_calls_response(first: str, second: str) -> ModelResponse:
    """One assistant message calling `first` then `second`, both with no arguments."""
    return ModelResponse(
        output=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": first, "arguments": "{}"},
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": second, "arguments": "{}"},
                    },
                ],
            }
        ],
        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, requests=1),
    )


def test_tool_calls_in_one_message_run_concurrently_and_keep_call_order() -> None:
    """Two calls that each wait on the other only finish if they run at the same time."""
    ping_seen, pong_seen = asyncio.Event(), asyncio.Event()

    @tool
    async def ping() -> str:
        """Wait for pong."""
        ping_seen.set()
        await asyncio.wait_for(pong_seen.wait(), timeout=1)
        return "ping done"

    @tool
    async def pong() -> str:
        """Wait for ping."""
        pong_seen.set()
        await asyncio.wait_for(ping_seen.wait(), timeout=1)
        return "pong done"

    model = ScriptedModel([_two_calls_response("ping", "pong"), text_response("both done")])
    agent = scripted_agent(tools=[ping, pong], model=model)

    result = agent.run_sync("go")

    assert result.output == "both done"
    assert [item["content"] for item in model.calls[1][2:]] == ["ping done", "pong done"]


def test_a_tripped_call_cancels_its_sibling_and_both_spans_say_why() -> None:
    """A tool guardrail trip cancels the other in-flight call; each span records its own cause."""

    @guardrail
    def always_trip(args: dict[str, Any]) -> bool:
        """Trip on every call."""
        return True

    @tool(guardrails=[always_trip.input])
    def blocked() -> str:
        """Never runs."""
        return "unreachable"

    @tool
    async def slow() -> str:
        """Take a long time."""
        await asyncio.sleep(10)
        return "unreachable"

    agent = scripted_agent(
        tools=[blocked, slow],
        model=ScriptedModel([_two_calls_response("slow", "blocked")]),
    )

    from runa.tracing import observe

    exported: dict[str, str | None] = {}

    class _Capture:
        def export(self, trace: Any) -> None:
            exported.update({s.name: s.error for s in trace.spans if s.type == "tool"})

    with observe.scoped(exporter=[_Capture()]):
        result = agent.run_sync("go")

    assert result.status == "error"
    # As exported, not after `asyncio.run` tidied up leftover tasks.
    assert exported["slow"] == "CancelledError"
    assert (exported["blocked"] or "").startswith("GuardrailTripwireTriggered")


def test_parallel_tool_calls_false_runs_calls_one_at_a_time() -> None:
    """`parallel_tool_calls=False` is the escape hatch for tools that can't overlap."""
    log: list[str] = []

    @tool
    async def first() -> str:
        """Run first."""
        log.append("first start")
        await asyncio.sleep(0.01)
        log.append("first end")
        return "first done"

    @tool
    async def second() -> str:
        """Run second."""
        log.append("second start")
        return "second done"

    agent = scripted_agent(
        tools=[first, second],
        model=ScriptedModel([_two_calls_response("first", "second"), text_response("done")]),
        model_settings=ModelSettings(parallel_tool_calls=False),
    )

    agent.run_sync("go")

    assert log == ["first start", "first end", "second start"]


def test_resuming_the_same_state_twice_reports_it_already_resumed() -> None:
    """Resuming the same paused `RunState` a second time doesn't silently re-run the tool."""

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        return "done"

    agent = scripted_agent(
        tools=[dangerous],
        model=ScriptedModel([tool_call_response("dangerous", "{}"), text_response("all done")]),
    )

    result = agent.run_sync("do it")
    state = result.to_state()
    state.approve(result.interruptions[0])

    agent.run_sync(state)

    replayed = agent.run_sync(state)

    assert replayed.status == "error"
    assert replayed.error == f"paused run {result.run_id!r} was already resumed"


def test_passing_input_guardrails_are_recorded_even_though_nothing_tripped() -> None:
    """A guardrail that runs and doesn't trip still shows up in `input_guardrail_results`."""

    @guardrail
    def check(message: str) -> bool:
        """Never trip."""
        return False

    agent = scripted_agent(guardrails=[check.input], model=ScriptedModel([text_response("hi")]))

    result = agent.run_sync("hi")

    assert len(result.input_guardrail_results) == 1
    assert result.input_guardrail_results[0].tripped is False


def test_a_tripped_input_guardrail_still_records_the_guardrails_that_passed_before_it() -> None:
    """Earlier passing guardrails aren't lost when a later one trips and ends the run."""

    @guardrail
    def first(message: str) -> bool:
        """Never trip."""
        return False

    @guardrail
    def second(message: str) -> bool:
        """Trip on every message."""
        return True

    agent = scripted_agent(
        guardrails=[first.input, second.input],
        model=ScriptedModel([text_response("should not be reached")]),
    )

    result = agent.run_sync("hi")

    assert result.status == "error"
    assert len(result.input_guardrail_results) == 2
    assert [r.tripped for r in result.input_guardrail_results] == [False, True]


def test_tool_input_guardrail_results_are_recorded_even_when_they_pass() -> None:
    """A tool input guardrail that passes still shows up in `tool_input_guardrail_results`."""

    @guardrail
    def allow_all(args: dict[str, Any]) -> bool:
        """Never trip."""
        return False

    @tool(guardrails=[allow_all.input])
    def search(query: str) -> str:
        """Search for something."""
        return "results"

    agent = scripted_agent(
        tools=[search],
        model=ScriptedModel([tool_call_response("search", '{"query": "x"}'), text_response("ok")]),
    )

    result = agent.run_sync("search for x")

    assert len(result.tool_input_guardrail_results) == 1
    assert result.tool_input_guardrail_results[0].tripped is False


def test_a_tool_guardrail_is_handed_the_calls_arguments_then_its_return_value() -> None:
    """The runner hands a tool phase its one raw value: the arguments JSON, the return value."""
    seen: list[Any] = []

    async def _record(value: Any) -> GuardrailVerdict:
        seen.append(value)
        return GuardrailVerdict(output_info=None, tripped=False)

    @tool
    def search(query: str) -> str:
        """Search for something."""
        return "results"

    search.guardrails = {
        Phase.TOOL_INPUT: [BoundGuardrail(_record, Phase.TOOL_INPUT)],
        Phase.TOOL_OUTPUT: [BoundGuardrail(_record, Phase.TOOL_OUTPUT)],
    }

    agent = scripted_agent(
        tools=[search],
        model=ScriptedModel([tool_call_response("search", '{"query": "x"}'), text_response("ok")]),
    )

    agent.run_sync("search for x")

    # Unreduced: parsing the arguments is `guardrail._checked`'s job, not the runner's.
    assert seen == ['{"query": "x"}', "results"]


def test_a_paused_run_states_guardrail_results_reflect_what_ran_before_the_pause() -> None:
    """A paused `RunState`'s guardrail results only include what ran before the interruption."""

    @guardrail
    def allow_all(args: dict[str, Any]) -> bool:
        """Never trip."""
        return False

    @tool(guardrails=[allow_all.input], needs_approval=True)
    def dangerous(query: str) -> str:
        """Needs approval."""
        return "done"

    agent = scripted_agent(
        tools=[dangerous],
        model=ScriptedModel(
            [tool_call_response("dangerous", '{"query": "x"}'), text_response("done")]
        ),
    )

    result = agent.run_sync("do it")

    # The tool input guardrail only runs once the call is actually executed (i.e. after
    # approval), so pausing on `needs_approval` records nothing yet.
    assert result.tool_input_guardrail_results == []

    state = result.to_state()
    state.approve(result.interruptions[0])
    resumed = agent.run_sync(state)

    assert len(resumed.tool_input_guardrail_results) == 1


def _streamed(agent: Any, message: Any, **kwargs: Any) -> tuple[list[Any], Run]:
    """Consume `agent.run_streamed(message)`: every event it yielded, and the `Run` it ended on."""
    stream = agent.run_streamed(message, **kwargs)

    async def collect() -> list[Any]:
        return [event async for event in stream]

    events = asyncio.run(collect())
    assert stream.run is not None
    return events, stream.run


def test_stream_response_yields_text_as_it_arrives() -> None:
    """Streaming a plain-text reply yields one `TextDelta` per piece, and nothing else."""
    from runa.stream_events import TextDelta

    agent = scripted_agent(
        model=_ScriptedStreamingModel(
            [StreamDelta(text="Hi"), StreamDelta(usage=Usage()), StreamDelta(text=" there")]
        )
    )

    events, run = _streamed(agent, "hi")

    assert events == [TextDelta("Hi"), TextDelta(" there")]
    assert run.output == "Hi there"


def test_stream_response_keeps_the_provider_blocks_it_was_handed() -> None:
    """What a streamed turn puts in the conversation matches what an unstreamed one returns."""
    block = {"type": "thinking", "thinking": "let me see", "signature": "sig-1"}
    agent = scripted_agent(
        model=_ScriptedStreamingModel(
            [StreamDelta(provider_blocks=[block]), StreamDelta(text="Hi")]
        )
    )

    _streamed(agent, "hi")

    assert agent.history[-1] == {
        "role": "assistant",
        "content": "Hi",
        "tool_calls": None,
        "provider_blocks": [block],
    }


def test_stream_response_runs_a_sticky_approved_tool_without_raising() -> None:
    """After an `always=True` approval, a later call to the same tool runs mid-stream unasked."""
    from runa.stream_events import ToolOutput

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Needs approval."""
        return "done"

    def call(call_id: str) -> list[StreamDelta]:
        return [
            StreamDelta(tool_call_index=0, tool_call_id=call_id, tool_call_name="dangerous"),
            StreamDelta(tool_call_index=0, tool_call_arguments="{}"),
        ]

    agent = scripted_agent(
        tools=[dangerous],
        model=_SequentialStreamingModel(
            [call("call_1"), call("call_2"), [StreamDelta(text="all done")]]
        ),
    )

    _, paused = _streamed(agent, "do it")
    state = paused.to_state()
    state.approve(paused.interruptions[0], always=True)
    events, resumed = _streamed(agent, state)

    assert resumed.output == "all done"
    tool_outputs = [e for e in events if isinstance(e, ToolOutput)]
    assert [e.item["content"] for e in tool_outputs] == ["done", "done"]


def test_stream_response_feeds_back_malformed_tool_arguments() -> None:
    """`run_streamed` turns invalid JSON arguments into a `ToolOutput` error too."""
    from runa.stream_events import ToolOutput

    @tool
    def lookup(city: str) -> str:
        """Look up a city."""
        return city

    agent = scripted_agent(
        tools=[lookup],
        model=_SequentialStreamingModel(
            [
                [
                    StreamDelta(tool_call_index=0, tool_call_id="call_1", tool_call_name="lookup"),
                    StreamDelta(tool_call_index=0, tool_call_arguments="{not json"),
                ],
                [StreamDelta(text="sorry")],
            ]
        ),
    )

    events, _ = _streamed(agent, "go")

    tool_outputs = [e for e in events if isinstance(e, ToolOutput)]
    assert tool_outputs[0].item["content"].startswith("error: invalid JSON arguments")


def test_stream_pauses_for_approval_and_resumes_streaming() -> None:
    """A gated call ends the stream with `interruptions`; the resolved state streams onward."""
    from runa.stream_events import TextDelta, ToolCalled, ToolOutput

    ran: list[str] = []

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Needs approval."""
        ran.append("dangerous")
        return "done"

    agent = scripted_agent(
        tools=[dangerous],
        model=_SequentialStreamingModel(
            [
                [
                    StreamDelta(
                        tool_call_index=0, tool_call_id="call_1", tool_call_name="dangerous"
                    ),
                    StreamDelta(tool_call_index=0, tool_call_arguments="{}"),
                ],
                [StreamDelta(text="all done")],
            ]
        ),
    )

    events, paused = _streamed(agent, "do it")

    assert ran == []
    assert [e.item["function"]["name"] for e in events if isinstance(e, ToolCalled)] == [
        "dangerous"
    ]
    assert paused.output is None
    assert [item.name for item in paused.interruptions] == ["dangerous"]

    state = paused.to_state()
    state.approve(paused.interruptions[0])
    events, resumed = _streamed(agent, state)

    assert ran == ["dangerous"]
    assert resumed.output == "all done"
    assert [type(e) for e in events] == [ToolOutput, TextDelta]
    assert events[0].item["content"] == "done"


def test_stream_announces_the_agent_a_handoff_switched_to() -> None:
    """A handoff mid-stream yields the call, its output, then `AgentChanged` naming the target."""
    from runa.stream_events import AgentChanged, TextDelta, ToolCalled, ToolOutput

    target = agent_class("Target", _ScriptedStreamingModel([StreamDelta(text="handled by target")]))
    main = scripted_agent(
        name="Main",
        subagents=[target.handoff],
        model=_ScriptedStreamingModel(
            [
                StreamDelta(
                    tool_call_index=0, tool_call_id="call_1", tool_call_name="transfer_to_target"
                ),
                StreamDelta(tool_call_index=0, tool_call_arguments="{}"),
            ]
        ),
    )

    events, run = _streamed(main, "please transfer")

    assert [type(e) for e in events] == [ToolCalled, ToolOutput, AgentChanged, TextDelta]
    assert events[2].agent.name == "Target"
    assert run.output == "handled by target"


def test_stream_response_ends_in_error_on_a_replayed_call_id() -> None:
    """A replayed `call_id` across turns doesn't silently re-run the tool during streaming."""

    @tool
    def now() -> str:
        """Return a fixed time."""
        return "2024-01-01"

    same_call = [
        StreamDelta(tool_call_index=0, tool_call_id="call_1", tool_call_name="now"),
        StreamDelta(tool_call_index=0, tool_call_arguments="{}"),
    ]
    agent = scripted_agent(
        tools=[now], model=_SequentialStreamingModel([same_call, list(same_call)])
    )

    _, result = _streamed(agent, "what time is it?")

    assert result.status == "error"
    assert "already executed" in str(result.error)


def test_stream_runs_input_guardrails() -> None:
    """`run_streamed` shares `run`'s loop, so a tripped input guardrail halts it too."""

    @guardrail
    def block_all(message: str) -> bool:
        """Trip on every message."""
        return True

    agent = scripted_agent(
        guardrails=[block_all.input],
        model=_ScriptedStreamingModel([StreamDelta(text="should not be reached")]),
    )

    _, result = _streamed(agent, "hi")

    assert result.status == "error"
    assert "block_all" in str(result.error)


def test_stream_exposes_the_finished_result_with_a_trace() -> None:
    """Once consumed, a stream carries its `Run`: the output, the full history, a trace."""
    agent = scripted_agent(model=_ScriptedStreamingModel([StreamDelta(text="Hi")]))

    _, result = _streamed(agent, "hello")

    assert result.output == "Hi"
    assert agent.history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi", "tool_calls": None},
    ]
    assert {span.type for span in result.trace.spans} >= {"agent", "llm"}


def test_stream_persists_the_turn_to_the_session(tmp_path: Any) -> None:
    """`run_streamed(session=...)` saves the turn like `run` does."""
    from runa.session import SQLiteSession

    session = SQLiteSession("s1", db_path=tmp_path / "runa.db")
    agent = scripted_agent(model=_ScriptedStreamingModel([StreamDelta(text="Hi")]))

    _streamed(agent, "hello", session=session)

    assert asyncio.run(session.get_items()) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi", "tool_calls": None},
    ]


def test_stopping_a_stream_early_cancels_the_run() -> None:
    """Breaking out of the iterator cancels the underlying run instead of leaving it running."""
    cancelled = asyncio.Event()

    class _SlowModel:
        async def get_response(self, *args: Any, **kwargs: Any):  # noqa: ANN002, ANN003
            raise NotImplementedError

        async def stream_response(self, *args: Any, **kwargs: Any):  # noqa: ANN002, ANN003
            yield StreamDelta(text="first")
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            yield StreamDelta(text="never")

    agent = scripted_agent(model=_SlowModel())

    async def first_event_then_stop() -> None:
        stream = agent.run_streamed("hi")
        events = cast(AsyncGenerator[Any], aiter(stream))
        await anext(events)
        await events.aclose()
        await asyncio.wait_for(cancelled.wait(), timeout=1)

    asyncio.run(first_event_then_stop())


def test_gen_trace_id_returns_a_fresh_id_each_time() -> None:
    """Two calls to `gen_trace_id` never collide."""
    assert gen_trace_id() != gen_trace_id()


def test_a_completed_turn_records_each_item_once() -> None:
    """The history a turn leaves holds what it started from and what it produced, once each."""
    agent = scripted_agent(model=ScriptedModel([text_response("hi there")]))

    agent.run_sync("hi")

    assert agent.history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hi there", "tool_calls": None},
    ]


def _huge_usage() -> Usage:
    return Usage(input_tokens=200_001, output_tokens=1, total_tokens=200_002, requests=1)


def test_compact_drops_history_before_the_latest_user_message_once_over_budget() -> None:
    """`default_compactor` (what `compact=True` resolves to) trims older turns past 200k tokens."""
    agent = scripted_agent(
        model=ScriptedModel([text_response("ok", usage=_huge_usage())]),
        compact=default_compactor,
    )
    earlier = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer", "tool_calls": None},
    ]
    agent.history = list(earlier)

    result = agent.run_sync("new question")

    assert agent.history == [
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "ok", "tool_calls": None},
    ]
    spans = result.trace.spans
    compact_spans = [s for s in spans if s.type == "custom" and s.name == "compact"]
    assert compact_spans
    assert all(s.output == {"dropped": 2} for s in compact_spans)


def test_compact_keeps_a_first_turn_whose_user_message_is_the_very_first_item() -> None:
    """A user message at index 0 is a found cut point, not a missing one.

    `items[0:]` is everything, so a first turn is kept whole either way -- but only one of the two
    is the strategy saying so. `None` is reserved for "no user message to cut at".
    """
    items: list[Any] = [
        {"role": "user", "content": "one question"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "content": "ok", "tool_call_id": "c1"},
    ]

    assert default_compactor(items, 300_000) == items
    assert default_compactor([items[1], items[2]], 300_000) is None


def test_compact_cuts_at_the_latest_plain_text_user_turn() -> None:
    """Where a user turn is is `_items.latest_user_index`' to say, multimodal rule included.

    A parts-list turn is not a cut point, so a conversation whose newest turn carries an image
    keeps the plain-text turn before it rather than being cut into.
    """
    items: list[Any] = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer", "tool_calls": None},
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "new answer", "tool_calls": None},
        {"role": "user", "content": [{"type": "text", "text": "and this"}]},
    ]

    assert default_compactor(items, 300_000) == items[2:]


def test_compact_follows_context_size_not_cumulative_run_usage() -> None:
    """A long tool loop whose total usage passes 200k, but whose context never does, isn't cut."""

    @tool
    def step() -> str:
        """Take one step."""
        return "ok"

    usage = Usage(input_tokens=150_000, output_tokens=1, total_tokens=150_001, requests=1)
    call = tool_call_response("step", "{}")
    call.usage = usage
    agent = scripted_agent(
        tools=[step],
        model=ScriptedModel([call, text_response("done", usage=usage)]),
        compact=default_compactor,
    )
    earlier = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer", "tool_calls": None},
    ]
    agent.history = list(earlier)

    result = agent.run_sync("new question")

    assert result.usage.total_tokens > 200_000
    assert agent.history[:3] == [*earlier, {"role": "user", "content": "new question"}]
    assert not [s for s in result.trace.spans if s.type == "custom" and s.name == "compact"]


def test_compact_defaults_to_off() -> None:
    """With no `compactor`, history is left alone no matter how large usage gets."""
    agent = scripted_agent(model=ScriptedModel([text_response("ok", usage=_huge_usage())]))
    earlier = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer", "tool_calls": None},
    ]
    agent.history = list(earlier)

    result = agent.run_sync("new question")

    assert agent.history == [
        *earlier,
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "ok", "tool_calls": None},
    ]
    assert not [s for s in result.trace.spans if s.type == "custom" and s.name == "compact"]


def test_compact_shrinks_session_backed_history_too(tmp_path: Any) -> None:
    """A compactor replaces a session's stored history too, not just `agent.history`."""
    from runa.session import SQLiteSession

    session = SQLiteSession("s1", db_path=tmp_path / "runa.db")
    asyncio.run(
        session.add_items(
            [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer", "tool_calls": None},
            ]
        )
    )
    agent = scripted_agent(
        model=ScriptedModel([text_response("ok", usage=_huge_usage())]),
        compact=default_compactor,
    )

    agent.run_sync("new question", session=session)

    assert asyncio.run(session.get_items()) == [
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "ok", "tool_calls": None},
    ]


def test_compact_runs_whatever_compactor_the_shape_carries() -> None:
    """The loop runs the agent's resolved `compactor`, whatever strategy it is.

    An arbitrary one (here: drop the oldest single item, ignoring `usage_tokens` entirely) takes
    effect -- proof the built-in `default_compactor` isn't secretly still in charge. What
    `compact=True`/`compact=my_fn` resolve *to* is `Agent.__init__`'s job; see
    `tests/test_agent.py`.
    """
    calls: list[int] = []

    def drop_oldest_item(items: list[Any], usage_tokens: int) -> list[Any] | None:
        calls.append(usage_tokens)
        return items[1:] if len(items) > 1 else None

    agent = scripted_agent(model=ScriptedModel([text_response("ok")]), compact=drop_oldest_item)
    earlier = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer", "tool_calls": None},
    ]
    agent.history = list(earlier)

    agent.run_sync("new question")

    assert calls == [2, 2]  # consulted twice: mid-run on `items`, again on the history
    assert agent.history == [
        {"role": "assistant", "content": "old answer", "tool_calls": None},
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "ok", "tool_calls": None},
    ]


def test_compact_applies_a_rewrite_that_drops_nothing(tmp_path: Any) -> None:
    """A compactor that only rewrites items (truncating long tool output) still takes effect.

    Mid-run on what the model is sent, and at the end on what the session stores.
    """
    from runa.session import SQLiteSession

    session = SQLiteSession("s1", db_path=tmp_path / "runa.db")

    def truncate(items: list[Any], usage_tokens: int) -> list[Any]:
        return [{**item, "content": (item.get("content") or "")[:5]} for item in items]

    @tool
    def lookup() -> str:
        """Look something up."""
        return "a very long tool output"

    model = ScriptedModel([tool_call_response("lookup", "{}"), text_response("done")])
    agent = scripted_agent(tools=[lookup], model=model, compact=truncate)

    agent.run_sync("hello there", session=session)

    assert model.calls[1][0] == {"role": "user", "content": "hello"}
    assert [item["content"] for item in asyncio.run(session.get_items())] == [
        "hello",
        "",
        "a ver",
        "done",
    ]


class _FakeSource:
    """A duck-typed `agent.memory`/`agent.knowledge`: hands back `block`, records what it's told."""

    def __init__(self, block: str | None = None) -> None:
        self.block = block
        self.consulted: list[tuple[str, str | None]] = []
        self.told: list[tuple[str, Any, str | None, Any]] = []

    async def retrieve(self, query: str, run_context: Any) -> dict[str, Any] | None:
        self.consulted.append((query, run_context.user_id))
        return None if self.block is None else {"role": "system", "content": self.block}

    async def after_turn(self, query: str, output: Any, run_context: Any, model: Any) -> list[str]:
        self.told.append((query, output, run_context.user_id, model))
        return []

    async def ingest(self) -> int:
        return 0


def test_memory_is_consulted_before_the_turn_and_its_block_reaches_the_model() -> None:
    """`agent.memory.context` runs with the user's message, and its block reaches the model."""
    memory = _FakeSource("Relevant memories:\n- User prefers Japanese.")
    model = ScriptedModel([text_response("ok")])
    agent = scripted_agent(model=model, memory=memory)

    result = agent.run_sync("hi")

    assert memory.consulted == [("hi", None)]
    sent = [(item.get("role"), item.get("content")) for item in model.calls[0]]
    assert ("system", "Relevant memories:\n- User prefers Japanese.") in sent
    assert result.output == "ok"

    (agent_span,) = [s for s in result.trace.spans if s.type == "agent"]
    (retrieval_span,) = [s for s in result.trace.spans if s.type == "retrieval"]
    assert retrieval_span.name == "memory"
    assert retrieval_span.parent_id == agent_span.id
    assert retrieval_span.status == "ok"
    assert retrieval_span.output == {
        "role": "system",
        "content": "Relevant memories:\n- User prefers Japanese.",
    }


def test_a_resumed_run_extracts_memory_from_the_original_turn() -> None:
    """Memory extraction runs once the paused turn completes, with the user's original message."""

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Needs approval."""
        return "done"

    memory = _FakeSource()
    agent = scripted_agent(
        tools=[dangerous],
        memory=memory,
        model=ScriptedModel([tool_call_response("dangerous", "{}"), text_response("all done")]),
    )

    result = agent.run_sync("do it")
    assert memory.told == []

    state = result.to_state()
    state.approve(result.interruptions[0])
    agent.run_sync(state)

    assert [(query, output) for query, output, _, _ in memory.told] == [("do it", "all done")]


def test_retrieved_blocks_go_right_before_the_message_they_were_retrieved_for() -> None:
    """With an input list that doesn't end on the user's message, blocks still precede it.

    `Agent.run` always ends its input on the user's message, so this drives `_run_async` itself.
    """
    memory = _FakeSource("Relevant memories:\n- User prefers Japanese.")
    model = ScriptedModel([text_response("ok")])
    agent = stand_in_agent(model=model, memory=memory)
    note = {"role": "system", "content": "Answer briefly."}

    asyncio.run(_run_async(agent, [{"role": "user", "content": "hi"}, note]))

    assert model.calls[0] == [
        {"role": "system", "content": "Relevant memories:\n- User prefers Japanese."},
        {"role": "user", "content": "hi"},
        note,
    ]


def test_memory_with_no_block_injects_nothing() -> None:
    """A `None` block leaves the model's input exactly as it would be without memory."""
    memory = _FakeSource()
    model = ScriptedModel([text_response("ok")])
    agent = scripted_agent(model=model, memory=memory)

    agent.run_sync("hi")

    assert model.calls[0] == [{"role": "user", "content": "hi"}]


def test_memory_is_told_the_exchange_after_the_turn() -> None:
    """After a run, the runner hands memory the turn's exchange, for the run's user."""
    memory = _FakeSource()
    agent = scripted_agent(model=ScriptedModel([text_response("sure, noted")]), memory=memory)

    result = agent.run_sync("I prefer Japanese")

    assert [told[:3] for told in memory.told] == [("I prefer Japanese", "sure, noted", None)]

    (extraction_span,) = [s for s in result.trace.spans if s.type == "custom"]
    assert extraction_span.name == "memory"
    assert extraction_span.status == "ok"
    assert extraction_span.output == {"stored": 0}


class _AskingSource(_FakeSource):
    """A source whose `after_turn` makes one call of its own through the model it's handed."""

    def __init__(self, *, streamed: bool = False) -> None:
        super().__init__()
        self.streamed = streamed

    async def after_turn(self, query: str, output: Any, run_context: Any, model: Any) -> list[str]:
        request = ModelRequest(input=[{"role": "user", "content": "anything to keep?"}])
        if self.streamed:
            async for _delta in model.stream_response(request):
                pass
        else:
            await model.get_response(request)
        return []


def test_a_sources_model_call_is_one_of_the_runs() -> None:
    """The call memory makes after the answer is in the run's usage, trace and hooks.

    But not held to `max_tokens`: the answer was already finished when it was spent.
    """
    extraction = Usage(input_tokens=40, output_tokens=10, total_tokens=50, requests=1)
    model = ScriptedModel([text_response("ok"), text_response("{}", extraction)])
    log: list[str] = []
    agent = scripted_agent(model=model, memory=_AskingSource(), max_tokens=10)

    result = agent.run_sync("hi", hooks=_RecordingHooks(log, "run"))

    assert result.status == "completed"
    assert result.usage.total_tokens == 52
    assert result.usage.requests == 2
    assert log.count("run:llm_start:TestAgent") == 2
    assert log.count("run:llm_end:TestAgent") == 2

    (source_span,) = [s for s in result.trace.spans if s.type == "custom"]
    (extraction_span,) = [s for s in result.trace.spans if s.parent_id == source_span.id]
    assert extraction_span.type == "llm"
    assert extraction_span.output["usage"]["total_tokens"] == 50


def test_a_sources_streamed_model_call_is_one_of_the_runs() -> None:
    """Streamed through the model it's handed, a source's call is charged once the stream ends."""
    extraction = Usage(input_tokens=4, output_tokens=1, total_tokens=5, requests=1)
    model = _SequentialStreamingModel(
        [[StreamDelta(text="ok")], [StreamDelta(text="{}"), StreamDelta(usage=extraction)]]
    )
    agent = scripted_agent(model=model, memory=_AskingSource(streamed=True))

    _events, result = _streamed(agent, "hi")

    assert result.status == "completed"
    assert result.usage.total_tokens == 5
    llm_spans = [s for s in result.trace.spans if s.type == "llm"]
    assert [s.output["usage"]["total_tokens"] for s in llm_spans] == [0, 5]


def test_memory_is_scoped_by_the_runs_user_id_without_a_session() -> None:
    """`run(user_id=...)` scopes both the retrieval and the extraction call, no session needed."""
    memory = _FakeSource()
    agent = scripted_agent(
        model=ScriptedModel([text_response("ok"), text_response("ok")]), memory=memory
    )

    agent.run_sync("hi", user_id="u1")
    agent.run_sync("hey", user_id="u2")

    assert memory.consulted == [("hi", "u1"), ("hey", "u2")]
    assert [told[2] for told in memory.told] == ["u1", "u2"]


def test_a_tool_and_a_delegate_see_the_runs_user_on_their_run_context() -> None:
    """`RunContext.user_id` is the run's, for a tool and for a delegate's tools alike."""
    seen: list[str | None] = []

    @tool
    def whoami(ctx: Any) -> str:
        """Report the run's user."""
        seen.append(ctx.user_id)
        return "ok"

    helper = agent_class(
        "helper",
        ScriptedModel(
            [tool_call_response("whoami", "{}", call_id="call_3"), text_response("done")]
        ),
        tools=[whoami],
    )

    agent = scripted_agent(
        model=ScriptedModel(
            [
                tool_call_response("whoami", "{}"),
                tool_call_response("helper", '{"input": "go"}', call_id="call_2"),
                text_response("ok"),
            ]
        ),
        tools=[whoami],
        subagents=[helper.delegate],
    )
    result = agent.run_sync("hi", user_id="u1")

    assert result.status == "completed", result.error
    assert seen == ["u1", "u1"]


def test_trace_session_id_is_derived_from_the_session(tmp_path: Any) -> None:
    """A session-backed run's `Trace.session_id` is that session's id, for grouping in the UI."""
    from runa.session import SQLiteSession

    session = SQLiteSession("s1", db_path=tmp_path / "runa.db")
    agent = scripted_agent(model=ScriptedModel([text_response("ok")]))

    result = agent.run_sync("hi", session=session)

    assert result.trace.session_id == "s1"


def test_trace_session_id_is_none_without_a_session() -> None:
    """A one-off run with no `session=` leaves `Trace.session_id` unset."""
    agent = scripted_agent(model=ScriptedModel([text_response("ok")]))

    result = agent.run_sync("hi")

    assert result.trace.session_id is None


def test_memory_retrieval_failure_degrades_gracefully() -> None:
    """A backend that doesn't answer doesn't fail the run; the turn proceeds without memory."""

    class _BoomMemory(_FakeSource):
        async def retrieve(self, query: str, run_context: Any) -> dict[str, Any] | None:
            raise ServiceUnavailable("boom")

    agent = scripted_agent(model=ScriptedModel([text_response("ok")]), memory=_BoomMemory())

    result = agent.run_sync("hi")

    assert result.output == "ok"

    (retrieval_span,) = [s for s in result.trace.spans if s.type == "retrieval"]
    assert retrieval_span.status == "error"
    assert retrieval_span.error == "boom"


def test_memory_extraction_failure_degrades_gracefully() -> None:
    """A backend that doesn't answer doesn't fail the run; the final output is unaffected."""

    class _BoomMemory(_FakeSource):
        async def after_turn(self, *args: Any) -> list[str]:
            raise ServiceUnavailable("boom")

    agent = scripted_agent(model=ScriptedModel([text_response("ok")]), memory=_BoomMemory())

    result = agent.run_sync("hi")

    assert result.output == "ok"

    (extraction_span,) = [s for s in result.trace.spans if s.type == "custom"]
    assert extraction_span.status == "error"
    assert extraction_span.error == "boom"


def test_retrieval_does_not_swallow_a_framework_bug() -> None:
    """Fail-open covers a backend fault, not Runa's own code: the run ends, and says why.

    An empty memory block is indistinguishable from a corpus with nothing relevant in it, so a
    bug swallowed here would never be reported -- retrieval would just quietly stop working.
    Ending the run is what "not swallowed" means; the bug comes back as the `"error"` `Run` every
    unclassified failure does, naming the exception it came from.
    """

    class _BuggyKnowledge(_FakeSource):
        async def retrieve(self, query: str, run_context: Any) -> dict[str, Any] | None:
            raise AttributeError("'NoneType' object has no attribute 'embed'")

    agent = scripted_agent(model=ScriptedModel([text_response("ok")]), knowledge=_BuggyKnowledge())

    result = agent.run_sync("hi")

    assert result.status == "error"
    assert re.search(r"AttributeError: .*no attribute 'embed'", result.error or "")


def test_retrieval_ends_the_run_on_a_failure_it_was_not_told_to_skip() -> None:
    """Only `ServiceUnavailable` is skipped; any other exception from a source ends the run.

    A built-in type says nothing about whether a service is down or misconfigured, so a source
    that means "skip me" says so.
    """

    class _StaleKnowledge(_FakeSource):
        async def retrieve(self, query: str, run_context: Any) -> dict[str, Any] | None:
            raise RuntimeError("Event loop is closed")

    agent = scripted_agent(model=ScriptedModel([text_response("ok")]), knowledge=_StaleKnowledge())

    result = agent.run_sync("hi")

    assert result.status == "error"
    assert "RuntimeError: Event loop is closed" in (result.error or "")
    (retrieval_span,) = [s for s in result.trace.spans if s.type == "retrieval"]
    assert retrieval_span.status == "error"


def test_after_turn_ends_the_run_on_a_refused_key() -> None:
    """A rejected key is a `UserError`, so a run says so rather than quietly forgetting."""

    class _RefusedMemory(_FakeSource):
        async def after_turn(self, *args: Any) -> list[str]:
            raise UserError("model request failed with 401: invalid api key")

    agent = scripted_agent(model=ScriptedModel([text_response("ok")]), memory=_RefusedMemory())

    result = agent.run_sync("hi")

    assert result.status == "error"
    assert "401" in (result.error or "")


def test_retrieval_misconfiguration_ends_the_run_with_its_trace() -> None:
    """A `UserError` (no embeddings key, say) ends the run the way every other failure does.

    What the run owes it is the trace around the failure, retrieval's span included.
    """

    class _UnconfiguredMemory(_FakeSource):
        async def retrieve(self, query: str, run_context: Any) -> dict[str, Any] | None:
            raise UserError("OPENAI_API_KEY is not set.")

    agent = scripted_agent(model=ScriptedModel([text_response("ok")]), memory=_UnconfiguredMemory())

    result = agent.run_sync("hi")

    assert result.status == "error"
    assert "OPENAI_API_KEY" in (result.error or "")
    (retrieval_span,) = [s for s in result.trace.spans if s.type == "retrieval"]
    assert retrieval_span.status == "error"
    assert "OPENAI_API_KEY" in (retrieval_span.error or "")


def test_knowledge_is_consulted_before_the_turn_and_its_block_reaches_the_model() -> None:
    """`agent.knowledge.context` runs with the user's message, and its block reaches the model."""
    knowledge = _FakeSource("Relevant knowledge:\n- Refunds take 5 business days.")
    model = ScriptedModel([text_response("ok")])
    agent = scripted_agent(model=model, knowledge=knowledge)

    result = agent.run_sync("hi")

    assert knowledge.consulted == [("hi", None)]
    sent = [(item.get("role"), item.get("content")) for item in model.calls[0]]
    assert ("system", "Relevant knowledge:\n- Refunds take 5 business days.") in sent
    assert result.output == "ok"

    (retrieval_span,) = [s for s in result.trace.spans if s.type == "retrieval"]
    assert retrieval_span.name == "knowledge"
    assert retrieval_span.status == "ok"


def test_knowledge_with_no_block_injects_nothing() -> None:
    """A `None` block leaves the model's input exactly as it would be without knowledge."""
    knowledge = _FakeSource()
    model = ScriptedModel([text_response("ok")])
    agent = scripted_agent(model=model, knowledge=knowledge)

    agent.run_sync("hi")

    assert model.calls[0] == [{"role": "user", "content": "hi"}]


def test_knowledge_and_memory_can_both_inject_blocks_before_the_final_message() -> None:
    """Memory and knowledge blocks are both injected, in order, right before the user's message."""
    memory = _FakeSource("Relevant memories:\n- User prefers Japanese.")
    knowledge = _FakeSource("Relevant knowledge:\n- Refunds take 5 business days.")
    model = ScriptedModel([text_response("ok")])
    agent = scripted_agent(model=model, memory=memory, knowledge=knowledge)

    agent.run_sync("hi")

    contents = [item.get("content") for item in model.calls[0]]
    assert contents == [
        "Relevant memories:\n- User prefers Japanese.",
        "Relevant knowledge:\n- Refunds take 5 business days.",
        "hi",
    ]


def test_knowledge_retrieval_failure_degrades_gracefully() -> None:
    """A backend that doesn't answer doesn't fail the run; the turn proceeds without knowledge."""

    class _BoomKnowledge(_FakeSource):
        async def retrieve(self, query: str, run_context: Any) -> dict[str, Any] | None:
            raise ServiceUnavailable("boom")

    agent = scripted_agent(model=ScriptedModel([text_response("ok")]), knowledge=_BoomKnowledge())

    result = agent.run_sync("hi")

    assert result.output == "ok"


def test_knowledge_none_behaves_exactly_as_before() -> None:
    """`knowledge=None` runs unaffected: no lookup, no injection, no retrieval span."""
    agent = scripted_agent(model=ScriptedModel([text_response("ok")]), knowledge=None)

    result = agent.run_sync("hi")

    assert result.output == "ok"
    assert [s for s in result.trace.spans if s.type == "retrieval"] == []


def test_memory_none_behaves_exactly_as_before() -> None:
    """`memory=None` runs unaffected: no lookup, no injection, no retrieval span."""
    agent = scripted_agent(model=ScriptedModel([text_response("ok")]), memory=None)

    result = agent.run_sync("hi")

    assert result.output == "ok"
    assert [s for s in result.trace.spans if s.type == "retrieval"] == []


def test_mcp_server_tools_are_merged_in_and_callable() -> None:
    """A tool listed by an `mcp` entry is callable exactly like a `@tool` function."""
    from runa.tool import FunctionTool

    async def on_invoke_tool(ctx: Any, arguments_json: str, call_id: str) -> Any:
        return "42"

    mcp_tool = FunctionTool(
        name="answer",
        description="The answer.",
        params_json_schema={},
        on_invoke_tool=on_invoke_tool,
    )

    class _FakeMCPServer:
        async def list_tools(self) -> list[FunctionTool]:
            return [mcp_tool]

    agent = scripted_agent(
        mcp=[_FakeMCPServer()],
        model=ScriptedModel([tool_call_response("answer", "{}"), text_response("it's 42")]),
    )

    result = agent.run_sync("what's the answer?")

    assert result.output == "it's 42"
    (tool_span,) = [s for s in result.trace.spans if s.type == "tool"]
    assert tool_span.name == "answer"


class _RecordingHooks(Hooks[Any]):
    """Records every callback into a log, tagged with the scope the instance was installed at.

    One recorder for both scopes, because there is one `Hooks` class: the same subclass is
    passed as `hooks=` for the run and assigned to an agent's `hooks`, and only the `label`
    it was built with says which of the two a line came from.
    """

    def __init__(self, log: list[str], label: str) -> None:
        self.log = log
        self.label = label

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        self.log.append(f"{self.label}:agent_start:{agent.name}")

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        self.log.append(f"{self.label}:agent_end:{agent.name}")

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        self.log.append(f"{self.label}:handoff:{from_agent.name}->{to_agent.name}")

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        self.log.append(f"{self.label}:tool_start:{tool.name}")

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: object) -> None:
        self.log.append(f"{self.label}:tool_end:{tool.name}")

    async def on_llm_start(self, context: Any, agent: Any, prompt: Any, items: Any) -> None:
        self.log.append(f"{self.label}:llm_start:{agent.name}")

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        self.log.append(f"{self.label}:llm_end:{agent.name}")


def test_every_lifecycle_event_reaches_both_hook_scopes() -> None:
    """Each event fires on the run's hooks first, then on the agent's own `hooks`."""
    log: list[str] = []
    agent = scripted_agent(
        hooks=_RecordingHooks(log, "own"),
        model=ScriptedModel([text_response("hi there")]),
    )

    agent.run_sync("hi", hooks=_RecordingHooks(log, "run"))

    assert log == [
        "run:agent_start:TestAgent",
        "own:agent_start:TestAgent",
        "run:llm_start:TestAgent",
        "own:llm_start:TestAgent",
        "run:llm_end:TestAgent",
        "own:llm_end:TestAgent",
        "run:agent_end:TestAgent",
        "own:agent_end:TestAgent",
    ]


def test_tool_events_reach_both_hook_scopes() -> None:
    """A tool call fires `on_tool_start`/`on_tool_end` at the run scope and the agent's own."""
    log: list[str] = []

    @tool
    def lookup() -> str:
        """Look something up."""
        return "found"

    agent = scripted_agent(
        tools=[lookup],
        hooks=_RecordingHooks(log, "own"),
        model=ScriptedModel([tool_call_response("lookup", "{}"), text_response("done")]),
    )

    agent.run_sync("look", hooks=_RecordingHooks(log, "run"))

    assert [entry for entry in log if "tool_" in entry] == [
        "run:tool_start:lookup",
        "own:tool_start:lookup",
        "run:tool_end:lookup",
        "own:tool_end:lookup",
    ]


def test_agent_scoped_hooks_fire_only_for_their_own_agent_across_a_handoff() -> None:
    """Each agent's `hooks` see only that agent; the target's `on_handoff` names its source."""
    log: list[str] = []
    target = agent_class(
        "Target",
        ScriptedModel([text_response("handled by target")]),
        hooks=_RecordingHooks(log, "target"),
    )
    main = scripted_agent(
        name="Main",
        subagents=[target.handoff],
        hooks=_RecordingHooks(log, "main"),
        model=ScriptedModel([tool_call_response("transfer_to_target", "{}")]),
    )

    main.run_sync("transfer", hooks=_RecordingHooks(log, "run"))

    assert "run:handoff:Main->Target" in log
    assert "target:handoff:Main->Target" in log
    # `main` is never told about the handoff: an agent-scoped `on_handoff` notifies the target
    # only, and is told both ends of it in the same argument order the run scope reads.
    assert not any(entry.startswith("main:handoff") for entry in log)
    assert [entry for entry in log if entry.startswith("main:")] == [
        "main:agent_start:Main",
        "main:llm_start:Main",
        "main:llm_end:Main",
    ]
    assert [entry for entry in log if entry.startswith("target:")] == [
        "target:handoff:Main->Target",
        "target:agent_start:Target",
        "target:llm_start:Target",
        "target:llm_end:Target",
        "target:agent_end:Target",
    ]


def test_a_resumed_run_pairs_agent_start_with_agent_end() -> None:
    """Resuming fires `on_agent_start`, so the resumed half is not an unpaired `on_agent_end`."""
    log: list[str] = []

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        return "done"

    agent = scripted_agent(
        tools=[dangerous],
        hooks=_RecordingHooks(log, "own"),
        model=ScriptedModel([tool_call_response("dangerous", "{}"), text_response("all done")]),
    )

    result = agent.run_sync("do it", hooks=_RecordingHooks(log, "run"))
    state = result.to_state()
    state.approve(result.interruptions[0])
    log.clear()
    agent.run_sync(state, hooks=_RecordingHooks(log, "run"))

    assert [entry for entry in log if "agent_start" in entry] == [
        "run:agent_start:TestAgent",
        "own:agent_start:TestAgent",
    ]
    assert log[:2] == ["run:agent_start:TestAgent", "own:agent_start:TestAgent"]
    assert log[-2:] == ["run:agent_end:TestAgent", "own:agent_end:TestAgent"]


def test_an_agent_without_hooks_still_runs() -> None:
    """An agent's own `hooks` is opt-in: unset, only the run-scoped instance fires."""
    log: list[str] = []
    agent = scripted_agent(model=ScriptedModel([text_response("fine")]))

    result = agent.run_sync("hi", hooks=_RecordingHooks(log, "run"))

    assert result.output == "fine"
    assert all(entry.startswith("run:") for entry in log)
