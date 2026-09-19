"""Tests for `runa.runner`: the in-house agent loop that replaces `agents.Runner`."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from runa._models import StreamDelta
from runa._types import ModelResponse, ModelSettings, Usage
from runa.exceptions import (
    ApprovalRequiredError,
    DuplicateToolCallError,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    OutputGuardrailTripwireTriggered,
    ToolInputGuardrailTripwireTriggered,
)
from runa.guardrail import GuardrailFunctionOutput, InputGuardrail, OutputGuardrail
from runa.handoff import Handoff
from runa.run_config import RunConfig
from runa.runner import Runner
from runa.tool import tool
from runa.tracing.util import gen_trace_id


def _agent(**overrides: Any) -> Any:
    defaults = dict(
        name="TestAgent",
        instructions="be helpful",
        model=None,
        tools=[],
        handoffs=[],
        input_guardrails=[],
        output_guardrails=[],
        output_type=None,
        model_settings=ModelSettings(),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _text_response(text: str, usage: Usage | None = None) -> ModelResponse:
    return ModelResponse(
        output=[{"role": "assistant", "content": text, "tool_calls": None}],
        usage=usage or Usage(input_tokens=1, output_tokens=1, total_tokens=2, requests=1),
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
        self.calls: list[list[Any]] = []
        self.received_handoffs: list[Any] = []

    async def get_response(
        self, system_instructions, input, model_settings, tools, output_schema, handoffs
    ):  # noqa: ANN001, ARG002
        self.calls.append(list(input))
        self.received_handoffs = handoffs
        return self._responses.pop(0)

    async def stream_response(self, *args: Any, **kwargs: Any):  # noqa: ANN001, ANN002, ANN003
        raise NotImplementedError


class _ScriptedStreamingModel:
    """A `Model` stand-in that yields pre-scripted `StreamDelta`s in order."""

    def __init__(self, deltas: list[StreamDelta]) -> None:
        self._deltas = deltas

    async def get_response(self, *args: Any, **kwargs: Any):  # noqa: ANN001, ANN002, ANN003
        raise NotImplementedError

    async def stream_response(
        self, system_instructions, input, model_settings, tools, output_schema, handoffs
    ):  # noqa: ANN001, ARG002
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


def _run_config() -> RunConfig:
    return RunConfig(workflow_name="TestAgent")


def test_plain_text_turn_returns_final_output() -> None:
    """A model reply with no tool calls becomes the run's final output directly."""
    agent = _agent(model=_ScriptedModel([_text_response("hello there")]))

    result = asyncio.run(Runner.run(agent, "hi", run_config=_run_config()))

    assert result.final_output == "hello there"
    assert result.interruptions == []
    assert result.context_wrapper.usage.input_tokens == 1


def test_run_sync_matches_run() -> None:
    """`run_sync` is a synchronous wrapper with identical behavior to `run`."""
    agent = _agent(model=_ScriptedModel([_text_response("ok")]))

    result = Runner.run_sync(agent, "hi", run_config=_run_config())

    assert result.final_output == "ok"


def test_tool_call_then_final_text() -> None:
    """A tool call is executed and its result fed back before the model gives a final answer."""

    @tool
    def now() -> str:
        """Return a fixed time."""
        return "2024-01-01"

    agent = _agent(
        tools=[now],
        model=_ScriptedModel(
            [_tool_call_response("now", "{}"), _text_response("it is 2024-01-01")]
        ),
    )

    result = asyncio.run(Runner.run(agent, "what time is it?", run_config=_run_config()))

    assert result.final_output == "it is 2024-01-01"
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

    agent = _agent(
        tools=[boom],
        model=_ScriptedModel([_tool_call_response("boom", "{}"), _text_response("handled it")]),
    )

    result = asyncio.run(Runner.run(agent, "go", run_config=_run_config()))

    assert result.final_output == "handled it"
    (tool_span,) = [s for s in result.trace.spans if s.type == "tool"]
    assert tool_span.status == "error"
    assert "boom" in (tool_span.error or "")


def test_handoff_switches_current_agent() -> None:
    """Calling a handoff's tool switches to the target agent for the rest of the run."""
    target = _agent(name="Target", model=_ScriptedModel([_text_response("handled by target")]))
    handoff = Handoff.from_agent(target)
    main = _agent(
        name="Main",
        handoffs=[handoff],
        model=_ScriptedModel([_tool_call_response(handoff.tool_name, "{}")]),
    )

    result = asyncio.run(Runner.run(main, "please transfer", run_config=_run_config()))

    assert result.final_output == "handled by target"
    handoff_spans = [s for s in result.trace.spans if s.type == "handoff"]
    assert len(handoff_spans) == 1


def test_delegate_call_is_traced_as_a_delegate_span_not_a_tool_span() -> None:
    """A `.delegate`-wrapped agent call traces as `type="delegate"`, unlike a plain tool call.

    Both run through the exact same tool-call plumbing (`tool_execution.py`), but a delegate call
    secretly runs a whole nested `Agent.run()` -- its own LLM call, in its own trace -- so it gets
    its own span type, the same way a handoff does despite also being a disguised tool call.
    """
    from runa.agent import Agent
    from runa.handoff import agent_as_tool

    class Researcher(Agent):
        name = "researcher"
        model = _ScriptedModel([_text_response("the policy is 30 days")])

    delegate_tool = agent_as_tool(Researcher(), None, None)
    caller = _agent(
        tools=[delegate_tool],
        model=_ScriptedModel(
            [
                _tool_call_response("researcher", '{"input": "what is the policy?"}'),
                _text_response("it's 30 days"),
            ]
        ),
    )

    result = asyncio.run(Runner.run(caller, "what's the policy?", run_config=_run_config()))

    assert result.final_output == "it's 30 days"
    (delegate_span,) = [s for s in result.trace.spans if s.name == "researcher"]
    assert delegate_span.type == "delegate"
    assert not [s for s in result.trace.spans if s.type == "tool"]


def test_bare_agent_handoff_is_normalized_before_reaching_the_model() -> None:
    """A raw `Agent` in `.handoffs` reaches the model wrapped as a `Handoff`, not as-is.

    `Agent.__init__` stores bare `Agent`s in `.handoffs` (see `agent.py`), but `_handoff_dict`
    (`_models/_base.py`) needs `.tool_name`/`.tool_description`, which a bare `Agent` doesn't have.
    """
    target = _agent(name="Target", model=_ScriptedModel([_text_response("handled by target")]))
    model = _ScriptedModel([_text_response("hi")])
    main = _agent(name="Main", handoffs=[target], model=model)

    asyncio.run(Runner.run(main, "hello", run_config=_run_config()))

    assert len(model.received_handoffs) == 1
    (received,) = model.received_handoffs
    assert isinstance(received, Handoff)
    assert received.tool_name == "transfer_to_target"


def test_input_guardrail_tripwire_halts_the_run() -> None:
    """A tripped input guardrail raises before the model is ever called."""

    async def _trip(ctx: Any, agent: Any, value: Any) -> GuardrailFunctionOutput:
        return GuardrailFunctionOutput(output_info="blocked", tripwire_triggered=True)

    agent = _agent(
        input_guardrails=[InputGuardrail(guardrail_function=_trip, name="block_all")],
        model=_ScriptedModel([_text_response("should not be reached")]),
    )

    with pytest.raises(InputGuardrailTripwireTriggered):
        asyncio.run(Runner.run(agent, "hi", run_config=_run_config()))


def test_output_guardrail_tripwire_halts_the_run() -> None:
    """A tripped output guardrail raises after the model responds, before returning."""

    async def _trip(ctx: Any, agent: Any, value: Any) -> GuardrailFunctionOutput:
        return GuardrailFunctionOutput(output_info="too long", tripwire_triggered=len(value) > 3)

    agent = _agent(
        output_guardrails=[OutputGuardrail(guardrail_function=_trip, name="block_long")],
        model=_ScriptedModel([_text_response("way too long")]),
    )

    with pytest.raises(OutputGuardrailTripwireTriggered):
        asyncio.run(Runner.run(agent, "hi", run_config=_run_config()))


def test_tool_input_guardrail_tripwire_halts_the_run() -> None:
    """A tripped tool input guardrail raises and aborts the run, unlike a plain tool exception."""
    from runa.guardrail import guardrail

    @guardrail
    def block_args(args: dict[str, Any]) -> bool:
        """Trip on any arguments."""
        return bool(args)

    @tool(guardrails=[block_args.input])
    def search(query: str) -> str:
        """Search for something."""
        return "results"

    agent = _agent(
        tools=[search],
        model=_ScriptedModel([_tool_call_response("search", '{"query": "x"}')]),
    )

    with pytest.raises(ToolInputGuardrailTripwireTriggered):
        asyncio.run(Runner.run(agent, "search for x", run_config=_run_config()))


def test_max_turns_exceeded() -> None:
    """A model that keeps calling tools forever eventually raises `MaxTurnsExceeded`."""

    @tool
    def loop_tool() -> str:
        """Always return the same thing."""
        return "again"

    responses = [_tool_call_response("loop_tool", "{}", call_id=f"call_{i}") for i in range(5)]
    agent = _agent(tools=[loop_tool], model=_ScriptedModel(responses))

    with pytest.raises(MaxTurnsExceeded):
        asyncio.run(Runner.run(agent, "go", run_config=RunConfig(workflow_name="x", max_turns=3)))


def test_needs_approval_pauses_then_resumes_on_approve() -> None:
    """A tool requiring approval pauses the run with an `Interruption`; approving resumes it."""

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        return "done"

    agent = _agent(
        tools=[dangerous],
        model=_ScriptedModel([_tool_call_response("dangerous", "{}"), _text_response("all done")]),
    )

    result = asyncio.run(Runner.run(agent, "do it", run_config=_run_config()))

    assert result.final_output is None
    assert len(result.interruptions) == 1
    interruption = result.interruptions[0]
    assert interruption.name == "dangerous"

    state = result.to_state()
    state.approve(interruption)
    resumed = asyncio.run(Runner.run(agent, state, run_config=_run_config()))

    assert resumed.final_output == "all done"


def test_needs_approval_rejected_feeds_back_and_continues() -> None:
    """Rejecting a paused tool call resumes with a rejection message, not the tool's result."""

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        return "should not run"

    agent = _agent(
        tools=[dangerous],
        model=_ScriptedModel(
            [_tool_call_response("dangerous", "{}"), _text_response("okay, skipped it")]
        ),
    )

    result = asyncio.run(Runner.run(agent, "do it", run_config=_run_config()))
    state = result.to_state()
    state.reject(result.interruptions[0])
    resumed = asyncio.run(Runner.run(agent, state, run_config=_run_config()))

    assert resumed.final_output == "okay, skipped it"


def test_needs_approval_always_approve_skips_future_prompts_for_the_same_tool() -> None:
    """Approving with `always=True` sticks; a later call to the same tool doesn't re-prompt."""

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        return "done"

    agent = _agent(
        tools=[dangerous],
        model=_ScriptedModel(
            [
                _tool_call_response("dangerous", "{}", call_id="call_1"),
                _tool_call_response("dangerous", "{}", call_id="call_2"),
                _text_response("all done"),
            ]
        ),
    )

    result = asyncio.run(Runner.run(agent, "do it", run_config=_run_config()))
    state = result.to_state()
    state.approve(result.interruptions[0], always=True)

    resumed = asyncio.run(Runner.run(agent, state, run_config=_run_config()))

    assert resumed.interruptions == []
    assert resumed.final_output == "all done"


def test_reject_with_a_custom_rejection_message_feeds_the_custom_text_back() -> None:
    """A custom `rejection_message` is fed back to the model instead of the default text."""

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        return "should not run"

    agent = _agent(
        tools=[dangerous],
        model=_ScriptedModel(
            [_tool_call_response("dangerous", "{}"), _text_response("okay, skipped it")]
        ),
    )

    result = asyncio.run(Runner.run(agent, "do it", run_config=_run_config()))
    state = result.to_state()
    state.reject(result.interruptions[0], rejection_message="not allowed today")
    resumed = asyncio.run(Runner.run(agent, state, run_config=_run_config()))

    tool_messages = [
        item["content"] for item in resumed.to_input_list() if item.get("role") == "tool"
    ]
    assert "not allowed today" in tool_messages


def test_needs_approval_always_reject_feeds_back_the_sticky_message_for_later_calls() -> None:
    """Rejecting with `always=True` + a message sticks; a later call reuses that message too."""

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        return "should not run"

    agent = _agent(
        tools=[dangerous],
        model=_ScriptedModel(
            [
                _tool_call_response("dangerous", "{}", call_id="call_1"),
                _tool_call_response("dangerous", "{}", call_id="call_2"),
                _text_response("noted twice"),
            ]
        ),
    )

    result = asyncio.run(Runner.run(agent, "do it", run_config=_run_config()))
    state = result.to_state()
    state.reject(result.interruptions[0], always=True, rejection_message="not allowed, ever")

    resumed = asyncio.run(Runner.run(agent, state, run_config=_run_config()))

    assert resumed.interruptions == []
    tool_messages = [
        item["content"] for item in resumed.to_input_list() if item.get("role") == "tool"
    ]
    assert tool_messages.count("not allowed, ever") == 2


def test_resuming_the_same_state_twice_raises_duplicate_call_id_error() -> None:
    """Resuming the same paused `RunState` a second time doesn't silently re-run the tool."""

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Do something that needs a human's OK."""
        return "done"

    agent = _agent(
        tools=[dangerous],
        model=_ScriptedModel([_tool_call_response("dangerous", "{}"), _text_response("all done")]),
    )

    result = asyncio.run(Runner.run(agent, "do it", run_config=_run_config()))
    state = result.to_state()
    state.approve(result.interruptions[0])

    asyncio.run(Runner.run(agent, state, run_config=_run_config()))

    with pytest.raises(DuplicateToolCallError):
        asyncio.run(Runner.run(agent, state, run_config=_run_config()))


def test_passing_input_guardrails_are_recorded_even_though_nothing_tripped() -> None:
    """A guardrail that runs and doesn't trip still shows up in `input_guardrail_results`."""

    async def _pass(ctx: Any, agent: Any, value: Any) -> GuardrailFunctionOutput:
        return GuardrailFunctionOutput(output_info="ok", tripwire_triggered=False)

    agent = _agent(
        input_guardrails=[InputGuardrail(guardrail_function=_pass, name="check")],
        model=_ScriptedModel([_text_response("hi")]),
    )

    result = asyncio.run(Runner.run(agent, "hi", run_config=_run_config()))

    assert len(result.input_guardrail_results) == 1
    assert result.input_guardrail_results[0].tripped is False


def test_a_tripped_input_guardrail_still_records_the_guardrails_that_passed_before_it() -> None:
    """Earlier passing guardrails aren't lost when a later one trips and raises."""

    async def _pass(ctx: Any, agent: Any, value: Any) -> GuardrailFunctionOutput:
        return GuardrailFunctionOutput(output_info="ok", tripwire_triggered=False)

    async def _trip(ctx: Any, agent: Any, value: Any) -> GuardrailFunctionOutput:
        return GuardrailFunctionOutput(output_info="blocked", tripwire_triggered=True)

    agent = _agent(
        input_guardrails=[
            InputGuardrail(guardrail_function=_pass, name="first"),
            InputGuardrail(guardrail_function=_trip, name="second"),
        ],
        model=_ScriptedModel([_text_response("should not be reached")]),
    )

    with pytest.raises(InputGuardrailTripwireTriggered) as exc_info:
        asyncio.run(Runner.run(agent, "hi", run_config=_run_config()))

    run_data = exc_info.value.run_data
    assert run_data is not None
    assert len(run_data.input_guardrail_results) == 2
    assert [r.tripped for r in run_data.input_guardrail_results] == [False, True]


def test_tool_input_guardrail_results_are_recorded_even_when_they_pass() -> None:
    """A tool input guardrail that passes still shows up in `tool_input_guardrail_results`."""
    from runa.guardrail import guardrail

    @guardrail
    def allow_all(args: dict[str, Any]) -> bool:
        """Never trip."""
        return False

    @tool(guardrails=[allow_all.input])
    def search(query: str) -> str:
        """Search for something."""
        return "results"

    agent = _agent(
        tools=[search],
        model=_ScriptedModel(
            [_tool_call_response("search", '{"query": "x"}'), _text_response("ok")]
        ),
    )

    result = asyncio.run(Runner.run(agent, "search for x", run_config=_run_config()))

    assert len(result.tool_input_guardrail_results) == 1
    assert result.tool_input_guardrail_results[0].tripped is False


def test_a_paused_run_states_guardrail_results_reflect_what_ran_before_the_pause() -> None:
    """A paused `RunState`'s guardrail results only include what ran before the interruption."""
    from runa.guardrail import guardrail

    @guardrail
    def allow_all(args: dict[str, Any]) -> bool:
        """Never trip."""
        return False

    @tool(guardrails=[allow_all.input], needs_approval=True)
    def dangerous(query: str) -> str:
        """Needs approval."""
        return "done"

    agent = _agent(
        tools=[dangerous],
        model=_ScriptedModel(
            [_tool_call_response("dangerous", '{"query": "x"}'), _text_response("done")]
        ),
    )

    result = asyncio.run(Runner.run(agent, "do it", run_config=_run_config()))

    # The tool input guardrail only runs once the call is actually executed (i.e. after
    # approval), so pausing on `needs_approval` records nothing yet.
    assert result.tool_input_guardrail_results == []

    state = result.to_state()
    state.approve(result.interruptions[0])
    resumed = asyncio.run(Runner.run(agent, state, run_config=_run_config()))

    assert len(resumed.tool_input_guardrail_results) == 1


def test_stream_response_yields_text_and_final_message() -> None:
    """Streaming a plain-text reply yields raw deltas, then a `message_output_created` item."""
    from runa.stream_events import RawResponsesStreamEvent, RunItemStreamEvent

    agent = _agent(
        model=_ScriptedStreamingModel([StreamDelta(text="Hi"), StreamDelta(text=" there")])
    )

    async def collect() -> list[Any]:
        return [event async for event in Runner.run_streamed(agent, "hi", run_config=_run_config())]

    events = asyncio.run(collect())

    raw = [e for e in events if isinstance(e, RawResponsesStreamEvent)]
    assert [d.data.text for d in raw] == ["Hi", " there"]
    (final,) = [
        e
        for e in events
        if isinstance(e, RunItemStreamEvent) and e.name == "message_output_created"
    ]
    assert final.item["content"] == "Hi there"


def test_stream_response_runs_a_sticky_approved_tool_without_raising() -> None:
    """A tool with a sticky `always=True` approval already on the context runs during streaming."""
    from runa.stream_events import RunItemStreamEvent

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Needs approval."""
        return "done"

    agent = _agent(
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

    result = Runner.run_streamed(agent, "do it", run_config=_run_config())
    result.context_wrapper.approval_ledger["dangerous"] = True

    async def collect() -> list[Any]:
        return [event async for event in result]

    events = asyncio.run(collect())

    tool_outputs = [
        e for e in events if isinstance(e, RunItemStreamEvent) and e.name == "tool_output"
    ]
    assert tool_outputs[0].item["content"] == "done"


def test_stream_response_raises_approval_required_for_an_unresolved_needs_approval_tool() -> None:
    """`run_streamed` can't pause for approval; it raises instead of silently bypassing the gate."""

    @tool(needs_approval=True)
    def dangerous() -> str:
        """Needs approval."""
        return "done"

    agent = _agent(
        tools=[dangerous],
        model=_SequentialStreamingModel(
            [
                [
                    StreamDelta(
                        tool_call_index=0, tool_call_id="call_1", tool_call_name="dangerous"
                    ),
                    StreamDelta(tool_call_index=0, tool_call_arguments="{}"),
                ]
            ]
        ),
    )

    async def collect() -> list[Any]:
        return [
            event async for event in Runner.run_streamed(agent, "do it", run_config=_run_config())
        ]

    with pytest.raises(ApprovalRequiredError):
        asyncio.run(collect())


def test_stream_response_raises_duplicate_call_id_error_on_a_replayed_call_id() -> None:
    """A replayed `call_id` across turns doesn't silently re-run the tool during streaming."""

    @tool
    def now() -> str:
        """Return a fixed time."""
        return "2024-01-01"

    same_call = [
        StreamDelta(tool_call_index=0, tool_call_id="call_1", tool_call_name="now"),
        StreamDelta(tool_call_index=0, tool_call_arguments="{}"),
    ]
    agent = _agent(tools=[now], model=_SequentialStreamingModel([same_call, list(same_call)]))

    async def collect() -> list[Any]:
        return [
            event
            async for event in Runner.run_streamed(
                agent, "what time is it?", run_config=_run_config()
            )
        ]

    with pytest.raises(DuplicateToolCallError):
        asyncio.run(collect())


def test_gen_trace_id_returns_a_fresh_id_each_time() -> None:
    """Two calls to `gen_trace_id` never collide."""
    assert gen_trace_id() != gen_trace_id()


def test_to_input_list_does_not_duplicate_generated_items() -> None:
    """`original_input` must stay independent of `items`, or the turn's own output doubles up.

    `_run_turns` appends every generated message straight onto the `items` list it's handed; if
    `original_input` aliased that same list (as it used to), `to_input_list` would return the
    turn's output twice -- once folded into `original_input`, once from `_generated_items`.
    """
    agent = _agent(model=_ScriptedModel([_text_response("hi there")]))

    result = asyncio.run(
        Runner.run(agent, [{"role": "user", "content": "hi"}], run_config=_run_config())
    )

    assert result.to_input_list() == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hi there", "tool_calls": None},
    ]


def _huge_usage() -> Usage:
    return Usage(input_tokens=200_001, output_tokens=1, total_tokens=200_002, requests=1)


def test_compact_drops_history_before_the_latest_user_message_once_over_budget() -> None:
    """`agent.compact=True` trims older turns once this run's usage crosses 200k tokens."""
    agent = _agent(model=_ScriptedModel([_text_response("ok", usage=_huge_usage())]), compact=True)
    history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer", "tool_calls": None},
        {"role": "user", "content": "new question"},
    ]

    result = asyncio.run(Runner.run(agent, history, run_config=_run_config()))

    assert result.to_input_list() == [
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "ok", "tool_calls": None},
    ]
    compact_spans = [s for s in result.trace.spans if s.type == "custom" and s.name == "compact"]
    assert compact_spans
    assert all(s.output == {"dropped": 2} for s in compact_spans)


def test_compact_defaults_to_off() -> None:
    """Without `compact=True`, history is left alone no matter how large usage gets."""
    agent = _agent(model=_ScriptedModel([_text_response("ok", usage=_huge_usage())]))
    history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer", "tool_calls": None},
        {"role": "user", "content": "new question"},
    ]

    result = asyncio.run(Runner.run(agent, history, run_config=_run_config()))

    assert result.to_input_list() == [
        *history,
        {"role": "assistant", "content": "ok", "tool_calls": None},
    ]
    assert not [s for s in result.trace.spans if s.type == "custom" and s.name == "compact"]


def test_compact_shrinks_session_backed_history_too(tmp_path: Any) -> None:
    """`compact=True` replaces a session's stored history too, not just `agent.history`."""
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
    agent = _agent(model=_ScriptedModel([_text_response("ok", usage=_huge_usage())]), compact=True)

    asyncio.run(Runner.run(agent, "new question", session=session, run_config=_run_config()))

    assert asyncio.run(session.get_items()) == [
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "ok", "tool_calls": None},
    ]


def test_compact_accepts_a_custom_compactor_callable() -> None:
    """`compact=` also accepts a plain `(items, tokens) -> items|None` callable, not just `True`.

    Its own arbitrary strategy (here: drop the oldest single item, ignoring `usage_tokens`
    entirely) takes effect -- proof the built-in `default_compactor` isn't secretly still in
    charge.
    """
    calls: list[int] = []

    def drop_oldest_item(items: list[Any], usage_tokens: int) -> list[Any] | None:
        calls.append(usage_tokens)
        return items[1:] if len(items) > 1 else None

    agent = _agent(model=_ScriptedModel([_text_response("ok")]), compact=drop_oldest_item)
    history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer", "tool_calls": None},
        {"role": "user", "content": "new question"},
    ]

    result = asyncio.run(Runner.run(agent, history, run_config=_run_config()))

    assert calls == [2, 2]  # consulted twice: mid-run on `items`, again on `original_input`
    assert result.to_input_list() == [
        {"role": "assistant", "content": "old answer", "tool_calls": None},
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "ok", "tool_calls": None},
    ]


class _MemoryMatchStub:
    """Just enough of `MemoryMatch` for `_memory_block` to format it -- no `runa.memory` import."""

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeMemory:
    """A duck-typed `agent.memory` stand-in: records what the runner calls it with."""

    def __init__(self, matches: list[Any] | None = None) -> None:
        self.matches = matches or []
        self.search_calls: list[tuple[str, str | None]] = []
        self.remembered: list[tuple[str, str | None, Any]] = []

    async def search(self, query: str, *, user_id: str | None = None, k: int = 5) -> list[Any]:
        self.search_calls.append((query, user_id))
        return self.matches

    async def remember_from_conversation(
        self, conversation: str, *, user_id: str | None, model: Any
    ) -> list[str]:
        self.remembered.append((conversation, user_id, model))
        return []


def test_memory_is_searched_before_the_turn_and_injected_as_a_labeled_block() -> None:
    """`agent.memory.search` runs with the user's message, and its matches reach the model."""
    memory = _FakeMemory(matches=[_MemoryMatchStub("User prefers Japanese.")])
    model = _ScriptedModel([_text_response("ok")])
    agent = _agent(model=model, memory=memory)

    result = asyncio.run(Runner.run(agent, "hi", run_config=_run_config()))

    assert memory.search_calls == [("hi", None)]
    sent = [(item.get("role"), item.get("content")) for item in model.calls[0]]
    assert ("system", "Relevant memories:\n- User prefers Japanese.") in sent
    assert result.final_output == "ok"

    (agent_span,) = [s for s in result.trace.spans if s.type == "agent"]
    (retrieval_span,) = [s for s in result.trace.spans if s.type == "retrieval"]
    assert retrieval_span.name == "memory"
    assert retrieval_span.parent_id == agent_span.id
    assert retrieval_span.status == "ok"
    assert retrieval_span.output == {"count": 1}


def test_memory_with_no_matches_injects_nothing() -> None:
    """An empty `search` result leaves the model's input exactly as it would be without memory."""
    memory = _FakeMemory(matches=[])
    model = _ScriptedModel([_text_response("ok")])
    agent = _agent(model=model, memory=memory)

    asyncio.run(Runner.run(agent, "hi", run_config=_run_config()))

    assert model.calls[0] == [{"role": "user", "content": "hi"}]


def test_memory_extraction_runs_after_the_turn_with_the_exchange_and_resolved_model() -> None:
    """After a run, the runner hands memory the turn's exchange and the agent's own model."""
    memory = _FakeMemory()
    model = _ScriptedModel([_text_response("sure, noted")])
    agent = _agent(model=model, memory=memory)

    result = asyncio.run(Runner.run(agent, "I prefer Japanese", run_config=_run_config()))

    assert len(memory.remembered) == 1
    conversation, user_id, resolved_model = memory.remembered[0]
    assert "I prefer Japanese" in conversation
    assert "sure, noted" in conversation
    assert user_id is None
    assert resolved_model is model

    (extraction_span,) = [s for s in result.trace.spans if s.type == "custom"]
    assert extraction_span.name == "memory"
    assert extraction_span.status == "ok"
    assert extraction_span.output == {"stored": 0}


def test_memory_user_id_is_derived_from_the_session(tmp_path: Any) -> None:
    """`Session(user_id=...)` scopes both the retrieval and the extraction call."""
    from runa.session import SQLiteSession

    memory = _FakeMemory()
    session = SQLiteSession("s1", db_path=tmp_path / "runa.db", user_id="u1")
    agent = _agent(model=_ScriptedModel([_text_response("ok")]), memory=memory)

    asyncio.run(Runner.run(agent, "hi", session=session, run_config=_run_config()))

    assert memory.search_calls == [("hi", "u1")]
    assert memory.remembered[0][1] == "u1"


def test_trace_session_id_is_derived_from_the_session(tmp_path: Any) -> None:
    """A session-backed run's `Trace.session_id` is that session's id, for grouping in the UI."""
    from runa.session import SQLiteSession

    session = SQLiteSession("s1", db_path=tmp_path / "runa.db")
    agent = _agent(model=_ScriptedModel([_text_response("ok")]))

    result = asyncio.run(Runner.run(agent, "hi", session=session, run_config=_run_config()))

    assert result.trace.session_id == "s1"


def test_trace_session_id_is_none_without_a_session() -> None:
    """A one-off run with no `session=` leaves `Trace.session_id` unset."""
    agent = _agent(model=_ScriptedModel([_text_response("ok")]))

    result = asyncio.run(Runner.run(agent, "hi", run_config=_run_config()))

    assert result.trace.session_id is None


def test_memory_retrieval_failure_degrades_gracefully() -> None:
    """A broken `memory.search` doesn't fail the run; the turn proceeds without memory."""

    class _BoomMemory:
        async def search(self, query: str, *, user_id: str | None = None, k: int = 5) -> list[Any]:
            raise RuntimeError("boom")

    agent = _agent(model=_ScriptedModel([_text_response("ok")]), memory=_BoomMemory())

    result = asyncio.run(Runner.run(agent, "hi", run_config=_run_config()))

    assert result.final_output == "ok"

    (retrieval_span,) = [s for s in result.trace.spans if s.type == "retrieval"]
    assert retrieval_span.status == "error"
    assert retrieval_span.error == "boom"


def test_memory_extraction_failure_degrades_gracefully() -> None:
    """A broken extraction step doesn't fail the run; the final output is unaffected."""

    class _BoomMemory:
        async def search(self, query: str, *, user_id: str | None = None, k: int = 5) -> list[Any]:
            return []

        async def remember_from_conversation(self, *args: Any, **kwargs: Any) -> list[str]:
            raise RuntimeError("boom")

    agent = _agent(model=_ScriptedModel([_text_response("ok")]), memory=_BoomMemory())

    result = asyncio.run(Runner.run(agent, "hi", run_config=_run_config()))

    assert result.final_output == "ok"

    (extraction_span,) = [s for s in result.trace.spans if s.type == "custom"]
    assert extraction_span.status == "error"
    assert extraction_span.error == "boom"


class _KnowledgeMatchStub:
    """Just enough of `KnowledgeMatch` for `_knowledge_block` to format -- no `runa.knowledge`."""

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeKnowledge:
    """A duck-typed `agent.knowledge` stand-in: records what the runner calls it with."""

    def __init__(self, matches: list[Any] | None = None) -> None:
        self.matches = matches or []
        self.search_calls: list[str] = []

    async def search(self, query: str, *, k: int = 5) -> list[Any]:
        self.search_calls.append(query)
        return self.matches


def test_knowledge_is_searched_before_the_turn_and_injected_as_a_labeled_block() -> None:
    """`agent.knowledge.search` runs with the user's message, and its matches reach the model."""
    knowledge = _FakeKnowledge(matches=[_KnowledgeMatchStub("Refunds take 5 business days.")])
    model = _ScriptedModel([_text_response("ok")])
    agent = _agent(model=model, knowledge=knowledge)

    result = asyncio.run(Runner.run(agent, "hi", run_config=_run_config()))

    assert knowledge.search_calls == ["hi"]
    sent = [(item.get("role"), item.get("content")) for item in model.calls[0]]
    assert ("system", "Relevant knowledge:\n- Refunds take 5 business days.") in sent
    assert result.final_output == "ok"

    (retrieval_span,) = [s for s in result.trace.spans if s.type == "retrieval"]
    assert retrieval_span.name == "knowledge"
    assert retrieval_span.status == "ok"
    assert retrieval_span.output == {"count": 1}


def test_knowledge_with_no_matches_injects_nothing() -> None:
    """An empty `search` result leaves the model's input exactly as it would be without it."""
    knowledge = _FakeKnowledge(matches=[])
    model = _ScriptedModel([_text_response("ok")])
    agent = _agent(model=model, knowledge=knowledge)

    asyncio.run(Runner.run(agent, "hi", run_config=_run_config()))

    assert model.calls[0] == [{"role": "user", "content": "hi"}]


def test_knowledge_and_memory_can_both_inject_blocks_before_the_final_message() -> None:
    """Memory and knowledge blocks are both injected, in order, right before the user's message."""
    memory = _FakeMemory(matches=[_MemoryMatchStub("User prefers Japanese.")])
    knowledge = _FakeKnowledge(matches=[_KnowledgeMatchStub("Refunds take 5 business days.")])
    model = _ScriptedModel([_text_response("ok")])
    agent = _agent(model=model, memory=memory, knowledge=knowledge)

    asyncio.run(Runner.run(agent, "hi", run_config=_run_config()))

    contents = [item.get("content") for item in model.calls[0]]
    assert contents == [
        "Relevant memories:\n- User prefers Japanese.",
        "Relevant knowledge:\n- Refunds take 5 business days.",
        "hi",
    ]


def test_knowledge_retrieval_failure_degrades_gracefully() -> None:
    """A broken `knowledge.search` doesn't fail the run; the turn proceeds without knowledge."""

    class _BoomKnowledge:
        async def search(self, query: str, *, k: int = 5) -> list[Any]:
            raise RuntimeError("boom")

    agent = _agent(model=_ScriptedModel([_text_response("ok")]), knowledge=_BoomKnowledge())

    result = asyncio.run(Runner.run(agent, "hi", run_config=_run_config()))

    assert result.final_output == "ok"


def test_agent_without_knowledge_behaves_exactly_as_before() -> None:
    """An agent with no `knowledge` attribute at all runs unaffected -- no lookup, no injection."""
    agent = _agent(model=_ScriptedModel([_text_response("ok")]))
    assert not hasattr(agent, "knowledge")

    result = asyncio.run(Runner.run(agent, "hi", run_config=_run_config()))

    assert result.final_output == "ok"


def test_agent_without_memory_behaves_exactly_as_before() -> None:
    """An agent with no `memory` attribute at all runs unaffected -- no lookup, no injection."""
    agent = _agent(model=_ScriptedModel([_text_response("ok")]))
    assert not hasattr(agent, "memory")

    result = asyncio.run(Runner.run(agent, "hi", run_config=_run_config()))

    assert result.final_output == "ok"


def test_mcp_server_tools_are_merged_in_and_callable() -> None:
    """A tool listed by an `mcp_servers` entry is callable exactly like a `@tool` function."""
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

    agent = _agent(
        mcp_servers=[_FakeMCPServer()],
        model=_ScriptedModel([_tool_call_response("answer", "{}"), _text_response("it's 42")]),
    )

    result = asyncio.run(Runner.run(agent, "what's the answer?", run_config=_run_config()))

    assert result.final_output == "it's 42"
    (tool_span,) = [s for s in result.trace.spans if s.type == "tool"]
    assert tool_span.name == "answer"
