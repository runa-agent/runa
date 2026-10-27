"""Shared helpers for the test suite."""

import asyncio
from collections.abc import Awaitable
from typing import Any

from runa._types import ModelResponse, RunContext, Usage
from runa.agent import Agent
from runa.run import Run, RunStatus
from runa.run_internal.agent_shape import AgentShape
from runa.run_internal.agent_shape import shape_of as shape_of_agent
from runa.run_internal.record import RunRecord, RunTree
from runa.tool import ToolCall
from runa.tracing import Trace
from runa.transcript import Transcript


def text_response(text: str, usage: Usage | None = None) -> ModelResponse:
    """A model reply that answers with `text` and calls no tools."""
    return ModelResponse(
        output=[{"role": "assistant", "content": text, "tool_calls": None}],
        usage=usage or Usage(input_tokens=1, output_tokens=1, total_tokens=2, requests=1),
    )


def tool_call_response(name: str, arguments: str, call_id: str = "call_1") -> ModelResponse:
    """A model reply that calls the tool `name` with the JSON `arguments`, and says nothing."""
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


class ScriptedModel:
    """A `Model` that answers each call with the next of `responses`, recording what it was sent.

    `calls` holds each request's input items, and `received_tools` the last request's tools.
    """

    def __init__(self, responses: list[ModelResponse]) -> None:
        """Answer with `responses`, in order, one per call."""
        self._responses = list(responses)
        self.calls: list[list[Any]] = []
        self.received_tools: list[Any] = []

    async def get_response(self, request: Any) -> ModelResponse:
        """Record `request`, then return the next scripted response."""
        self.calls.append(list(request.input))
        self.received_tools = request.tools
        return self._responses.pop(0)

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN002, ANN003
        """Scripted models answer unstreamed only."""
        raise NotImplementedError


class _TestAgent(Agent):
    name = "TestAgent"
    instructions = "be helpful"


def scripted_agent(model: Any, **settings: Any) -> Agent:
    """A real `Agent`, answering with `model` and configured by `settings`, for `agent.run`.

    `settings` are `Agent`'s own constructor keywords (`tools`, `guardrails`, `subagents`,
    `hooks`, `compact`, ...), resolved the way an application's are, so a test through it also
    covers the wiring between a declaration and the turn loop. The agent is named `TestAgent`
    unless `name=` says otherwise.
    """
    return _TestAgent(model=model, **settings)


def agent_class(name: str, model: Any, **settings: Any) -> type[Agent]:
    """An `Agent` subclass named `name` declaring `model` and `settings`, for a `subagents` entry.

    `subagents` lists classes, which `Agent.__init__` builds itself, so a handoff or delegate
    target is declared rather than constructed.
    """
    return type(
        name, (Agent,), {"name": name, "instructions": "be helpful", "model": model, **settings}
    )


def stand_in_agent(**attributes: Any) -> Agent:
    """A real `Agent` with `attributes` set on it, for a test that drives the turn loop directly.

    Only for what `Agent.run` cannot reach, such as an input list that doesn't end on the user's
    message; everything else goes through `scripted_agent` and `agent.run`. `attributes` are
    `Agent` setting names (`model`, `memory`); one the agent has no attribute for is refused
    rather than set and never read.
    """
    agent = _TestAgent()
    for name, value in attributes.items():
        assert hasattr(agent, name), f"Agent has no attribute {name!r}"
        setattr(agent, name, value)
    return agent


def shape_of(agent: Agent) -> AgentShape:
    """`agent` as a run resolves it: its guardrails by phase, its compactor, its MCP tools."""
    return asyncio.run(shape_of_agent(agent).with_mcp_tools())


class _StandInAgent(Agent):
    name = "stand_in"
    model = "gpt-5.4-nano"
    instructions = "Stands in for the agent of a run no test drives."


def finished_run(
    output: Any = "ok",
    *,
    status: RunStatus = "completed",
    error: str | None = None,
    trace: Trace | None = None,
    usage: Usage | None = None,
    tool_calls: tuple[ToolCall, ...] = (),
    transcript: Transcript | None = None,
) -> Run:
    """A finished `Run`, for a test that grades or renders one without driving the turn loop.

    A `Run` reads through to the record and tree the turn loop fills, and a test asserting on an
    eval metric or an evaluation page cares about one or two of their facts. Pass those; the
    rest stand in -- a run always has a trace, so an anonymous one when it doesn't matter.
    """
    agent = _StandInAgent()
    record = RunRecord(
        start=agent,
        agent=agent,
        transcript=transcript if transcript is not None else Transcript(),
        run_context=RunContext(usage=usage if usage is not None else Usage()),
        tool_calls=list(tool_calls),
    )
    tree = RunTree(
        trace=trace if trace is not None else Trace(id="trace_stub", name="stub", start_time=0.0)
    )
    return Run(output, status, error, record, tree)


def run[T](awaitable: Awaitable[T]) -> T:
    """Run any awaitable to completion, not just a coroutine.

    `asyncio.run` takes a coroutine specifically. Several things under test are declared as
    returning `Awaitable` (a tool's `on_invoke_tool`, a guardrail function), which is the right
    public contract and a type error to hand straight to `asyncio.run`. Awaiting
    it inside a real coroutine keeps the contract broad and the call site honest.
    """

    async def _await() -> Any:
        return await awaitable

    return asyncio.run(_await())
