"""Tests for the built-in `Hooks` implementation."""

import asyncio
import logging

import pytest

from runa import Agent, LoggingHooks
from runa._types import ModelResponse, RunContext, Usage
from runa.lifecycle import Hooks, _Dispatch
from runa.tool import tool as tool_decorator
from runa.tracing import observe


class Researcher(Agent):
    """An agent used across tests."""

    name = "Researcher"
    instructions = "You research topics."


class Translator(Agent):
    """Another agent used across tests."""

    name = "Translator"
    instructions = "You translate text."


@tool_decorator
def search(query: str) -> str:
    """Search for `query`."""
    return query


_response = ModelResponse(
    output=[],
    usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
    response_id=None,
)


async def _run_all_hooks(hooks: Hooks[None]) -> None:
    context: RunContext[None] = RunContext(context=None)
    researcher, translator = Researcher(), Translator()

    await hooks.on_agent_start(context, researcher)
    await hooks.on_agent_end(context, researcher, "final output")
    await hooks.on_handoff(context, researcher, translator)
    await hooks.on_tool_start(context, researcher, search)
    await hooks.on_tool_end(context, researcher, search, "tool result")
    await hooks.on_llm_start(context, researcher, "system prompt", [])
    await hooks.on_llm_end(context, researcher, _response)


def test_logging_hooks_log_every_callback(caplog: pytest.LogCaptureFixture) -> None:
    """Each `LoggingHooks` callback logs a message naming the agent(s) involved."""
    with caplog.at_level(logging.DEBUG, logger="runa"):
        asyncio.run(_run_all_hooks(LoggingHooks()))

    messages = [r.getMessage() for r in caplog.records]
    assert messages == [
        "agent start: Researcher",
        "agent end: Researcher",
        "agent output: Researcher -> 'final output'",
        "handoff: Researcher -> Translator",
        "tool start: search (Researcher)",
        "tool end: search (Researcher)",
        "tool output: search -> 'tool result'",
        "llm start: Researcher",
        "llm end: Researcher",
    ]


def test_dispatch_overrides_every_hooks_event() -> None:
    """An event added to `Hooks` without a `_Dispatch` override would silently fire nowhere."""
    events = {name for name in vars(Hooks) if name.startswith("on_")}
    assert sorted(events) == [
        "on_agent_end",
        "on_agent_start",
        "on_handoff",
        "on_llm_end",
        "on_llm_start",
        "on_tool_end",
        "on_tool_start",
    ]
    assert events <= set(vars(_Dispatch))


class _Recorder(Hooks[None]):
    def __init__(self, label: str, seen: list[str]) -> None:
        self.label, self.seen = label, seen

    async def on_agent_start(self, context: RunContext[None], agent: Agent) -> None:
        self.seen.append(f"{self.label}:start:{agent.name}")

    async def on_handoff(
        self, context: RunContext[None], from_agent: Agent, to_agent: Agent
    ) -> None:
        self.seen.append(f"{self.label}:handoff:{from_agent.name}->{to_agent.name}")


def test_dispatch_fires_the_run_scope_then_the_agents_own() -> None:
    """Run hooks fire first; a handoff notifies the target's own hooks, never the sender's."""
    seen: list[str] = []
    researcher, translator = Researcher(), Translator()
    researcher.hooks = _Recorder("researcher", seen)
    translator.hooks = _Recorder("translator", seen)
    dispatch = _Dispatch(_Recorder("run", seen))
    context: RunContext[None] = RunContext(context=None)

    async def fire() -> None:
        await dispatch.on_agent_start(context, researcher)
        await dispatch.on_handoff(context, researcher, translator)

    asyncio.run(fire())
    assert seen == [
        "run:start:Researcher",
        "researcher:start:Researcher",
        "run:handoff:Researcher->Translator",
        "translator:handoff:Researcher->Translator",
    ]


def test_info_level_logging_carries_no_content(caplog: pytest.LogCaptureFixture) -> None:
    """At INFO the hooks name what happened and never log the agent's or a tool's output.

    The default hooks run in every production app, so INFO is the level that decides whether
    user data lands in stdout. Content belongs at DEBUG, behind the tracing policy.
    """
    with caplog.at_level(logging.INFO, logger="runa"):
        asyncio.run(_run_all_hooks(LoggingHooks()))

    messages = [r.getMessage() for r in caplog.records]
    assert not any("final output" in m or "tool result" in m for m in messages)
    assert "agent end: Researcher" in messages
    assert "tool end: search (Researcher)" in messages


def test_debug_content_obeys_the_tracing_policy(caplog: pytest.LogCaptureFixture) -> None:
    """`capture_outputs=False` silences the DEBUG content lines too, not just spans."""
    with observe.scoped(capture_outputs=False), caplog.at_level(logging.DEBUG, logger="runa"):
        asyncio.run(_run_all_hooks(LoggingHooks()))

    messages = [r.getMessage() for r in caplog.records]
    assert not any("final output" in m or "tool result" in m for m in messages)


def test_debug_content_is_redacted(caplog: pytest.LogCaptureFixture) -> None:
    """A `redactor` registered with `observe` scrubs the DEBUG content lines."""
    with (
        observe.scoped(redactor=lambda value: "[scrubbed]"),
        caplog.at_level(logging.DEBUG, logger="runa"),
    ):
        asyncio.run(_run_all_hooks(LoggingHooks()))

    messages = [r.getMessage() for r in caplog.records]
    assert "agent output: Researcher -> '[scrubbed]'" in messages
    assert not any("final output" in m for m in messages)


def test_hooks_can_be_set_on_an_agent_class() -> None:
    """`hooks` is a plain `Agent` field, so the same `LoggingHooks` can be assigned to it.

    One class covers both scopes: what makes an instance agent-scoped is living here rather
    than being passed to `run(hooks=...)`.
    """

    class Editor(Agent):
        name = "Editor"
        instructions = "You edit text."
        hooks = LoggingHooks()

    assert isinstance(Editor().hooks, LoggingHooks)
