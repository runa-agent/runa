"""Tests for the built-in `RunHooks`/`AgentHooks` implementations."""

import asyncio
import logging

import pytest

from runa import Agent, LoggingAgentHooks, LoggingRunHooks
from runa._types import ModelResponse, RunContextWrapper, Usage
from runa.lifecycle import AgentHooks, RunHooks
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


async def _run_all_run_hooks(hooks: RunHooks[None]) -> None:
    context: RunContextWrapper[None] = RunContextWrapper(context=None)
    researcher, translator = Researcher(), Translator()

    await hooks.on_agent_start(context, researcher)
    await hooks.on_agent_end(context, researcher, "final output")
    await hooks.on_handoff(context, researcher, translator)
    await hooks.on_tool_start(context, researcher, search)
    await hooks.on_tool_end(context, researcher, search, "tool result")
    await hooks.on_llm_start(context, researcher, "system prompt", [])
    await hooks.on_llm_end(context, researcher, _response)


async def _run_all_agent_hooks(hooks: AgentHooks[None]) -> None:
    context: RunContextWrapper[None] = RunContextWrapper(context=None)
    researcher, translator = Researcher(), Translator()

    await hooks.on_start(context, researcher)
    await hooks.on_end(context, researcher, "final output")
    await hooks.on_handoff(context, researcher, translator)
    await hooks.on_tool_start(context, researcher, search)
    await hooks.on_tool_end(context, researcher, search, "tool result")
    await hooks.on_llm_start(context, researcher, "system prompt", [])
    await hooks.on_llm_end(context, researcher, _response)


def test_run_hooks_log_every_callback(caplog: pytest.LogCaptureFixture) -> None:
    """Each `LoggingRunHooks` callback logs a message naming the agent(s) involved."""
    with caplog.at_level(logging.DEBUG, logger="runa"):
        asyncio.run(_run_all_run_hooks(LoggingRunHooks()))

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


def test_agent_hooks_log_every_callback(caplog: pytest.LogCaptureFixture) -> None:
    """Each `LoggingAgentHooks` callback logs a message naming the agent(s) involved."""
    with caplog.at_level(logging.DEBUG, logger="runa"):
        asyncio.run(_run_all_agent_hooks(LoggingAgentHooks()))

    messages = [r.getMessage() for r in caplog.records]
    assert messages == [
        "agent start: Researcher",
        "agent end: Researcher",
        "agent output: Researcher -> 'final output'",
        "handoff: Translator -> Researcher",
        "tool start: search (Researcher)",
        "tool end: search (Researcher)",
        "tool output: search -> 'tool result'",
        "llm start: Researcher",
        "llm end: Researcher",
    ]


def test_info_level_logging_carries_no_content(caplog: pytest.LogCaptureFixture) -> None:
    """At INFO the hooks name what happened and never log the agent's or a tool's output.

    The default hooks run in every production app, so INFO is the level that decides whether
    user data lands in stdout. Content belongs at DEBUG, behind the tracing policy.
    """
    with caplog.at_level(logging.INFO, logger="runa"):
        asyncio.run(_run_all_run_hooks(LoggingRunHooks()))

    messages = [r.getMessage() for r in caplog.records]
    assert not any("final output" in m or "tool result" in m for m in messages)
    assert "agent end: Researcher" in messages
    assert "tool end: search (Researcher)" in messages


def test_debug_content_obeys_the_tracing_policy(caplog: pytest.LogCaptureFixture) -> None:
    """`observe(capture_outputs=False)` silences the DEBUG content lines too, not just spans."""
    with observe(capture_outputs=False), caplog.at_level(logging.DEBUG, logger="runa"):
        asyncio.run(_run_all_run_hooks(LoggingRunHooks()))

    messages = [r.getMessage() for r in caplog.records]
    assert not any("final output" in m or "tool result" in m for m in messages)


def test_debug_content_is_redacted(caplog: pytest.LogCaptureFixture) -> None:
    """A `redactor` registered with `observe` scrubs the DEBUG content lines."""
    with (
        observe(redactor=lambda value: "[scrubbed]"),
        caplog.at_level(logging.DEBUG, logger="runa"),
    ):
        asyncio.run(_run_all_run_hooks(LoggingRunHooks()))

    messages = [r.getMessage() for r in caplog.records]
    assert "agent output: Researcher -> '[scrubbed]'" in messages
    assert not any("final output" in m for m in messages)


def test_agent_hooks_can_be_set_on_an_agent_class() -> None:
    """`hooks` is a plain `Agent` field, so `LoggingAgentHooks` can be assigned to it."""

    class Editor(Agent):
        name = "Editor"
        instructions = "You edit text."
        hooks = LoggingAgentHooks()

    assert isinstance(Editor().hooks, LoggingAgentHooks)
