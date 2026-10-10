"""Tests for the built-in `Hooks` implementation."""

import asyncio
import logging

import pytest

from runa import Agent, LoggingHooks
from runa._types import ModelResponse, RunContextWrapper, Usage
from runa.lifecycle import _EVENTS, Hooks, _Dispatch
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
    context: RunContextWrapper[None] = RunContextWrapper(context=None)
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


def test_hooks_declares_one_name_per_event() -> None:
    """One class, one spelling per event: `_Dispatch` forwards over these names and no others.

    The split that used to exist (`on_agent_start` against an agent-scoped `on_start`) is what
    made the dispatcher restate every event, so the set is pinned rather than left to drift.
    """
    assert sorted(_EVENTS) == [
        "on_agent_end",
        "on_agent_start",
        "on_handoff",
        "on_llm_end",
        "on_llm_start",
        "on_tool_end",
        "on_tool_start",
    ]
    assert all(hasattr(_Dispatch(Hooks()), event) for event in _EVENTS)


def test_dispatch_refuses_an_event_hooks_does_not_declare() -> None:
    """A name that isn't a `Hooks` event raises rather than returning a callable firing nowhere."""
    with pytest.raises(AttributeError):
        _ = _Dispatch(Hooks()).on_tool_strt


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
    """`observe(capture_outputs=False)` silences the DEBUG content lines too, not just spans."""
    with observe(capture_outputs=False), caplog.at_level(logging.DEBUG, logger="runa"):
        asyncio.run(_run_all_hooks(LoggingHooks()))

    messages = [r.getMessage() for r in caplog.records]
    assert not any("final output" in m or "tool result" in m for m in messages)


def test_debug_content_is_redacted(caplog: pytest.LogCaptureFixture) -> None:
    """A `redactor` registered with `observe` scrubs the DEBUG content lines."""
    with (
        observe(redactor=lambda value: "[scrubbed]"),
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
