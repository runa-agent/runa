"""Tests for the rule that one `Agent` instance runs one conversation at a time.

An instance carries `history`, so two overlapping session-less runs would read the same base
conversation and race to write it back, losing one into the other. In a server that is a data
leak between users, so it is refused rather than allowed to happen quietly.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from runa import Agent
from runa._types import ModelResponse, Usage
from runa.exceptions import UserError
from runa.session import SQLiteSession
from runa.tool import tool


def _final_message(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text, "tool_calls": None}


def _tool_call_message(call_id: str, name: str, arguments: str = "{}") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


class _YieldingModel:
    """Suspends before answering, so two runs genuinely overlap instead of running to completion.

    A model that never awaits would let `asyncio.gather` finish each run before starting the
    next, which would hide exactly the interleaving these tests are about.
    """

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = list(messages)

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:  # noqa: ANN002, ANN003
        await asyncio.sleep(0.01)
        return ModelResponse(output=[self._messages.pop(0)], usage=Usage(total_tokens=1))


def test_overlapping_session_less_runs_are_refused() -> None:
    """The second concurrent run on one instance raises instead of corrupting `history`."""

    class Solo(Agent):
        name = "Solo"
        instructions = "Answer."
        model = _YieldingModel([_final_message("a"), _final_message("b")])

    agent = Solo()

    async def both() -> Any:
        return await asyncio.gather(agent.run("one"), agent.run("two"))

    with pytest.raises(UserError, match="cannot run concurrently"):
        asyncio.run(both())


def test_the_error_names_both_ways_out() -> None:
    """The message has to be actionable: it names the session and the per-run instance."""

    class Solo(Agent):
        name = "Solo"
        instructions = "Answer."
        model = _YieldingModel([_final_message("a"), _final_message("b")])

    agent = Solo()

    async def both() -> Any:
        return await asyncio.gather(agent.run("one"), agent.run("two"))

    with pytest.raises(UserError) as caught:
        asyncio.run(both())

    assert "session=" in str(caught.value)
    assert "one Agent per run" in str(caught.value)


def test_concurrent_runs_with_a_session_are_allowed(tmp_path: Any) -> None:
    """With a `session` each run's history is its own, so overlapping is safe and permitted."""

    class Shared(Agent):
        name = "Shared"
        instructions = "Answer."
        model = _YieldingModel([_final_message("a"), _final_message("b")])

    agent = Shared()
    db = tmp_path / "runa.db"

    async def both() -> Any:
        return await asyncio.gather(
            agent.run("one", session=SQLiteSession("s1", db)),
            agent.run("two", session=SQLiteSession("s2", db)),
        )

    runs = asyncio.run(both())

    assert [r.status for r in runs] == ["completed", "completed"]
    assert agent.history == []


def test_one_instance_per_run_is_allowed() -> None:
    """The other way out: a fresh `Agent` per run, which shares no state at all."""

    class PerRun(Agent):
        name = "PerRun"
        instructions = "Answer."

    async def both() -> Any:
        first, second = PerRun(), PerRun()
        first.model = _YieldingModel([_final_message("a")])
        second.model = _YieldingModel([_final_message("b")])
        return await asyncio.gather(first.run("one"), second.run("two"))

    runs = asyncio.run(both())

    assert sorted(r.output for r in runs) == ["a", "b"]


def test_sequential_runs_on_one_instance_still_accumulate_history() -> None:
    """The latch is about overlap only; the normal conversational loop is untouched."""

    class Chatty(Agent):
        name = "Chatty"
        instructions = "Answer."
        model = _YieldingModel([_final_message("a"), _final_message("b")])

    agent = Chatty()
    asyncio.run(agent.run("one"))
    asyncio.run(agent.run("two"))

    assert [m.get("content") for m in agent.history if m.get("role") == "user"] == ["one", "two"]


def test_the_latch_is_released_after_a_failed_run() -> None:
    """A run that errors must not leave the instance permanently locked."""

    class Broken(Agent):
        name = "Broken"
        instructions = "Answer."
        max_turns = 1
        model = _YieldingModel([_tool_call_message("c1", "missing"), _final_message("ok")])

    agent = Broken()
    assert agent.run_sync("one").status == "error"
    assert agent.run_sync("two").status in {"completed", "error"}  # not a UserError about locking


def test_a_delegate_does_not_accumulate_history_across_calls() -> None:
    """Each delegation runs on a fresh copy, so `agent_as_tool`'s documented contract holds."""
    seen: list[int] = []

    class Worker(Agent):
        name = "Worker"
        instructions = "Work."

    worker = Worker()

    class _RecordingModel:
        def __init__(self, messages: list[dict[str, Any]]) -> None:
            self._messages = list(messages)

        async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:  # noqa: ANN002, ANN003
            seen.append(len(args[1]))  # how many items this nested run was handed
            return ModelResponse(output=[self._messages.pop(0)], usage=Usage(total_tokens=1))

    worker.model = _RecordingModel([_final_message("x"), _final_message("y")])

    class Boss(Agent):
        name = "Boss"
        instructions = "Delegate."
        model = _YieldingModel(
            [
                _tool_call_message("c1", "worker", '{"input": "first"}'),
                _tool_call_message("c2", "worker", '{"input": "second"}'),
                _final_message("done"),
            ]
        )

    boss = Boss()
    boss.tools = [*boss.tools, worker.as_tool(None, None)]
    run = boss.run_sync("go")

    assert run.status == "completed"
    assert seen == [1, 1]  # the second delegation saw one item, not the first one's leftovers
    assert worker.history == []


def test_a_tool_that_runs_the_same_agent_concurrently_is_refused() -> None:
    """The guard covers the realistic shape too: a shared agent reached from inside a tool."""

    class Inner(Agent):
        name = "Inner"
        instructions = "Answer."
        model = _YieldingModel([_final_message("a"), _final_message("b")])

    inner = Inner()

    @tool
    def ask(question: str) -> str:
        """Ask the inner agent.

        question: what to ask
        """
        return "unused"

    async def both() -> Any:
        return await asyncio.gather(inner.run("one"), inner.run("two"))

    with pytest.raises(UserError):
        asyncio.run(both())
