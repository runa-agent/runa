"""Tests for `runa.cli.chat`: `find_agent_class`/`run_agent_repl`."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from runa.agent import Agent
from runa.cli._project import NotARunaProject, loaded_app
from runa.cli.chat import AgentNotFound, find_agent_class, run_agent_repl
from runa.cli.new import scaffold_project
from runa.run_state import Interruption
from runa.session import SQLiteSession


def _write_agent(project_dir: Path, filename: str, source: str) -> None:
    (project_dir / "app" / "agents" / filename).write_text(source)


class _SupportAgentStub(Agent):
    """A stand-in for an `Interruption.agent`; never actually run."""

    name = "SupportAgent"


def test_find_agent_class_matches_the_declared_name(tmp_path: Path) -> None:
    """`find_agent_class` finds an Agent subclass by its declared `name` attribute."""
    project_dir = scaffold_project("demo", root=tmp_path)
    _write_agent(
        project_dir,
        "support_agent.py",
        "from runa import Agent\n\n\nclass SupportAgent(Agent):\n    name = 'Support'\n",
    )

    with loaded_app(project_dir):
        agent_cls = find_agent_class("Support", agents_dir=project_dir / "app" / "agents")

    assert agent_cls.__name__ == "SupportAgent"


def test_find_agent_class_ignores_the_python_class_name(tmp_path: Path) -> None:
    """A class whose Python name differs from its declared `name` is still found by `name`."""
    project_dir = scaffold_project("demo", root=tmp_path)
    _write_agent(
        project_dir,
        "weird_agent.py",
        "from runa import Agent\n\n\nclass WeirdlyNamedClass(Agent):\n    name = 'Support'\n",
    )

    with loaded_app(project_dir):
        agent_cls = find_agent_class("Support", agents_dir=project_dir / "app" / "agents")

    assert agent_cls.__name__ == "WeirdlyNamedClass"


def test_find_agent_class_raises_when_nothing_matches(tmp_path: Path) -> None:
    """`find_agent_class` raises `AgentNotFound` when no Agent subclass matches."""
    project_dir = scaffold_project("demo", root=tmp_path)

    with pytest.raises(AgentNotFound):
        find_agent_class("Nope", agents_dir=project_dir / "app" / "agents")


def test_run_agent_repl_raises_outside_a_runa_project(tmp_path: Path) -> None:
    """`run_agent_repl` refuses to run where `app/agents/` doesn't exist."""
    with pytest.raises(NotARunaProject):
        run_agent_repl("Support", root=tmp_path)


@dataclass
class _FakeResult:
    """A stand-in for `Run`, just enough for `run_agent_repl` to consume."""

    output: Any = "the answer"
    interruptions: list[Any] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "paused" if self.interruptions else "completed"


def _scaffold_with_agent(tmp_path: Path) -> Path:
    project_dir = scaffold_project("demo", root=tmp_path)
    _write_agent(
        project_dir,
        "support_agent.py",
        "from runa import Agent\n\n\nclass SupportAgent(Agent):\n    name = 'Support'\n",
    )
    return project_dir


def _feed_input(monkeypatch: pytest.MonkeyPatch, lines: list[str]) -> None:
    """Make `input()` return each of `lines` in turn, then raise `EOFError`."""
    remaining = iter(lines)

    def fake_input(prompt: str = "") -> str:
        try:
            return next(remaining)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)


def test_run_agent_repl_sends_each_line_and_prints_the_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each REPL line becomes a turn, and its final output is printed before the next prompt."""
    project_dir = _scaffold_with_agent(tmp_path)
    _feed_input(monkeypatch, ["hi", "how are you"])
    replies = iter(["hello!", "doing fine"])

    def fake_run_sync(agent: Any, message: Any, **kwargs: Any) -> _FakeResult:
        return _FakeResult(output=next(replies))

    monkeypatch.setattr("runa.agent.Agent.run_sync", fake_run_sync)

    run_agent_repl("Support", root=project_dir)

    out = capsys.readouterr().out
    assert "hello!" in out
    assert "doing fine" in out


def test_run_agent_repl_starts_a_new_session_each_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no `--session`/`--continue`/`--resume`, each chat gets its own fresh session id."""
    project_dir = _scaffold_with_agent(tmp_path)
    session_ids: list[str] = []

    def fake_run_sync(agent: Any, message: Any, **kwargs: Any) -> _FakeResult:
        session_ids.append(kwargs["session"].session_id)
        return _FakeResult()

    monkeypatch.setattr("runa.agent.Agent.run_sync", fake_run_sync)

    _feed_input(monkeypatch, ["hi"])
    run_agent_repl("Support", root=project_dir)
    _feed_input(monkeypatch, ["hi again"])
    run_agent_repl("Support", root=project_dir)

    assert len(session_ids) == 2
    assert session_ids[0] != session_ids[1]
    assert all(session_id.startswith("Support-") for session_id in session_ids)


def test_run_agent_repl_continue_resumes_the_most_recent_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`continue_last=True` reuses the Agent's most recently updated session."""
    project_dir = _scaffold_with_agent(tmp_path)
    db_path = project_dir / "db" / "runa.db"
    asyncio.run(
        SQLiteSession("Support-old", db_path=db_path).add_items([{"role": "user", "content": "hi"}])
    )

    seen: list[str] = []

    def fake_run_sync(agent: Any, message: Any, **kwargs: Any) -> _FakeResult:
        seen.append(kwargs["session"].session_id)
        return _FakeResult()

    monkeypatch.setattr("runa.agent.Agent.run_sync", fake_run_sync)
    _feed_input(monkeypatch, ["hi"])

    run_agent_repl("Support", root=project_dir, continue_last=True)

    assert seen == ["Support-old"]


def test_run_agent_repl_continue_with_no_history_starts_a_new_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`continue_last=True` with no past session falls back to a fresh one."""
    project_dir = _scaffold_with_agent(tmp_path)
    seen: list[str] = []

    def fake_run_sync(agent: Any, message: Any, **kwargs: Any) -> _FakeResult:
        seen.append(kwargs["session"].session_id)
        return _FakeResult()

    monkeypatch.setattr("runa.agent.Agent.run_sync", fake_run_sync)
    _feed_input(monkeypatch, ["hi"])

    run_agent_repl("Support", root=project_dir, continue_last=True)

    assert seen[0].startswith("Support-")


def test_run_agent_repl_resume_with_an_id_uses_it_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`resume="some-id"` resumes that exact session, no prompt involved."""
    project_dir = _scaffold_with_agent(tmp_path)
    seen: list[str] = []

    def fake_run_sync(agent: Any, message: Any, **kwargs: Any) -> _FakeResult:
        seen.append(kwargs["session"].session_id)
        return _FakeResult()

    monkeypatch.setattr("runa.agent.Agent.run_sync", fake_run_sync)
    _feed_input(monkeypatch, ["hi"])

    run_agent_repl("Support", root=project_dir, resume="Support-custom")

    assert seen == ["Support-custom"]


def test_run_agent_repl_resume_without_an_id_prompts_a_picker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`resume=""` lists past sessions, newest first, and resumes the operator's pick."""
    project_dir = _scaffold_with_agent(tmp_path)
    db_path = project_dir / "db" / "runa.db"
    asyncio.run(
        SQLiteSession("Support-a", db_path=db_path).add_items([{"role": "user", "content": "hi"}])
    )
    asyncio.run(
        SQLiteSession("Support-b", db_path=db_path).add_items([{"role": "user", "content": "hi"}])
    )

    seen: list[str] = []

    def fake_run_sync(agent: Any, message: Any, **kwargs: Any) -> _FakeResult:
        seen.append(kwargs["session"].session_id)
        return _FakeResult()

    monkeypatch.setattr("runa.agent.Agent.run_sync", fake_run_sync)
    _feed_input(monkeypatch, ["", "hi"])  # blank picker answer -> default to the most recent

    run_agent_repl("Support", root=project_dir, resume="")

    assert seen == ["Support-b"]
    assert "Support-a" in capsys.readouterr().out


def test_run_agent_repl_resume_without_an_id_and_no_history_starts_a_new_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`resume=""` with no past sessions falls back to starting a fresh one."""
    project_dir = _scaffold_with_agent(tmp_path)
    seen: list[str] = []

    def fake_run_sync(agent: Any, message: Any, **kwargs: Any) -> _FakeResult:
        seen.append(kwargs["session"].session_id)
        return _FakeResult()

    monkeypatch.setattr("runa.agent.Agent.run_sync", fake_run_sync)
    _feed_input(monkeypatch, ["hi"])

    run_agent_repl("Support", root=project_dir, resume="")

    assert seen[0].startswith("Support-")


def test_run_agent_repl_exits_on_the_exit_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Typing `exit` ends the loop without running the agent."""
    project_dir = _scaffold_with_agent(tmp_path)
    _feed_input(monkeypatch, ["exit"])
    calls: list[Any] = []

    def fake_run_sync(agent: Any, message: Any, **kwargs: Any) -> _FakeResult:
        calls.append(message)
        return _FakeResult()

    monkeypatch.setattr("runa.agent.Agent.run_sync", fake_run_sync)

    run_agent_repl("Support", root=project_dir)

    assert calls == []


@dataclass
class _FakeApprovalState:
    """A stand-in for `RunState`, recording `approve`/`reject` calls."""

    approved: list[Any] = field(default_factory=list)
    rejected: list[Any] = field(default_factory=list)
    always_approved: list[Any] = field(default_factory=list)

    def approve(self, item: Any, *, always: bool = False) -> None:
        """Record `item` as approved, mirroring `RunState.approve()`."""
        self.approved.append(item)
        if always:
            self.always_approved.append(item)

    def reject(
        self, item: Any, *, always: bool = False, rejection_message: str | None = None
    ) -> None:
        """Record `item` as rejected, mirroring `RunState.reject()`."""
        self.rejected.append(item)


def test_run_agent_repl_approves_a_pending_tool_call_when_the_operator_says_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Answering `y` to the approval prompt resumes the run with the item approved."""
    project_dir = _scaffold_with_agent(tmp_path)
    agent = _SupportAgentStub()
    interruption = Interruption(
        name="delete_file", arguments="{}", call_id="call_1", tool=cast(Any, None), agent=agent
    )
    _feed_input(monkeypatch, ["delete it", "y"])
    state = _FakeApprovalState()

    @dataclass
    class _InterruptedResult(_FakeResult):
        output: Any = None

        def to_state(self) -> _FakeApprovalState:
            return state

    results = iter(
        [_InterruptedResult(interruptions=[interruption]), _InterruptedResult(output="done")]
    )

    def fake_run_sync(agent: Any, message: Any, **kwargs: Any) -> Any:
        return next(results)

    monkeypatch.setattr("runa.agent.Agent.run_sync", fake_run_sync)

    run_agent_repl("Support", root=project_dir)

    assert state.approved == [interruption]
    assert state.rejected == []


def test_run_agent_repl_rejects_a_pending_tool_call_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any answer other than `y` rejects the tool call rather than approving it."""
    project_dir = _scaffold_with_agent(tmp_path)
    agent = _SupportAgentStub()
    interruption = Interruption(
        name="delete_file", arguments="{}", call_id="call_1", tool=cast(Any, None), agent=agent
    )
    _feed_input(monkeypatch, ["delete it", "n"])
    state = _FakeApprovalState()

    @dataclass
    class _InterruptedResult(_FakeResult):
        output: Any = None

        def to_state(self) -> _FakeApprovalState:
            return state

    results = iter(
        [_InterruptedResult(interruptions=[interruption]), _InterruptedResult(output="done")]
    )

    def fake_run_sync(agent: Any, message: Any, **kwargs: Any) -> Any:
        return next(results)

    monkeypatch.setattr("runa.agent.Agent.run_sync", fake_run_sync)

    run_agent_repl("Support", root=project_dir)

    assert state.rejected == [interruption]
    assert state.approved == []


def test_run_agent_repl_always_approves_a_pending_tool_call_when_the_operator_says_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Answering `a` approves the item with `always=True`, not just a one-off approval."""
    project_dir = _scaffold_with_agent(tmp_path)
    agent = _SupportAgentStub()
    interruption = Interruption(
        name="delete_file", arguments="{}", call_id="call_1", tool=cast(Any, None), agent=agent
    )
    _feed_input(monkeypatch, ["delete it", "a"])
    state = _FakeApprovalState()

    @dataclass
    class _InterruptedResult(_FakeResult):
        output: Any = None

        def to_state(self) -> _FakeApprovalState:
            return state

    results = iter(
        [_InterruptedResult(interruptions=[interruption]), _InterruptedResult(output="done")]
    )

    def fake_run_sync(agent: Any, message: Any, **kwargs: Any) -> Any:
        return next(results)

    monkeypatch.setattr("runa.agent.Agent.run_sync", fake_run_sync)

    run_agent_repl("Support", root=project_dir)

    assert state.approved == [interruption]
    assert state.always_approved == [interruption]
    assert state.rejected == []
