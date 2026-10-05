"""Tests for `runa.project`: `find_agent_class`/`list_agents`."""

from pathlib import Path

import pytest

from runa.cli.generate import generate_agent
from runa.cli.new import scaffold_project
from runa.project import AgentNotFound, NotARunaProject, find_agent_class, list_agents, loaded_app


def _write_agent(project_dir: Path, filename: str, source: str) -> None:
    (project_dir / "app" / "agents" / filename).write_text(source)


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


def test_list_agents_finds_every_declared_agent(tmp_path: Path) -> None:
    """`list_agents` returns one `AgentInfo` per Agent subclass under `app/agents/`."""
    project_dir = scaffold_project("demo", root=tmp_path)
    generate_agent("SupportAgent", root=project_dir)

    infos = list_agents(root=project_dir)

    assert [info.name for info in infos] == ["support_agent"]
    assert infos[0].class_name == "SupportAgent"


def test_list_agents_reports_defaults_for_a_bare_agent(tmp_path: Path) -> None:
    """A freshly generated Agent has no tools/guardrails/subagents and memory/knowledge off."""
    project_dir = scaffold_project("demo", root=tmp_path)
    generate_agent("SupportAgent", root=project_dir)

    info = list_agents(root=project_dir)[0]

    assert info.tools == []
    assert info.guardrails == []
    assert info.subagents == []
    assert info.memory == "off"
    assert info.knowledge == "off"


def test_list_agents_returns_empty_list_when_no_agents_declared(tmp_path: Path) -> None:
    """`list_agents` returns `[]`, not an error, for a project with no Agent subclasses yet."""
    project_dir = scaffold_project("demo", root=tmp_path)

    assert list_agents(root=project_dir) == []


def test_list_agents_raises_outside_a_runa_project(tmp_path: Path) -> None:
    """`list_agents` raises `NotARunaProject` when `root` has no `app/agents/` directory."""
    with pytest.raises(NotARunaProject):
        list_agents(root=tmp_path)
