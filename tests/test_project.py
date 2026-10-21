"""Tests for `runa.project`: `find_agent_class`/`list_agents`/`use`."""

from pathlib import Path

import pytest

from runa import db
from runa.cli.generate import generate_agent
from runa.cli.new import scaffold_project
from runa.exceptions import UserError
from runa.project import (
    AgentNotFound,
    AppLoadError,
    NotARunaProject,
    find_agent_class,
    list_agents,
    loaded_app,
    use,
)

_COUNTING_MAIN = '''"""A `main.py` that records every time it is executed."""

from pathlib import Path

with (Path(__file__).parent / "loads.txt").open("a") as marker:
    marker.write("x")
'''


def _write_agent(project_dir: Path, filename: str, source: str) -> None:
    (project_dir / "app" / "agents" / filename).write_text(source)


def _loads(project_dir: Path) -> int:
    """How many times `_COUNTING_MAIN` has run in this project."""
    marker = project_dir / "loads.txt"
    return len(marker.read_text()) if marker.exists() else 0


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


def test_list_agents_names_guardrails_however_they_were_declared(tmp_path: Path) -> None:
    """Bound and bare `guardrails=` entries both list the predicate's name."""
    project_dir = scaffold_project("demo", root=tmp_path)
    _write_agent(
        project_dir,
        "guarded_agent.py",
        "from runa import Agent, guardrail\n\n\n"
        "@guardrail\n"
        "def block_empty(x: str) -> bool:\n"
        "    'Trip on an empty message.'\n"
        "    return not x\n\n\n"
        "class BoundAgent(Agent):\n"
        "    name = 'bound'\n"
        "    guardrails = [block_empty.input]\n\n\n"
        "class BareAgent(Agent):\n"
        "    name = 'bare'\n"
        "    guardrails = [block_empty]\n",
    )

    named = {info.name: info.guardrails for info in list_agents(root=project_dir)}

    assert named["bound"] == ["block_empty"]
    assert named["bare"] == ["block_empty"]


def test_list_agents_returns_empty_list_when_no_agents_declared(tmp_path: Path) -> None:
    """`list_agents` returns `[]`, not an error, for a project with no Agent subclasses yet."""
    project_dir = scaffold_project("demo", root=tmp_path)

    assert list_agents(root=project_dir) == []


def test_list_agents_raises_outside_a_runa_project(tmp_path: Path) -> None:
    """`list_agents` raises `NotARunaProject` when `root` has no `app/agents/` directory."""
    with pytest.raises(NotARunaProject):
        list_agents(root=tmp_path)


def test_list_agents_executes_main_once_however_often_it_is_called(tmp_path: Path) -> None:
    """The dashboard calls this once per page view, and `main.py` is not a page view.

    `runa ui`'s `/agents` route renders through `list_agents`, which used to enter a scoped
    `loaded_app` block: every view re-executed the developer's entry point, so an app that calls
    `tracing.add_exporter` there held one exporter per view and wrote every finished trace that
    many times. Loading is a fact now (`project.use`), so the import happens once and the listing
    is still read fresh each call.
    """
    project_dir = scaffold_project("demo", root=tmp_path)
    (project_dir / "main.py").write_text(_COUNTING_MAIN)
    generate_agent("SupportAgent", root=project_dir)

    first = list_agents(root=project_dir)
    second = list_agents(root=project_dir)

    assert [info.name for info in first] == ["support_agent"]
    assert second == first
    assert _loads(project_dir) == 1


def test_use_says_both_halves_of_one_fact(tmp_path: Path) -> None:
    """Saying which app this process reads places its database as well as its imports.

    The two used to be two calls, which meant a surface could say one: a process whose data root
    and import root disagree serves one project's agents off another project's database.
    """
    project_dir = scaffold_project("demo", root=tmp_path)
    generate_agent("SupportAgent", root=project_dir)

    use(project_dir)

    assert db._configured().sqlite_path() == project_dir / "db" / "runa.db"
    assert [info.name for info in list_agents(root=project_dir)] == ["support_agent"]


def test_loaded_app_says_both_halves_too(tmp_path: Path) -> None:
    """The scoped shape places the database as well, so a one-shot command says it once."""
    project_dir = scaffold_project("demo", root=tmp_path)

    with loaded_app(project_dir):
        assert db._configured().sqlite_path() == project_dir / "db" / "runa.db"


def test_use_imports_the_app_only_once_something_reads_it(tmp_path: Path) -> None:
    """Saying which app this process reads doesn't execute it; reading its agents does.

    `runa ui` says it while building the app and renders Sessions, Traces and Evaluations without
    ever importing the developer's code -- so a broken `main.py` costs the Agents page, not the
    whole dashboard. `runa serve` lists its agents at startup, so it still fails there.
    """
    project_dir = scaffold_project("demo", root=tmp_path)
    (project_dir / "main.py").write_text(_COUNTING_MAIN)
    generate_agent("SupportAgent", root=project_dir)

    use(project_dir)
    assert _loads(project_dir) == 0

    list_agents(root=project_dir)
    list_agents(root=project_dir)
    assert _loads(project_dir) == 1


def test_use_refuses_a_second_project(tmp_path: Path) -> None:
    """Two apps in one process would both want the `main`/`app.*` names, so saying so is the fix.

    The alternative is the quiet one: the loser's `sys.modules` entries are deleted underneath it
    and it serves the winner's agents. The message names both roots, since which two projects
    collided is the only thing the caller can act on -- and there is one message, because there
    is one root, read by `runa.db` and this module alike.
    """
    first = scaffold_project("first", root=tmp_path)
    second = scaffold_project("second", root=tmp_path)
    use(first)

    with pytest.raises(UserError) as exc_info:
        use(second)

    assert "first" in str(exc_info.value)
    assert "second" in str(exc_info.value)


def test_loaded_app_refuses_a_second_project_too(tmp_path: Path) -> None:
    """The scoped shape is refused on the way in, so the process still reads the first app.

    `loaded_app` keeps the root after its block and undoes only the import, so a block that could
    open on a *different* root would leave the process reading the second app's data with the
    first app's modules loaded. It says `use` on the way in, and that refuses it before anything
    moves.
    """
    first = scaffold_project("first", root=tmp_path)
    second = scaffold_project("second", root=tmp_path)
    generate_agent("SupportAgent", root=first)
    use(first)

    with pytest.raises(UserError), loaded_app(second):
        pass  # pragma: no cover -- the block is never entered

    assert db._configured().sqlite_path() == first / "db" / "runa.db"
    assert [info.name for info in list_agents(root=first)] == ["support_agent"]


def test_use_none_clears_the_whole_fact(tmp_path: Path) -> None:
    """`None` is the process boundary a second app really has, and it clears the whole fact."""
    first = scaffold_project("first", root=tmp_path)
    second = scaffold_project("second", root=tmp_path)
    generate_agent("SupportAgent", root=second)
    use(first)
    list_agents(root=first)

    use(None)
    use(second)

    assert db._configured().sqlite_path() == second / "db" / "runa.db"
    assert [info.name for info in list_agents(root=second)] == ["support_agent"]


def test_a_broken_main_can_be_retried(tmp_path: Path) -> None:
    """A `main.py` that raises leaves nothing loaded, so the next attempt is a first attempt.

    An `AppLoadError` is the developer's own entry point failing (see the class), which the
    dashboard renders as a 500 page. Recording that root as loaded anyway would turn a fixable
    import error into a process that can never load the app it was pointed at.
    """
    project_dir = scaffold_project("demo", root=tmp_path)
    (project_dir / "main.py").write_text("raise RuntimeError('boom')\n")
    generate_agent("SupportAgent", root=project_dir)

    with pytest.raises(AppLoadError):
        list_agents(root=project_dir)

    (project_dir / "main.py").write_text(_COUNTING_MAIN)
    list_agents(root=project_dir)

    assert _loads(project_dir) == 1
