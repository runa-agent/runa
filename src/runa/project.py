"""`runa.project`: a Runa app on disk -- load it, find its agents, describe them.

Every surface that drives someone else's app needs the same three answers: is `root` a Runa
project, has its `main.py` been imported (so its `load_dotenv()`, or whatever else it does, has
run, same as `python main.py` would), and which `Agent` subclasses does it declare. `runa chat`,
`runa serve` and `runa ui` all ask, so the answers live here rather than in whichever surface
asked first. They used to live in `cli/_project.py`, which meant `runa.serve` and `runa.web`
importing a private module of the CLI for machinery that was never about argv.

Where a project's *data* lives is still not here: that is `runa.db`'s answer, given a `root` of
its own through `db.use_project`.

No `Agent` is ever instantiated: `list_agents` reads class attributes only, so listing an app's
agents never triggers memory/knowledge setup, prompt-file creation, or any API call.
"""

import importlib
import inspect
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from runa.agent import Agent, Subagent, _flatten_subagents
from runa.exceptions import OperatorError


class NotARunaProject(OperatorError):
    """Raised when a caller needs a project subdirectory `root` doesn't have.

    Shared across every command and both web apps, so which directory check failed never changes
    how it is reported.
    """


class AppLoadError(Exception):
    """Raised when importing `root/main.py` raises, for any reason.

    A generated `main.py` typically does nothing but `load_dotenv()`, so this is most often a
    missing `.env` or a bug in the developer's own `main.py`/`app/` code, not a Runa bug.
    `cli/main.py` catches this and prints one clean line instead of a raw multi-frame traceback,
    and points at `python main.py` for the full one, since that traceback belongs to the
    developer's own entry point.

    Deliberately not an `OperatorError`: the operator typed nothing wrong, their app is broken,
    and the message that helps says that instead of the one every operator error shares.
    """


class AgentNotFound(OperatorError):
    """Raised when no Agent under `app/agents/` declares the given `name`."""


_PROJECT_MODULE_NAMES = ("main", "app", "tests", "evals")


def _reset_project_modules() -> None:
    """Drop cached `main`/`app`/`tests`/`evals` modules from a previous project's import.

    Each call may target a different project root, but Python caches imports by name in
    `sys.modules`; without this, a later call in the same process (e.g. across tests) would
    silently reuse a previous project's `main`/`app`/`tests`/`evals` instead of the one at `root`.
    """
    for name in list(sys.modules):
        if name in _PROJECT_MODULE_NAMES or name.startswith(
            tuple(f"{prefix}." for prefix in _PROJECT_MODULE_NAMES)
        ):
            del sys.modules[name]


@contextmanager
def loaded_app(root: Path) -> Iterator[None]:
    """Import `root/main.py` for the block, then remove `root` from `sys.path`."""
    root_str = str(root)
    _reset_project_modules()
    sys.path.insert(0, root_str)
    try:
        try:
            importlib.import_module("main")
        except ModuleNotFoundError as exc:
            if exc.name == "main":
                raise  # cli/main.py already gives this its own clean message
            raise AppLoadError(str(exc)) from exc
        except Exception as exc:
            raise AppLoadError(str(exc)) from exc
        yield
    finally:
        sys.path.remove(root_str)


def require_agents_dir(root: Path) -> Path:
    """Return `root/app/agents`, raising `NotARunaProject` if it doesn't exist.

    Every surface that resolves an Agent by name checks this first, so all of them give the same
    clean error outside a `runa new` project.
    """
    agents_dir = root / "app" / "agents"
    if not agents_dir.is_dir():
        raise NotARunaProject(
            f"{agents_dir} does not exist, run this from inside a Runa "
            "project created with `runa new`"
        )
    return agents_dir


def iter_agent_classes(agents_dir: Path) -> Iterator[type[Agent]]:
    """Yield every `Agent` subclass declared directly in a module under `agents_dir`.

    Each module is imported as `app.agents.<stem>`, so callers must run this inside
    `loaded_app(root)` (or otherwise have `root` on `sys.path`) first. `obj.__module__ ==
    module.__name__` excludes an `Agent` subclass merely imported into the module (e.g. a
    subagent imported for its `.handoff`/`.delegate` reference) from one actually defined there.
    """
    for agent_file in sorted(agents_dir.glob("*.py")):
        if agent_file.stem == "__init__":
            continue
        module = importlib.import_module(f"app.agents.{agent_file.stem}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Agent) and obj is not Agent and obj.__module__ == module.__name__:
                yield obj


def find_agent_class(agent_name: str, *, agents_dir: Path) -> type[Agent]:
    """Find the Agent subclass under `agents_dir` whose declared `name` is `agent_name`.

    Matches the `name` class attribute (e.g. `class SupportAgent(Agent): name = "Support"`),
    not the Python class name: `name` is the identity Runa itself uses for traces, instructions,
    and handoffs, so it's what an operator should type too.
    """
    for agent_cls in iter_agent_classes(agents_dir):
        if getattr(agent_cls, "name", None) == agent_name:
            return agent_cls
    raise AgentNotFound(f"no Agent named {agent_name!r} found under {agents_dir}")


@dataclass
class AgentInfo:
    """One declared `Agent` subclass, summarized for display."""

    name: str
    class_name: str
    model: str
    tools: list[str] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)
    subagents: list[str] = field(default_factory=list)
    memory: str = "off"
    knowledge: str = "off"


def _tool_name(tool: object) -> str:
    return getattr(tool, "name", None) or getattr(tool, "__name__", None) or str(tool)


def _guardrail_name(guardrail: object) -> str:
    name = getattr(guardrail, "name", None)
    if name:
        return str(name)
    function = getattr(guardrail, "guardrail_function", None)
    return getattr(function, "__name__", None) or str(guardrail)


def _subagent_label(subagent: Subagent | type[Agent]) -> str:
    """Label one flattened subagent entry, `Subagent`-wrapped or bare (auto mode).

    `_flatten_subagents` only wraps a bare `Agent` subclass in a `Subagent` when a dict-shaped
    `subagents` gave it an explicit `.handoff`/`.delegate` mode; unwrapped means "auto".
    """
    if isinstance(subagent, Subagent):
        agent_name = getattr(subagent.agent, "name", subagent.agent.__name__)
        return f"{agent_name} ({subagent.mode})"
    agent_name = getattr(subagent, "name", getattr(subagent, "__name__", str(subagent)))
    return f"{agent_name} (auto)"


def _retrieval_label(setting: object) -> str:
    if setting is None:
        return "off"
    return setting if isinstance(setting, str) else "on"


def _describe(agent_cls: type[Agent]) -> AgentInfo:
    """Summarize `agent_cls` for display: the settings worth a row on the Agents page.

    Deliberately a subset of `agent._AGENT_FIELDS`, and deliberately the declared values rather
    than the resolved ones (nothing here instantiates the class). A setting missing from this
    list configures the agent exactly as before; it just doesn't get a row, so this is not one of
    the paths a new setting has to join.
    """
    subagents_raw = getattr(agent_cls, "subagents", [])
    return AgentInfo(
        name=getattr(agent_cls, "name", agent_cls.__name__),
        class_name=agent_cls.__name__,
        model=getattr(agent_cls, "model", Agent.model),
        tools=[_tool_name(tool) for tool in getattr(agent_cls, "tools", [])],
        guardrails=[_guardrail_name(g) for g in getattr(agent_cls, "guardrails", [])],
        subagents=[_subagent_label(sub) for sub in _flatten_subagents(subagents_raw)],
        memory=_retrieval_label(getattr(agent_cls, "memory", None)),
        knowledge=_retrieval_label(getattr(agent_cls, "knowledge", None)),
    )


def list_agents(*, root: Path) -> list[AgentInfo]:
    """Return every `Agent` subclass declared under `root/app/agents/`, alphabetically by name."""
    agents_dir = require_agents_dir(root)
    with loaded_app(root):
        infos = [_describe(agent_cls) for agent_cls in iter_agent_classes(agents_dir)]
    return sorted(infos, key=lambda info: info.name)


__all__ = [
    "AgentInfo",
    "AgentNotFound",
    "AppLoadError",
    "NotARunaProject",
    "find_agent_class",
    "iter_agent_classes",
    "list_agents",
    "loaded_app",
    "require_agents_dir",
]
