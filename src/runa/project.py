"""`runa.project`: a Runa app on disk -- load it, find its agents, describe them.

Every surface that drives someone else's app needs the same three answers: is `root` a Runa
project, has its `main.py` been imported (so its `load_dotenv()`, or whatever else it does, has
run, same as `python main.py` would), and which `Agent` subclasses does it declare. `runa chat`,
`runa serve` and `runa ui` all ask, so the answers live here rather than in whichever surface
asked first. They used to live in `cli/_project.py`, which meant `runa.serve` and `runa.web`
importing a private module of the CLI for machinery that was never about argv.

Loading comes in exactly two shapes, and which one a surface wants follows from how long it
lives: `loaded_app(root)` is the scoped import a one-shot command makes and unmakes, and
`ensure_loaded(root)` is the process-global fact a server or a dashboard asks for once, however
many requests ask it. Nothing re-imports an app per request; see `ensure_loaded` for why.

Where a project's *data* lives is still not here: that is `runa.db`'s answer, given a `root` of
its own through `db.use_project`.

No `Agent` is ever instantiated: `list_agents` reads class attributes only, so listing an app's
agents never triggers memory/knowledge setup, prompt-file creation, or any API call.
"""

import importlib
import inspect
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from runa.agent import Agent, Subagent, _flatten_subagents
from runa.exceptions import OperatorError, UserError
from runa.guardrail import declared_names


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


def _import_main() -> None:
    """Import the `main` module off `sys.path`, the way `python main.py` would run it.

    A `ModuleNotFoundError` for `main` itself passes through untouched: `cli/main.py` already
    gives "no main.py found" its own clean message. Anything else the developer's own entry point
    raises becomes `AppLoadError`, which is the failure that isn't Runa's (see the class).
    """
    try:
        importlib.import_module("main")
    except ModuleNotFoundError as exc:
        if exc.name == "main":
            raise  # cli/main.py already gives this its own clean message
        raise AppLoadError(str(exc)) from exc
    except Exception as exc:
        raise AppLoadError(str(exc)) from exc


@contextmanager
def loaded_app(root: Path) -> Iterator[None]:
    """Import `root/main.py` for the block, then remove `root` from `sys.path`.

    The scoped load, for the one-shot commands that import an app, use what they found and exit:
    `runa chat`, `runa eval`, `runa test`. Each entry re-executes `main.py`, which is exactly what
    a process living longer than one command must not do -- those ask `ensure_loaded` instead.
    """
    root_str = str(root)
    _reset_project_modules()
    sys.path.insert(0, root_str)
    try:
        _import_main()
        yield
    finally:
        sys.path.remove(root_str)


_loaded_root: Path | None = None
"""The app `ensure_loaded` imported into this process, or `None` before the first one."""

_load_lock = threading.Lock()
"""Held across the check and the import, so concurrent first callers load the app once.

`runa ui`'s routes are plain `def`, which uvicorn runs in a thread pool: without this, two page
views arriving together would both find `_loaded_root` unset and both re-execute `main.py`, each
tearing down the other's half-imported `app.*` modules through `_reset_project_modules`.
"""


def ensure_loaded(root: Path) -> None:
    """Make sure this process has imported the app at `root`, and do nothing once it has.

    Loading an app is a process-global fact rather than something a caller does for itself: it
    executes the developer's `main.py` and imports their `app/`, so doing it a second time re-runs
    whatever that file set up -- a client, a pool, an exporter `tracing.add_exporter` appends to
    rather than replaces -- underneath everything already holding the first copy. Every surface
    that outlives one command therefore asks for the fact instead: `list_agents`, reached once per
    `runa ui` page view, and `serve.create_app` at startup.

    `root` stays on `sys.path` afterwards, the way `python main.py` leaves the script's directory
    there, so an `app/` module the app imports lazily still resolves on the hundredth request.

    A second, *different* `root` in one process is a `UserError`: both apps' modules would want
    the same `main`/`app.*` names, so the one that lost would be quietly serving the other's
    agents. `runa.db.use_project` refuses that for the same reason; a deployment that really
    fronts two apps runs two processes.
    """
    global _loaded_root
    root = root.resolve()
    with _load_lock:
        if _loaded_root == root:
            return
        if _loaded_root is not None:
            raise UserError(
                f"this process already loaded the Runa app at {_loaded_root}, so it cannot also "
                f"load the one at {root}: a process reads one app."
            )
        root_str = str(root)
        sys.path.insert(0, root_str)
        _reset_project_modules()
        try:
            _import_main()
        except BaseException:
            sys.path.remove(root_str)
            _reset_project_modules()
            raise
        _loaded_root = root


def reset_loaded_app() -> None:
    """Forget the app `ensure_loaded` imported, for a test that wants a clean process.

    `db.reset_ephemeral`'s counterpart for the import half of a project. Process-global is what
    `ensure_loaded` means, so a test pointing it at a `tmp_path` project stands in for the
    `create_app` that would have done it once at startup -- and the next test is a different
    directory. Nothing in a running app calls this: there, the second app is a second process.
    """
    global _loaded_root
    with _load_lock:
        if _loaded_root is not None:
            root_str = str(_loaded_root)
            if root_str in sys.path:
                sys.path.remove(root_str)
            _loaded_root = None
            _reset_project_modules()


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

    Each module is imported as `app.agents.<stem>`, so callers must have loaded the app first,
    through `ensure_loaded(root)` or a `loaded_app(root)` block. `obj.__module__ ==
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
        guardrails=declared_names(getattr(agent_cls, "guardrails", [])),
        subagents=[_subagent_label(sub) for sub in _flatten_subagents(subagents_raw)],
        memory=_retrieval_label(getattr(agent_cls, "memory", None)),
        knowledge=_retrieval_label(getattr(agent_cls, "knowledge", None)),
    )


def list_agents(*, root: Path) -> list[AgentInfo]:
    """Return every `Agent` subclass declared under `root/app/agents/`, alphabetically by name.

    `ensure_loaded`, not a `loaded_app` block: this is what a `runa ui` page view calls, and a
    scoped load would re-execute the developer's `main.py` once per view (see `ensure_loaded`).
    The listing itself is read fresh every time -- only the import is the once-per-process part.
    """
    agents_dir = require_agents_dir(root)
    ensure_loaded(root)
    infos = [_describe(agent_cls) for agent_cls in iter_agent_classes(agents_dir)]
    return sorted(infos, key=lambda info: info.name)


__all__ = [
    "AgentInfo",
    "AgentNotFound",
    "AppLoadError",
    "NotARunaProject",
    "ensure_loaded",
    "find_agent_class",
    "iter_agent_classes",
    "list_agents",
    "loaded_app",
    "require_agents_dir",
    "reset_loaded_app",
]
