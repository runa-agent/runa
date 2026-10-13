"""`runa.project`: a Runa app on disk -- load it, find its agents, describe them.

Every surface that drives someone else's app needs the same three answers: is `root` a Runa
project, has its `main.py` been imported (so its `load_dotenv()`, or whatever else it does, has
run, same as `python main.py` would), and which `Agent` subclasses does it declare. `runa chat`,
`runa serve` and `runa ui` all ask, so the answers live here rather than in whichever surface
asked first. They used to live in `cli/_project.py`, which meant `runa.serve` and `runa.web`
importing a private module of the CLI for machinery that was never about argv.

**Which app this process reads is one fact, and `use(root)` is how it is said.** It has two
halves -- the directory this app's state resolves against, which is `runa.db`'s answer, and the
directory its `main.py` was imported from, which is this module's -- and a process whose halves
disagree serves one project's agents off another project's database, the same split-in-two
failure `runa.db` was centralized to prevent, one level further out. So no surface that fronts an
app says half of it: `use` hands the root to `db.use_project` and records it here, and a second,
different app is refused there, once, in one message. `use(None)` clears the whole fact again,
standing in for the process boundary a deployment has (see `tests/conftest.py`). `db.use_project`
alone is the data root alone, which is the honest answer for exactly one caller: `runa`, pointing
at `cwd` before it knows whether the command it dispatches reads an app at all.

That is also the direction the dependency goes, and the only direction it can: this module
imports `runa.db` for the half it doesn't own, so `runa.db` must never import this one. The two
globals stay where they are, one per module; what is shared is this entry point, not the state.

`use(root)` states the fact; the app's `main.py` is imported the first time anything reads its
modules, and once per process however many callers ask (see `_ensure_imported`). That is what
lets `runa serve` fail at startup, where it lists the agents it is about to serve, while `runa ui`
still renders Sessions and Evaluations for an app whose `main.py` is broken -- both surfaces say
the same one thing, and the import lands where their own work puts it rather than each ordering
the two halves itself.

`loaded_app(root)` is the scoped variant of that fact, for the one-shot commands that import an
app, use what they found and exit: it states both halves too, and unmakes the import on the way
out. Which shape a surface wants follows from how long it lives; nothing re-imports an app per
request.

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

from runa import db
from runa.agent import Agent, Subagent, _flatten_subagents
from runa.exceptions import OperatorError
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


_root: Path | None = None
"""The app this process reads, or `None` while it reads none. Written by `use`/`loaded_app`."""

_imported = False
"""Whether `_root`'s `main.py` has run in this process yet; see `_ensure_imported`."""

_load_lock = threading.Lock()
"""Held across the check and the import, so concurrent first callers load the app once.

`runa ui`'s routes are plain `def`, which uvicorn runs in a thread pool: without this, two page
views arriving together would both find `_imported` false and both re-execute `main.py`, each
tearing down the other's half-imported `app.*` modules through `_reset_project_modules`.
"""


def use(root: Path | None) -> None:
    """Point this process at the Runa app in `root`; `None` means it reads no app.

    Startup configuration, said once, by every surface that fronts someone else's app: `runa
    serve` and `runa ui` at build time, `runa chat`/`eval`/`test` through the `loaded_app` block
    below. Saying it covers both halves of the one fact -- `runa.db` resolves this app's state
    against `root`, and this app's `main.py` is the one that gets imported -- because a surface
    able to say only one of them is a process whose data root and import root can disagree (see
    this module's docstring).

    A second, *different* `root` in one process is a `UserError`, raised by `db.use_project`:
    both apps' modules would want the same `main`/`app.*` names, so the one that lost would be
    quietly serving the other's agents off the winner's database. The refusal lives there rather
    than here because the root reaches `runa.db` either way -- through this function or, for a
    command that reads an app's data without importing it, on its own -- and one fact deserves
    one message. A deployment that really fronts two apps runs two processes.

    `None` is how a process goes back to reading no app at all: the import is undone along with
    the fact, so the next `use` is a first `use`. Nothing in a running app calls it -- there, the
    second app is a second process -- but a test pointing this at a `tmp_path` project stands in
    for the `create_app` that would have said it once at startup, and the next test is a
    different directory.
    """
    global _root, _imported
    db.use_project(root)
    root = root.resolve() if root is not None else None
    with _load_lock:
        if root == _root:
            return
        previous, was_imported = _root, _imported
        _root, _imported = root, False
        if previous is not None and was_imported:
            _unimport(previous)


def _ensure_imported() -> None:
    """Import the app `use` named, the first time anything reads its modules.

    Importing an app is a process-global fact rather than something a caller does for itself: it
    executes the developer's `main.py` and imports their `app/`, so doing it a second time re-runs
    whatever that file set up -- a client, a pool, an exporter `tracing.add_exporter` appends to
    rather than replaces -- underneath everything already holding the first copy. So this is a
    fact the module keeps, and `iter_agent_classes` below asks for it on behalf of every caller
    that reads an app's agents: `runa ui` reaches it once per page view, `runa serve` once at
    startup, and both get the one import.

    `root` stays on `sys.path` afterwards, the way `python main.py` leaves the script's directory
    there, so an `app/` module the app imports lazily still resolves on the hundredth request.
    A `main.py` that raises records nothing, so the developer who fixes it and reloads the page
    gets a first attempt rather than a process that can never load the app it was pointed at.

    Inside a `loaded_app` block the import has already happened and this does nothing, which is
    the same answer it gives when no app has been said at all (`_root is None`): then there is
    nothing to import and the caller is reading a directory it put on `sys.path` itself.
    """
    global _imported
    with _load_lock:
        if _imported or _root is None:
            return
        root_str = str(_root)
        sys.path.insert(0, root_str)
        _reset_project_modules()
        try:
            _import_main()
        except BaseException:
            _unimport(_root)
            raise
        _imported = True


def _unimport(root: Path) -> None:
    """Undo `root`'s import: drop its `sys.path` entry and the modules it brought in."""
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    _reset_project_modules()


@contextmanager
def loaded_app(root: Path) -> Iterator[None]:
    """Say `use(root)` for the length of the block, importing `root/main.py` now.

    The scoped shape of the same fact, for the one-shot commands that import an app, use what
    they found and exit: `runa chat`, `runa eval`, `runa test`. Each entry re-executes `main.py`,
    which is exactly what a process living longer than one command must not do -- those say
    `use` and let the import happen once.

    The data half outlives the block, the way it outlives the command: a `Session` or a `Memory`
    built inside is still resolving against this project when a `finally:` elsewhere writes to it.
    What the block restores on the way out is what was true before it, not nothing -- a process
    that had already said `use` is still reading that app afterwards, so its one import is still
    its own to undo. That is how a test opens a block inside a `create_app`'s process (see
    `tests/serve/test_serve.py`) without stranding the import `create_app` made.
    """
    global _root, _imported
    db.use_project(root)
    root_str = str(root)
    previous = (_root, _imported)
    _reset_project_modules()
    sys.path.insert(0, root_str)
    try:
        _import_main()
        _root, _imported = root.resolve(), True
        yield
    finally:
        _root, _imported = previous
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

    Each module is imported as `app.agents.<stem>`, which is why this is where the app said by
    `use` is imported, on behalf of every caller that reads its agents (see `_ensure_imported`);
    inside a `loaded_app` block the import has already happened. `obj.__module__ ==
    module.__name__` excludes an `Agent` subclass merely imported into the module (e.g. a
    subagent imported for its `.handoff`/`.delegate` reference) from one actually defined there.
    """
    _ensure_imported()
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

    `use(root)`, not a `loaded_app` block: this is what a `runa ui` page view calls, and a scoped
    load would re-execute the developer's `main.py` once per view (see `_ensure_imported`). The
    listing itself is read fresh every time -- only the import is the once-per-process part.
    """
    agents_dir = require_agents_dir(root)
    use(root)
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
    "use",
]
