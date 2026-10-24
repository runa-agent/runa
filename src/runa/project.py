"""`runa.project`: a Runa app on disk -- load it, find its agents, describe them.

Every surface that drives someone else's app needs the same three answers: is `root` a Runa
project, has its `main.py` been imported (so its `load_dotenv()`, or whatever else it does, has
run, same as `python main.py` would), and which `Agent` subclasses does it declare. `runa chat`,
`runa serve` and `runa ui` all ask, so the answers live here rather than in whichever surface
asked first.

**Which app this process reads is one fact, and `use(root)` is how it is said.** The fact itself
lives in `runa._app`, a leaf this module and `runa.db` both read: `runa.db` resolves the app's
state against it, and this module imports the app's `main.py` from it. It is one value, so the two
can never name different apps. A second, different app is refused there, once, in one message.

`use(root)` states the fact; the app's `main.py` is imported the first time anything reads its
modules, and once per process however many callers ask. That is what
lets `runa serve` fail at startup, where it lists the agents it is about to serve, while `runa ui`
still renders Sessions and Evaluations for an app whose `main.py` is broken -- both surfaces say
the same one thing, and the import lands where their own work puts it.

`loaded_app(root)` is the scoped variant, for the one-shot commands that import an app, use what
they found and exit: it says `use(root)` too, and undoes the import on the way out. Which shape a
surface wants follows from how long it lives; nothing re-imports an app per request.

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

from runa import _app
from runa._app import use
from runa.agent import Agent, Subagent, snake_case
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


def _ensure_imported() -> None:
    """Import the app `use` named, the first time anything reads its modules.

    Importing an app is a process-global fact rather than something a caller does for itself: it
    executes the developer's `main.py` and imports their `app/`, so doing it a second time re-runs
    whatever that file set up -- a client, a pool, an exporter `tracing.add_exporter` appends to
    rather than replaces -- underneath everything already holding the first copy. So the fact is
    kept in `runa._app`, and `iter_agent_classes` below asks for it on behalf of every caller that
    reads an app's agents: `runa ui` reaches it once per page view, `runa serve` once at startup,
    and both get the one import.

    `root` stays on `sys.path` afterwards, the way `python main.py` leaves the script's directory
    there, so an `app/` module the app imports lazily still resolves on the hundredth request.
    A `main.py` that raises records nothing, so the developer who fixes it and reloads the page
    gets a first attempt rather than a process that can never load the app it was pointed at.

    Inside a `loaded_app` block the import has already happened and this does nothing, which is
    the same answer it gives when no app has been said at all (`root is None`): then there is
    nothing to import and the caller is reading a directory it put on `sys.path` itself.
    """
    with _app.lock:
        root = _app.root
        if _app.imported or root is None:
            return
        sys.path.insert(0, _app.import_path(root))
        _app.reset_modules()
        try:
            _import_main()
        except BaseException:
            _app.unimport(root)
            raise
        _app.imported = True


@contextmanager
def loaded_app(root: Path) -> Iterator[None]:
    """Say `use(root)` and import `root/main.py` now, undoing the import when the block ends.

    The scoped shape of the same fact, for the one-shot commands that import an app, use what
    they found and exit: `runa chat`, `runa eval`, `runa test`. Each entry re-executes `main.py`,
    which is exactly what a process living longer than one command must not do -- those say
    `use` and let the import happen once.

    The root outlives the block, the way it outlives the command, so a `Session` or a `Memory`
    built inside still resolves against this project when a `finally:` elsewhere writes to it.
    Only the import is the block's own, and on the way out it goes back to what was true before.
    A process that had already imported this app (a block opened inside a `create_app`'s
    process) still has that import afterwards, so it is still
    that process's to undo. A block on a *different* root is refused by `use` before anything
    moves.
    """
    use(root)
    entry = _app.import_path(root)
    was_imported = _app.imported
    _app.reset_modules()
    sys.path.insert(0, entry)
    try:
        _import_main()
        _app.imported = True
        yield
    finally:
        _app.imported = was_imported
        sys.path.remove(entry)


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


def iter_agent_classes(agents_dir: Path) -> list[type[Agent]]:
    """Every `Agent` subclass declared directly in a module under `agents_dir`.

    Each module is imported as `app.agents.<stem>`, which is why this is where the app said by
    `use` is imported, on behalf of every caller that reads its agents (see `_ensure_imported`);
    inside a `loaded_app` block the import has already happened. `obj.__module__ ==
    module.__name__` excludes an `Agent` subclass merely imported into the module (e.g. a
    subagent imported for its `.handoff`/`.delegate` reference) from one actually defined there.

    Every class found is an agent the app can run, so each must declare its `name`: one that
    doesn't is a `UserError` here, before any surface lists, serves or builds it. A shared base
    class with no `name` of its own belongs outside `app/agents/`.
    """
    _ensure_imported()
    found: list[type[Agent]] = []
    for agent_file in sorted(agents_dir.glob("*.py")):
        if agent_file.stem == "__init__":
            continue
        module = importlib.import_module(f"app.agents.{agent_file.stem}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Agent) and obj is not Agent and obj.__module__ == module.__name__:
                found.append(obj)
    for agent_cls in found:
        if not hasattr(agent_cls, "name"):
            raise UserError(
                f"{agent_cls.__name__} in {agents_dir} declares no 'name'; give it one, e.g. "
                f"`name = {snake_case(agent_cls.__name__)!r}`, or move a shared base class "
                f"out of app/agents/"
            )
    return found


def find_agent_class(agent_name: str, *, agents_dir: Path) -> type[Agent]:
    """Find the Agent subclass under `agents_dir` whose declared `name` is `agent_name`.

    Matches the `name` class attribute (e.g. `class SupportAgent(Agent): name = "Support"`)
    rather than the Python class name: `name` is the identity Runa itself uses for traces,
    instructions, and handoffs, so it's what an operator should type too.
    """
    for agent_cls in iter_agent_classes(agents_dir):
        if agent_cls.name == agent_name:
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
    """Label one `subagents` entry: `.handoff`/`.delegate`-bound, or bare, which means auto."""
    if isinstance(subagent, Subagent):
        return f"{subagent.agent.name} ({subagent.mode})"
    return f"{subagent.name} (auto)"


def _retrieval_label(setting: object) -> str:
    if setting is None:
        return "off"
    return setting if isinstance(setting, str) else "on"


def _describe(agent_cls: type[Agent]) -> AgentInfo:
    """Summarize `agent_cls` for display: the settings worth a row on the Agents page.

    Deliberately a subset of `Agent`'s settings, and deliberately the declared values rather
    than the resolved ones (nothing here instantiates the class). A setting missing from this
    list configures the agent exactly as before; it just doesn't get a row, so this is not one of
    the paths a new setting has to join.
    """
    return AgentInfo(
        name=agent_cls.name,
        class_name=agent_cls.__name__,
        model=agent_cls.model,
        tools=[_tool_name(tool) for tool in getattr(agent_cls, "tools", [])],
        guardrails=declared_names(getattr(agent_cls, "guardrails", [])),
        subagents=[_subagent_label(sub) for sub in getattr(agent_cls, "subagents", [])],
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
