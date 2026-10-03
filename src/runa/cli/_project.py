"""cli/_project.py: shared machinery for CLI commands that load a Runa app.

`run`, `eval`, and `test` all need `root/main.py` imported (so its
`load_dotenv()`, or whatever else it does, runs, same as `python main.py`
would) before they can do anything; factored out so no command duplicates
the sys.path / sys.modules bookkeeping.
"""

import importlib
import inspect
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from runa.agent import Agent


class NotARunaProject(Exception):
    """Raised when a command needs a project subdirectory `root` doesn't have.

    Shared across `generate.py`, `run.py`, `eval.py`, and `test.py` so `cli/main.py` can catch
    it once, regardless of which command's directory check failed.
    """


def resolve_db_path(root: Path) -> Path:
    """The project's `db/runa.db`, creating `db/` first if it isn't there yet.

    Centralizes the convention `cli/new.py` scaffolds, so `chat.py`, `sessions.py`, and
    `traces.py` all agree on where a project's SQLite data lives, and it still works if `db/`
    was never committed (it's gitignored) or was deleted.
    """
    db_dir = root / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "runa.db"


class AppLoadError(Exception):
    """Raised when importing `root/main.py` raises, for any reason.

    A generated `main.py` typically does nothing but `load_dotenv()`, so this is most often a
    missing `.env` or a bug in the developer's own `main.py`/`app/` code, not a Runa bug.
    `cli/main.py` catches this and prints one clean line instead of a raw multi-frame traceback,
    and points at `python main.py` for the full one, since that traceback belongs to the
    developer's own entry point.
    """


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

    Shared by `chat.py` (resolving an Agent by name) and `cli/agents.py` (listing every
    declared Agent), so both give the same clean error outside a `runa new` project.
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
