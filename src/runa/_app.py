"""`runa._app`: the facts this process holds once, and the one `reset` that forgets them all.

A process runs one app, the way a Rails process has one `Rails.root`. What `runa.db` and
`runa.project` share about it is held here: the root, and the `memory://` database, resolved the
first time something reads it (`held`), never at import, so the environment that decides it is
whatever the app's `main.py` has set up by then. A module whose own process state must be
forgotten with the app's registers a callback with `on_reset`.

This module is a leaf: it imports nothing from Runa but its exceptions, so any module can hold a
fact here without an import cycle. A fact's type belongs to the module that owns it, which passes
in the `factory` that builds it.

Which app is the first fact, and it has three readers. `runa.db` resolves the app's state against
its root, `project_path` resolves the files it keeps against it, and `runa.project` imports its
`main.py` from there. It is held here, in a leaf both of
them import, because `runa.db` cannot import `runa.project` (the dependency runs the other way),
and one value means the two can never name different apps. `runa.project.use` is the public way
to say it; this module is the state it writes.

Undoing an import lives here as well, since changing the fact is what undoes one. Doing the
import is `runa.project`'s job: it runs the developer's code and reports their failures.
"""

import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from runa.exceptions import UserError

root: Path | None = None
"""The app this process reads, as it was first said, or `None` while it reads the cwd's."""

imported = False
"""Whether `root`'s `main.py` has run in this process yet."""

lock = threading.Lock()
"""Held across checking and changing the fact, so concurrent first callers load the app once.

`runa ui`'s routes are plain `def`, which uvicorn runs in a thread pool. Without the lock, two page
views arriving together would both find `imported` false and both re-execute `main.py`, each
tearing down the other's half-imported `app.*` modules.
"""

_MODULE_NAMES = ("main", "app", "tests", "evals")

_held: dict[str, Any] = {}

_reset_callbacks: list[Callable[[], None]] = []

MEMORY_DATABASE = "memory://"
"""The name the `memory://` database is held under: emptying it is one `forget(MEMORY_DATABASE)`."""


def use(new_root: Path | None) -> None:
    """Point this process at the Runa app in `new_root`; `None` means it reads no app.

    Startup configuration, said once by every surface that fronts an app. `runa` says it from
    `cwd` before dispatching a command, `runa serve` and `runa ui` say it at build time, and
    `runa chat`/`eval`/`test` say it through `runa.project.loaded_app`. Saying it moves every
    concern `runa.db` resolves (sessions, traces, eval history, memory, knowledge, the cache), and
    it names the `main.py` that gets imported the first time anything reads the app's modules.
    Saying it imports nothing, so `runa traces show` reads a project's history without running
    a line of that project's code.

    A second, *different* root in one process raises `UserError`. Both apps' modules would want
    the same `main`/`app.*` names, so the loser would quietly serve the winner's agents off the
    winner's database. A deployment that really fronts two apps runs two processes.

    `None` clears the fact and undoes the import with it, so the next `use` counts as the first.
    Nothing in a running app calls it, because there a second app is a second process. A test
    that points this at a `tmp_path` is standing in for that process boundary.
    """
    global root, imported
    with lock:
        if new_root is not None and root is not None:
            if new_root.resolve() != root.resolve():
                raise UserError(
                    f"this process already reads the Runa app at {root}, so it cannot also read "
                    f"the one at {new_root}: a process reads one app."
                )
            return
        if root is not None and imported:
            unimport(root)
        root, imported = new_root, False


def held[T](name: str, factory: Callable[[], T]) -> T:
    """The fact held as `name`, built by `factory` the first time anything asks for it.

    Two first callers racing may each build one, but only the first to land is kept and both get
    that one, so `factory` must have no side effects beyond the object it returns.
    """
    if name not in _held:
        _held.setdefault(name, factory())
    return cast("T", _held[name])


def on_reset(callback: Callable[[], None]) -> None:
    """Call `callback` whenever `reset` returns the process to how it started.

    For process state a module keeps itself because its type or lifetime is its own, so it is
    forgotten with the app's rather than surviving into the next one.
    """
    _reset_callbacks.append(callback)


def project_path(path: str | Path) -> Path:
    """Resolve a relative `path` against the app `use` named; absolute ones unchanged.

    For the state an app keeps as files rather than rows, such as `app/knowledge/`: it is read
    from the same root the app's database is written under, or the two split in two.
    """
    path = Path(path)
    if root is None or path.is_absolute():
        return path
    return root / path


def forget(prefix: str) -> None:
    """Forget every fact whose name starts with `prefix`, so the next `held` builds it again.

    A fact holding a resource (one with a `close()`, like the `memory://` database's connection)
    is closed as it is forgotten, rather than left for the garbage collector.
    """
    for name in [name for name in _held if name.startswith(prefix)]:
        fact = _held.pop(name)
        if callable(close := getattr(fact, "close", None)):
            close()


def reset() -> None:
    """Return this process to how it started: no app said or imported, and no fact resolved.

    Nothing in a running app calls this, because there a fresh start means a fresh process. A test
    suite calls it between tests, since each test stands in for one.
    """
    use(None)
    forget("")
    for callback in _reset_callbacks:
        callback()


def import_path(app_root: Path) -> str:
    """The `sys.path` entry `app_root`'s modules are imported from."""
    return str(app_root.resolve())


def unimport(app_root: Path) -> None:
    """Undo `app_root`'s import: drop its `sys.path` entry and the modules it brought in."""
    entry = import_path(app_root)
    if entry in sys.path:
        sys.path.remove(entry)
    reset_modules()


def reset_modules() -> None:
    """Drop cached `main`/`app`/`tests`/`evals` modules left over from an earlier import.

    Python caches imports by name in `sys.modules`. Without this, a later import in the same
    process (across tests, say) would silently reuse a previous project's `main`/`app`/`tests`/
    `evals` instead of the one at `root`.
    """
    prefixes = tuple(f"{name}." for name in _MODULE_NAMES)
    for name in list(sys.modules):
        if name in _MODULE_NAMES or name.startswith(prefixes):
            del sys.modules[name]
