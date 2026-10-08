"""`runa.db`: where state lives, decided once, by the environment.

Everything Runa persists -- sessions, memory, knowledge, the cache, traces, eval history -- has to
answer one question before it can read or write: is this deployment sharing a database, keeping
its own file, or keeping nothing at all? This module is the only place that answers it, and
`RUNA_DATABASE_URL` is the only thing it asks.

Unset means local: one `db/runa.db` SQLite file, created on first write, no setup. A
`postgresql://` URL means shared: every concern moves to that database, and more than one replica
becomes possible. SQLite is per-process by design, so three pods on the default settings keep
three disjoint histories and a dashboard that can only ever show one of them. `memory://` means
nothing is persisted at all -- the whole store set lives in this process and dies with it, which
is what a test suite wants and what no deployment does.

The factories below are how every call site asks, one per concern. Each returns a store object,
so nothing above this module names a backend, a URL or a path: `runa ui` holds a `TraceStore`, not
a file, and moving a deployment to Postgres stays one environment variable rather than an edit.
Resolving here is also what keeps the promise honest, since a deployment where traces follow the
variable but sessions don't is worse than one where neither does.

Which project is being read is this module's one other decision, and it is asked once too:
`use_project(root)` points the process at an app's directory, and every factory above resolves
against it. The process's cwd is the default, which is what a bare `python main.py` wants. The
rule for the rest is that a function handed someone else's project `root` says so here --
`main(cwd=...)`, both `create_app(root)`s, `run_agent_repl`, `run_project_evals`,
`run_project_tests` -- so that taking a `root` and reading that project's state are never two
different things. Everything downstream of them (`runa ui`'s readers, `agent.evaluate()`, a
`@tool` opening a session) gets the answer without naming a directory, and a `root` that survives
elsewhere means a filesystem path, like `evals/`, rather than a second opinion about the database.

A `root=` parameter per factory was the same promise made seven times and kept four: `Memory`
and `Knowledge` are built inside `Agent.__init__`, which has no notion of a project root and
should not grow one, so their stores silently used the cwd while sessions, traces and eval
history followed the argument. One application's state split across two files is the exact
failure this module exists to prevent, so the root is held here rather than threaded, the way
`RUNA_DATABASE_URL` already is. It only moves the *local* file; a shared deployment has one
database and no such choice to make, so it goes unread.

Every adapter import is deferred into the function that needs it: an app without the `postgres`
extra has to be able to ask the question and get the SQLite answer.

Each factory repeats the same three-way branch rather than delegating to one resolved backend
object. That is deliberate: backends are a closed set of three, concerns are the axis that
grows, and inline branching keeps a new concern at one function in one file. See
`docs/adr/0005-runa-db-repeats-its-branch-per-concern.md` before collapsing it.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

from runa.exceptions import OperatorError

if TYPE_CHECKING:
    from runa.cache import Cache
    from runa.db.vectors import VectorSpec, VectorStore
    from runa.eval.store import EvalStore
    from runa.knowledge import KnowledgeStore
    from runa.memory import MemoryStore
    from runa.session import Session
    from runa.session.store import SessionStore
    from runa.tracing.store import TraceStore

DATABASE_URL_ENV = "RUNA_DATABASE_URL"

_DEFAULT_DB_PATH = Path("db/runa.db")

_POSTGRES_SCHEMES = ("postgresql://", "postgres://")
_SQLITE_SCHEME = "sqlite://"
_MEMORY_SCHEME = "memory://"


class InvalidDatabaseURL(OperatorError):
    """Raised when `RUNA_DATABASE_URL` is set to something that names no supported backend.

    An `OperatorError`, not a bare `ValueError`, so a typo in the environment surfaces as one
    clean line like every other operator error, not as a traceback through the storage layer.
    """


def shared_url() -> str | None:
    """The Postgres URL this deployment shares, or `None` when its state is not shared.

    `None` is the answer for an unset `RUNA_DATABASE_URL` and for an explicit `sqlite://` or
    `memory://` one, so a caller only ever has to branch on "shared or not", never on which of
    the local forms said so. Raises `InvalidDatabaseURL` for a URL that names none of the three:
    a typo in a deployment's database URL should stop the app, not silently route a replica's
    history to its own disk.
    """
    url = os.environ.get(DATABASE_URL_ENV) or None
    if url is None:
        return None
    if url.startswith(_POSTGRES_SCHEMES):
        return url
    if url.startswith((_SQLITE_SCHEME, _MEMORY_SCHEME)):
        return None
    raise InvalidDatabaseURL(
        f"{DATABASE_URL_ENV} must start with postgresql://, sqlite:// or memory://, "
        f"got {url.split(':')[0]!r}"
    )


def ephemeral() -> bool:
    """Whether this deployment keeps its state in this process only (`memory://`).

    The honest answer to "where does state live" for a test suite: one `monkeypatch.setenv`
    swaps all six concerns onto their in-process adapters at once, with no live Postgres and no
    temporary file, and `runa.db` stays the only thing that decided.
    """
    url = os.environ.get(DATABASE_URL_ENV) or None
    return url is not None and url.startswith(_MEMORY_SCHEME)


_project_root: Path | None = None
"""The project directory this process reads, or `None` for its cwd. Written by `use_project`."""


def use_project(root: Path | None) -> None:
    """Point this process's local state at the project in `root`; `None` means its cwd.

    Startup configuration, called once, like `tracing.add_exporter` and for the same reason: a
    process reads one app, and a root passed per call site is a root three of the seven concerns
    below can't be passed at all. `runa` sets this from `cwd` before dispatching, and both
    `create_app(root)`s set it while building, so an embedder serving an app from somewhere other
    than its own working directory gets every concern moved, not four of them.

    Unread unless this deployment's state is local: a `postgresql://` or `memory://` URL has no
    file to place.
    """
    global _project_root
    _project_root = root


def sqlite_path() -> Path:
    """Where the local SQLite file lives: `RUNA_DATABASE_URL`'s path, or `db/runa.db`.

    Three slashes is a relative path and four is absolute, the form SQLAlchemy and every
    `DATABASE_URL` convention already use, so `sqlite:///data/runa.db` relocates the file without
    a `db_path=` argument on five different constructors.

    A relative path is taken relative to the project `use_project` named, so under
    `use_project(Path("../other-app"))` this is `../other-app/db/runa.db` -- the same file that
    app's own `runa chat` writes to. Absolute paths and an unset project (an app running in its
    own directory) are unaffected.
    """
    url = os.environ.get(DATABASE_URL_ENV) or None
    if url is None or not url.startswith(_SQLITE_SCHEME):
        path = _DEFAULT_DB_PATH
    else:
        tail = url.removeprefix(_SQLITE_SCHEME)
        path = Path(tail[1:]) if tail.startswith("//") else Path(tail.lstrip("/"))
    if _project_root is None or path.is_absolute():
        return path
    return _project_root / path


def session(session_id: str, *, user_id: str | None = None) -> Session:
    """This deployment's session store for `session_id`.

    `user_id` scopes the session's automatic memory, if its agent has any; see `Session`.
    """
    if ephemeral():
        from runa.session.ephemeral import EphemeralSession

        return EphemeralSession(session_id, user_id=user_id)
    if (url := shared_url()) is not None:
        from runa.session.postgres import PostgresSession

        return PostgresSession(session_id, url, user_id=user_id)
    from runa.session.sqlite import SQLiteSession

    return SQLiteSession(session_id, sqlite_path(), user_id=user_id)


def sessions() -> SessionStore:
    """This deployment's `SessionStore`: the read side of its conversation history."""
    if ephemeral():
        from runa.session.ephemeral import EphemeralSessionStore

        return EphemeralSessionStore()
    if (url := shared_url()) is not None:
        from runa.session.postgres import PostgresSessionStore

        return PostgresSessionStore(url)
    from runa.session.sqlite import SQLiteSessionStore

    return SQLiteSessionStore(sqlite_path())


def traces() -> TraceStore:
    """This deployment's `TraceStore`: where finished traces are written and read back."""
    if ephemeral():
        from runa.tracing.ephemeral import EphemeralTraceStore

        return EphemeralTraceStore()
    if (url := shared_url()) is not None:
        from runa.tracing.postgres import PostgresTraceStore

        return PostgresTraceStore(url)
    from runa.tracing.sqlite import SQLiteTraceStore

    return SQLiteTraceStore(sqlite_path())


def evals() -> EvalStore:
    """This deployment's `EvalStore`: every `agent.evaluate()` run and its baseline."""
    if ephemeral():
        from runa.eval.ephemeral import EphemeralEvalStore

        return EphemeralEvalStore()
    if (url := shared_url()) is not None:
        from runa.eval.postgres import PostgresEvalStore

        return PostgresEvalStore(url)
    from runa.eval.sqlite import SQLiteEvalStore

    return SQLiteEvalStore(sqlite_path())


def memory_store(*, dimensions: int) -> MemoryStore:
    """This deployment's `MemoryStore`, for vectors of `dimensions` floats.

    The in-process answer is paired here rather than in a named `memory/ephemeral.py`: nothing
    reaches it by name, so there is nothing for a module to hold but this one line. The two
    named adapters below stay, because `Memory(store=...)` is a sanctioned override and their
    import paths are documented (see `docs/adr/0001-vector-store-is-app-private.md`).
    """
    if ephemeral():
        from runa.db.vectors.ephemeral import EphemeralVectorStore
        from runa.memory.vector import VectorMemoryStore, spec

        return VectorMemoryStore(EphemeralVectorStore(spec(dimensions)))
    if (url := shared_url()) is not None:
        from runa.memory.postgres import PostgresMemoryStore

        return PostgresMemoryStore(url, dimensions=dimensions)
    from runa.memory.sqlite import SQLiteMemoryStore

    return SQLiteMemoryStore(sqlite_path(), dimensions=dimensions)


def knowledge_store(*, dimensions: int) -> KnowledgeStore:
    """This deployment's `KnowledgeStore`, for vectors of `dimensions` floats.

    In-process is paired inline for the same reason as `memory_store` above.
    """
    if ephemeral():
        from runa.db.vectors.ephemeral import EphemeralVectorStore
        from runa.knowledge.vector import VectorKnowledgeStore, spec

        return VectorKnowledgeStore(EphemeralVectorStore(spec(dimensions)))
    if (url := shared_url()) is not None:
        from runa.knowledge.postgres import PostgresKnowledgeStore

        return PostgresKnowledgeStore(url, dimensions=dimensions)
    from runa.knowledge.sqlite import SQLiteKnowledgeStore

    return SQLiteKnowledgeStore(sqlite_path(), dimensions=dimensions)


def vector_store(spec: VectorSpec) -> VectorStore:
    """This deployment's `VectorStore` for `spec`: the storage `Memory` and `Knowledge` share.

    Plumbing rather than a concern, which is why there is no `Agent` attribute that reaches it:
    `memory_store()` and `knowledge_store()` above are the two callers, and each wraps what this
    returns in the typed store its own concern promises.
    """
    if ephemeral():
        from runa.db.vectors.ephemeral import EphemeralVectorStore

        return EphemeralVectorStore(spec)
    if (url := shared_url()) is not None:
        from runa.db.vectors.postgres import PostgresVectorStore

        return PostgresVectorStore(spec, url)
    from runa.db.vectors.sqlite import SQLiteVectorStore

    return SQLiteVectorStore(spec, sqlite_path())


def cache() -> Cache:
    """This deployment's persistent `Cache`: a table in whichever database it already has.

    No second variable and no second service: the cache follows `RUNA_DATABASE_URL` like
    everything else, which under `memory://` means the same `MemoryCache` you can also ask for by
    name. `MemoryCache()` stays the explicit choice for a cache that should die with the process
    while everything else persists, and `RedisCache(url)` for a deployment that wants its hot
    keys off the query path.
    """
    if ephemeral():
        from runa.cache.memory import shared

        return shared()
    if (url := shared_url()) is not None:
        from runa.cache.postgres import PostgresCache

        return PostgresCache(url)
    from runa.cache.sqlite import SQLiteCache

    return SQLiteCache(sqlite_path())


def reset_ephemeral() -> None:
    """Empty every `memory://` store, for a test that wants a clean process.

    The ephemeral adapters keep their tables at module level, the way a database keeps them on
    disk: two `db.traces()` calls have to see each other's writes, so a fresh object per call
    would make the whole backend a no-op. That makes emptying them an explicit step, and this is
    it -- one call, all six concerns, which is the point of resolving them in one place. Memory
    and knowledge are emptied by the one vector reset, since they share that backend.
    """
    from runa.cache.memory import reset as reset_cache
    from runa.db.vectors.ephemeral import reset as reset_vectors
    from runa.eval.ephemeral import reset as reset_evals
    from runa.session.ephemeral import reset as reset_sessions
    from runa.tracing.ephemeral import reset as reset_traces

    for reset in (reset_cache, reset_evals, reset_sessions, reset_traces, reset_vectors):
        reset()


__all__ = [
    "DATABASE_URL_ENV",
    "InvalidDatabaseURL",
    "cache",
    "ephemeral",
    "evals",
    "knowledge_store",
    "memory_store",
    "reset_ephemeral",
    "session",
    "sessions",
    "shared_url",
    "sqlite_path",
    "traces",
    "use_project",
    "vector_store",
]
