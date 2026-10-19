"""`runa.db`: where state lives, decided once, by the environment.

Everything Runa persists -- sessions, memory, knowledge, the cache, traces, eval history -- is
local, shared or ephemeral, and this module is the only place that decides which, from
`RUNA_DATABASE_URL` alone. Unset means one `db/runa.db` SQLite file, created on first write. A
`postgresql://` URL moves every concern to that database, which is what running more than one
replica needs. `memory://` persists nothing: the stores live and die with the process, as a test
suite wants. Every concern follows the variable together.

The factories below are how every call site asks, one per concern, each returning a store
object, so nothing above this module names a backend, a URL or a path. Each factory branches on
the closed set of three backends itself; a new concern is one function here.

`runa.project.use(root)` points the process at an app's directory (the cwd by default), and every
factory resolves against it: the local SQLite file, and files an app keeps as state rather than
rows (`Knowledge`'s source directory, via `project_path`). A shared database ignores the root.

Every adapter import is deferred into the function that needs it, so an app without the
`postgres` extra still gets the SQLite answer.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

from runa import _app
from runa.exceptions import OperatorError

if TYPE_CHECKING:
    from runa.cache import Cache
    from runa.db.sqlite import Database
    from runa.db.vectors import VectorSpec, VersionedVectorStore
    from runa.eval.store import EvalStore
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
    moves all six concerns into one in-memory SQLite database at once, with no live Postgres and
    no temporary file, and `runa.db` stays the only thing that decided.
    """
    url = os.environ.get(DATABASE_URL_ENV) or None
    return url is not None and url.startswith(_MEMORY_SCHEME)


def project_path(path: str | Path) -> Path:
    """Resolve a relative `path` against the project `project.use` named; absolute ones unchanged.

    The non-database half of `sqlite_path` below, for the state an app keeps as files rather than
    as rows: `app/knowledge/` is this application's state exactly as much as `db/runa.db` is, and
    a corpus read from the cwd while its vectors are written under the project root is the same
    split-in-two failure, one directory further out. Whoever would otherwise resolve it --
    `Knowledge.__init__`, reached through `Agent.__init__` -- is precisely the call site with no
    root to resolve against, which is why the answer lives here.
    """
    path = Path(path)
    if _app.root is None or path.is_absolute():
        return path
    return _app.root / path


def sqlite_path() -> Path:
    """Where the local SQLite file lives: `RUNA_DATABASE_URL`'s path, or `db/runa.db`.

    Three slashes is a relative path and four is absolute, the form SQLAlchemy and every
    `DATABASE_URL` convention already use, so `sqlite:///data/runa.db` relocates the file without
    a `db_path=` argument on five different constructors.

    A relative path is taken relative to the project `project.use` named, so under
    `project.use(Path("../other-app"))` this is `../other-app/db/runa.db` -- the same file that
    app's own `runa chat` writes to. Absolute paths and an unset project (an app running in its
    own directory) are unaffected.
    """
    url = os.environ.get(DATABASE_URL_ENV) or None
    if url is None or not url.startswith(_SQLITE_SCHEME):
        path = _DEFAULT_DB_PATH
    else:
        tail = url.removeprefix(_SQLITE_SCHEME)
        path = Path(tail[1:]) if tail.startswith("//") else Path(tail.lstrip("/"))
    return project_path(path)


def sqlite_database() -> Database:
    """The SQLite database this deployment's local state lives in.

    The file at `sqlite_path()`, or under `memory://` this process's one `MemoryDatabase`, which
    every local adapter shares until `reset_ephemeral` drops it. Every SQLite adapter defaults to
    this, so both answers go through the same adapters and the same SQL.
    """
    if not ephemeral():
        return sqlite_path()
    from runa.db.sqlite import MemoryDatabase

    return _app.held(_app.MEMORY_DATABASE, MemoryDatabase)


def session(session_id: str) -> Session:
    """This deployment's session store for `session_id`."""
    if (url := shared_url()) is not None:
        from runa.session.postgres import PostgresSession

        return PostgresSession(session_id, url)
    from runa.session.sqlite import SQLiteSession

    return SQLiteSession(session_id)


def sessions() -> SessionStore:
    """This deployment's `SessionStore`: the read side of its conversation history."""
    if (url := shared_url()) is not None:
        from runa.session.postgres import PostgresSessionStore

        return PostgresSessionStore(url)
    from runa.session.sqlite import SQLiteSessionStore

    return SQLiteSessionStore()


def traces() -> TraceStore:
    """This deployment's `TraceStore`: where finished traces are written and read back."""
    if (url := shared_url()) is not None:
        from runa.tracing.postgres import PostgresTraceStore

        return PostgresTraceStore(url)
    from runa.tracing.sqlite import SQLiteTraceStore

    return SQLiteTraceStore()


def evals() -> EvalStore:
    """This deployment's `EvalStore`: every `agent.evaluate()` run and its baseline."""
    if (url := shared_url()) is not None:
        from runa.eval.postgres import PostgresEvalStore

        return PostgresEvalStore(url)
    from runa.eval.sqlite import SQLiteEvalStore

    return SQLiteEvalStore()


def vector_store(spec: VectorSpec) -> VersionedVectorStore:
    """This deployment's vector storage for `spec`: what a bare `Memory()` or `Knowledge()` holds.

    Each of the two passes its own spec, so one call serves both concerns and each gets its own
    tables.
    """
    if (url := shared_url()) is not None:
        from runa.db.vectors.postgres import PostgresVectorStore

        return PostgresVectorStore(spec, url)
    from runa.db.vectors.sqlite import SQLiteVectorStore

    return SQLiteVectorStore(spec, sqlite_database())


def cache() -> Cache:
    """This deployment's persistent `Cache`: a table in whichever database it already has.

    No second variable and no second service: the cache follows `RUNA_DATABASE_URL` like
    everything else, which under `memory://` means a table in the in-memory database.
    `MemoryCache()` stays the explicit choice for a cache that should die with the process while
    everything else persists, and `RedisCache(url)` for a deployment that wants its hot keys off
    the query path.
    """
    if (url := shared_url()) is not None:
        from runa.cache.postgres import PostgresCache

        return PostgresCache(url)
    from runa.cache.sqlite import SQLiteCache

    return SQLiteCache()


def reset_ephemeral() -> None:
    """Empty the `memory://` database, for a test that wants an empty one.

    The in-memory database lives for the whole process, the way a file stays on disk: two
    `db.traces()` calls have to see each other's writes, so a fresh database per call would make
    the whole backend a no-op. That makes emptying it an explicit step, and this is it. One call
    covers all six concerns, because they share the one database: this drops it, and the next
    store to ask gets a new, empty one (`runa._app` closes the old one as it forgets it). Only
    the database is emptied: the app this process reads and its tracing policy stay as they were.
    """
    _app.forget(_app.MEMORY_DATABASE)


__all__ = [
    "DATABASE_URL_ENV",
    "InvalidDatabaseURL",
    "cache",
    "ephemeral",
    "evals",
    "project_path",
    "reset_ephemeral",
    "session",
    "sessions",
    "shared_url",
    "sqlite_database",
    "sqlite_path",
    "traces",
    "vector_store",
]
