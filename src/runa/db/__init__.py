"""`runa.db`: where state lives, decided once, by the environment.

Everything Runa persists -- sessions, memory, knowledge, the cache, traces, eval history, paused
runs -- is local, shared or ephemeral, and this module is the only place that decides which, from
`RUNA_DATABASE_URL` alone. Unset means one `db/runa.db` SQLite file, created on first write. A
`postgresql://` URL moves every concern to that database, which is what running more than one
replica needs. `memory://` persists nothing: the stores live and die with the process, as a test
suite wants. Every concern follows the variable together.

The factories below are how every call site asks, one per concern, each returning a store
object, so nothing above this module names a backend, a URL or a path. A concern is one class over
the `Database` that `database` hands out, so the closed set of three backends is decided in that
one function and a new concern is a new factory here and no new adapter.

`runa.project.use(root)` points the process at an app's directory (the cwd by default), and every
factory resolves against it: the local SQLite file. A shared database ignores the root.

Every backend import is deferred into the function that needs it, so an app without the
`postgres` extra still gets the SQLite answer.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

from runa import _app
from runa.exceptions import OperatorError

if TYPE_CHECKING:
    from runa.cache import Cache
    from runa.db.handle import Database
    from runa.db.schema import Table
    from runa.db.sqlite import Location
    from runa.db.vectors import VectorSpec, VersionedVectorStore
    from runa.eval.store import EvalStore
    from runa.run_state.store import PausedRunStore
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
    moves every concern into one in-memory SQLite database at once, with no live Postgres and
    no temporary file, and `runa.db` stays the only thing that decided.
    """
    url = os.environ.get(DATABASE_URL_ENV) or None
    return url is not None and url.startswith(_MEMORY_SCHEME)


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
    return _app.project_path(path)


def sqlite_database() -> Location:
    """The SQLite database this deployment's local state lives in.

    The file at `sqlite_path()`, or under `memory://` this process's one `MemoryDatabase`, which
    every local adapter shares until `reset_ephemeral` drops it. Every SQLite adapter defaults to
    this, so both answers go through the same adapters and the same SQL.
    """
    if not ephemeral():
        return sqlite_path()
    from runa.db.sqlite import MemoryDatabase

    return _app.held(_app.MEMORY_DATABASE, MemoryDatabase)


def database(*tables: Table) -> Database:
    """The database `tables` live in: this deployment's shared Postgres, or its local SQLite.

    The one place a concern's backend is decided. Each concern below is a single class over what
    this returns, so it never learns which engine answered.
    """
    if (url := shared_url()) is not None:
        from runa.db.pool import PostgresDatabase

        return PostgresDatabase(url, *tables)
    from runa.db.sqlite import SQLiteDatabase

    return SQLiteDatabase(sqlite_database(), *tables)


def session(session_id: str, agent: str | None = None) -> Session:
    """This deployment's session store for `session_id`, the conversation of `agent` if named."""
    from runa.session.database import DatabaseSession
    from runa.session.store import MESSAGES, SESSIONS

    return DatabaseSession(session_id, database(SESSIONS, MESSAGES), agent)


def sessions() -> SessionStore:
    """This deployment's `SessionStore`: the read side of its conversation history."""
    from runa.session.database import DatabaseSessionStore
    from runa.session.store import MESSAGES, SESSIONS

    return DatabaseSessionStore(database(SESSIONS, MESSAGES))


def traces() -> TraceStore:
    """This deployment's `TraceStore`: where finished traces are written and read back."""
    from runa.tracing.database import DatabaseTraceStore
    from runa.tracing.store import SPANS, TRACES

    return DatabaseTraceStore(database(TRACES, SPANS))


def evals() -> EvalStore:
    """This deployment's `EvalStore`: every `agent.evaluate()` run and its baseline."""
    from runa.eval.database import DatabaseEvalStore
    from runa.eval.store import CASES, RUNS

    return DatabaseEvalStore(database(RUNS, CASES))


def vector_store(spec: VectorSpec) -> VersionedVectorStore:
    """This deployment's vector storage for `spec`: what a bare `Memory()` or `Knowledge()` holds.

    Each of the two passes its own spec, so one call serves both concerns and each gets its own
    tables. The one concern that stays two adapters: `vec0` and `pgvector` are different query
    languages, not two spellings of one.
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
    from runa.cache import ENTRIES
    from runa.cache.database import DatabaseCache

    return DatabaseCache(database(ENTRIES))


def paused_runs() -> PausedRunStore:
    """This deployment's `PausedRunStore`: every paused run, held until it resumes."""
    from runa.run_state.database import DatabasePausedRunStore
    from runa.run_state.store import PAUSED_RUNS

    return DatabasePausedRunStore(database(PAUSED_RUNS))


def reset_ephemeral() -> None:
    """Empty the `memory://` database, for a test that wants an empty one.

    The in-memory database lives for the whole process, the way a file stays on disk: two
    `db.traces()` calls have to see each other's writes, so a fresh database per call would make
    the whole backend a no-op. That makes emptying it an explicit step, and this is it. One call
    covers every concern, because they share the one database: this drops it, and the next
    store to ask gets a new, empty one (`runa._app` closes the old one as it forgets it). Only
    the database is emptied: the app this process reads stays as it was.
    """
    _app.forget(_app.MEMORY_DATABASE)


__all__ = [
    "DATABASE_URL_ENV",
    "InvalidDatabaseURL",
    "cache",
    "database",
    "ephemeral",
    "evals",
    "paused_runs",
    "reset_ephemeral",
    "session",
    "sessions",
    "shared_url",
    "sqlite_database",
    "sqlite_path",
    "traces",
    "vector_store",
]
