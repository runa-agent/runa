"""`runa.db`: where state lives, decided once, by the environment.

Everything Runa persists -- sessions, memory, knowledge, the cache, traces, eval history, paused
runs -- is local, shared or ephemeral, and this module is the only place that decides which, from
`RUNA_DATABASE_URL` alone. Unset means one `db/runa.db` SQLite file, created on first write. A
`postgresql://` URL moves every concern to that database, which is what running more than one
replica needs. `memory://` persists nothing: the stores live and die with the process, as a test
suite wants. Every concern follows the variable together.

A `Place` is that decision for one URL, with one factory per concern, each returning a store
object. The module-level factories below are the `Place` the environment names, so nothing above
this module names a backend, a URL or a path. `at(url)` is the one escape hatch, for state that
belongs in some other database: the same factories, bound to an explicit URL. A concern is one
class over the `Database` that `Place.database` hands out, so the closed set of three backends is
decided in that one method and a new concern is a new factory here and no new adapter.

`runa.project.use(root)` points the process at an app's directory (the cwd by default), and every
factory resolves a relative SQLite path against it. A shared database ignores the root.

Every backend import is deferred into the method that needs it, so an app without the
`postgres` extra still gets the SQLite answer.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

from runa import _app
from runa.exceptions import OperatorError, UserError

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
_SCHEMES = (*_POSTGRES_SCHEMES, _SQLITE_SCHEME, _MEMORY_SCHEME)


class InvalidDatabaseURL(OperatorError):
    """Raised when `RUNA_DATABASE_URL` is set to something that names no supported backend.

    An `OperatorError`, not a bare `ValueError`, so a typo in the environment surfaces as one
    clean line like every other operator error, not as a traceback through the storage layer.
    """


def _unsupported(url: str) -> str:
    return f"must start with postgresql://, sqlite:// or memory://, got {url.split(':')[0]!r}"


class Place:
    """One database every concern's store can live in, named by a URL.

    `None` is the default `db/runa.db` file; otherwise the URL is a `postgresql://` (or
    `postgres://`) one, a `sqlite://` path, or `memory://`, this process's one in-memory database.
    Reach one through the module-level factories (the environment's) or `at(url)`.
    """

    def __init__(self, url: str | None) -> None:
        """Hold `url`, which the caller has already checked names one of the three backends."""
        self.url = url

    def shared_url(self) -> str | None:
        """The Postgres URL this place is, or `None` when it is local or in memory.

        So a caller only ever has to branch on "shared or not", never on which of the local
        forms said so.
        """
        return self.url if self.url is not None and self.url.startswith(_POSTGRES_SCHEMES) else None

    def ephemeral(self) -> bool:
        """Whether this place keeps its state in this process only (`memory://`)."""
        return self.url is not None and self.url.startswith(_MEMORY_SCHEME)

    def sqlite_path(self) -> Path:
        """Where the local SQLite file lives: the URL's path, or `db/runa.db`.

        Three slashes is a relative path and four is absolute, the form SQLAlchemy and every
        `DATABASE_URL` convention already use, so `sqlite:///data/runa.db` relocates the file.

        A relative path is taken relative to the project `project.use` named, so under
        `project.use(Path("../other-app"))` this is `../other-app/db/runa.db` -- the same file that
        app's own `runa chat` writes to. Absolute paths and an unset project (an app running in its
        own directory) are unaffected.
        """
        if self.url is None or not self.url.startswith(_SQLITE_SCHEME):
            path = _DEFAULT_DB_PATH
        else:
            tail = self.url.removeprefix(_SQLITE_SCHEME)
            path = Path(tail[1:]) if tail.startswith("//") else Path(tail.lstrip("/"))
        return _app.project_path(path)

    def sqlite_database(self) -> Location:
        """The SQLite database this place's local state lives in.

        The file at `sqlite_path()`, or under `memory://` this process's one `MemoryDatabase`,
        which every local store shares until `reset_ephemeral` drops it. Both answers go through
        the same adapters and the same SQL.
        """
        if not self.ephemeral():
            return self.sqlite_path()
        from runa.db.sqlite import MemoryDatabase

        return _app.held(_app.MEMORY_DATABASE, MemoryDatabase)

    def database(self, *tables: Table) -> Database:
        """The database `tables` live in here: a shared Postgres, or a local SQLite.

        The one place a concern's backend is decided. Each concern below is a single class over
        what this returns, so it never learns which engine answered.
        """
        if (url := self.shared_url()) is not None:
            from runa.db.pool import PostgresDatabase

            return PostgresDatabase(url, *tables)
        from runa.db.sqlite import SQLiteDatabase

        return SQLiteDatabase(self.sqlite_database(), *tables)

    def session(self, session_id: str, agent: str | None = None) -> Session:
        """The session store for `session_id`, the conversation of `agent` if named."""
        from runa.session.store import MESSAGES, SESSIONS, DatabaseSession

        return DatabaseSession(session_id, self.database(SESSIONS, MESSAGES), agent)

    def sessions(self) -> SessionStore:
        """The `SessionStore`: the read side of the conversation history kept here."""
        from runa.session.store import MESSAGES, SESSIONS, SessionStore

        return SessionStore(self.database(SESSIONS, MESSAGES))

    def traces(self) -> TraceStore:
        """The `TraceStore`: where finished traces are written and read back."""
        from runa.tracing.store import SPANS, TRACES, TraceStore

        return TraceStore(self.database(TRACES, SPANS))

    def evals(self) -> EvalStore:
        """The `EvalStore`: every `agent.evaluate()` run and its baseline."""
        from runa.eval.store import CASES, RUNS, EvalStore

        return EvalStore(self.database(RUNS, CASES))

    def vector_store(self, spec: VectorSpec) -> VersionedVectorStore:
        """The vector storage for `spec`: what a bare `Memory()` or `Knowledge()` holds.

        Each of the two passes its own spec, so one call serves both concerns and each gets its
        own tables. The one concern that stays two adapters: `vec0` and `pgvector` are different
        query languages, not two spellings of one.
        """
        if (url := self.shared_url()) is not None:
            from runa.db.vectors.postgres import PostgresVectorStore

            return PostgresVectorStore(spec, url)
        from runa.db.vectors.sqlite import SQLiteVectorStore

        return SQLiteVectorStore(spec, self.sqlite_database())

    def cache(self) -> Cache:
        """The persistent `Cache`: a table in this database.

        No second variable and no second service: under `memory://` that means a table in the
        in-memory database. `MemoryCache()` stays the explicit choice for a cache that should die
        with the process while everything else persists, and `RedisCache(url)` for a deployment
        that wants its hot keys off the query path.
        """
        from runa.cache.database import ENTRIES, DatabaseCache

        return DatabaseCache(self.database(ENTRIES))

    def paused_runs(self) -> PausedRunStore:
        """The `PausedRunStore`: every paused run, held until it resumes."""
        from runa.run_state.store import PAUSED_RUNS, PausedRunStore

        return PausedRunStore(self.database(PAUSED_RUNS))

    def __repr__(self) -> str:
        """`Place('<url>')`, the URL it was built with."""
        return f"Place({self.url!r})"


def at(url: str) -> Place:
    """Every concern's store at `url` instead of where `RUNA_DATABASE_URL` says.

    For state that belongs in a database other than this deployment's: `at(url).session(id)`,
    `at(url).cache()`, `StoreExporter(at(url).traces())`. The URL takes the forms the variable
    does, so `sqlite:///other/runa.db` is a relative file and `sqlite:////var/runa.db` an absolute
    one. Raises `UserError` for a URL that names no supported backend.
    """
    if not url.startswith(_SCHEMES):
        raise UserError(f"db.at(url): url {_unsupported(url)}")
    return Place(url)


def _configured() -> Place:
    """The `Place` `RUNA_DATABASE_URL` names, read afresh so a changed environment is seen.

    Raises `InvalidDatabaseURL` for a URL that names none of the three: a typo in a deployment's
    database URL should stop the app, not silently route a replica's history to its own disk.
    """
    url = os.environ.get(DATABASE_URL_ENV) or None
    if url is not None and not url.startswith(_SCHEMES):
        raise InvalidDatabaseURL(f"{DATABASE_URL_ENV} {_unsupported(url)}")
    return Place(url)


def shared_url() -> str | None:
    """The Postgres URL this deployment shares, or `None` when its state is not shared."""
    return _configured().shared_url()


def ephemeral() -> bool:
    """Whether this deployment keeps its state in this process only (`memory://`).

    The honest answer to "where does state live" for a test suite: one `monkeypatch.setenv`
    moves every concern into one in-memory SQLite database at once, with no live Postgres and
    no temporary file, and `runa.db` stays the only thing that decided.
    """
    return _configured().ephemeral()


def sqlite_path() -> Path:
    """Where this deployment's SQLite file lives: `RUNA_DATABASE_URL`'s path, or `db/runa.db`."""
    return _configured().sqlite_path()


def sqlite_database() -> Location:
    """The SQLite database this deployment's local state lives in."""
    return _configured().sqlite_database()


def database(*tables: Table) -> Database:
    """The database `tables` live in: this deployment's shared Postgres, or its local SQLite."""
    return _configured().database(*tables)


def session(session_id: str, agent: str | None = None) -> Session:
    """This deployment's session store for `session_id`, the conversation of `agent` if named."""
    return _configured().session(session_id, agent)


def sessions() -> SessionStore:
    """This deployment's `SessionStore`: the read side of its conversation history."""
    return _configured().sessions()


def traces() -> TraceStore:
    """This deployment's `TraceStore`: where finished traces are written and read back."""
    return _configured().traces()


def evals() -> EvalStore:
    """This deployment's `EvalStore`: every `agent.evaluate()` run and its baseline."""
    return _configured().evals()


def vector_store(spec: VectorSpec) -> VersionedVectorStore:
    """This deployment's vector storage for `spec`, what a bare `Memory()`/`Knowledge()` holds."""
    return _configured().vector_store(spec)


def cache() -> Cache:
    """This deployment's persistent `Cache`: a table in whichever database it already has."""
    return _configured().cache()


def paused_runs() -> PausedRunStore:
    """This deployment's `PausedRunStore`: every paused run, held until it resumes."""
    return _configured().paused_runs()


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
    "Place",
    "at",
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
