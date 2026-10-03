"""`runa.db`: where state lives, decided once, by the environment.

Everything Runa persists -- sessions, memory, knowledge, the cache, traces, eval history -- has to
answer one question before it can read or write: is this deployment sharing a database, or does it
have its own file? This module is the only place that answers it, and `RUNA_DATABASE_URL` is the
only thing it asks.

Unset means local: one `db/runa.db` SQLite file, created on first write, no setup. Set to a
`postgresql://` URL means shared: every concern moves to that database, and more than one replica
becomes possible. SQLite is per-process by design, so three pods on the default settings keep
three disjoint histories and a dashboard that can only ever show one of them.

The `session`/`memory_store`/`knowledge_store`/`cache` factories below are how every call site
asks. Resolving here rather than threading a backend choice through each of them is what keeps
`Memory()`, `runa serve` and `list_traces(...)` the same code either way -- and what keeps the
promise honest, since a deployment where traces follow the variable but sessions don't is worse
than one where neither does.

Every adapter import is deferred into the function that needs it: an app without the `postgres`
extra has to be able to ask the question and get the SQLite answer.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runa.cache import Cache
    from runa.knowledge import KnowledgeStore
    from runa.memory import MemoryStore
    from runa.session import SessionABC

DATABASE_URL_ENV = "RUNA_DATABASE_URL"

DEFAULT_DB_PATH = Path("db/runa.db")

_POSTGRES_SCHEMES = ("postgresql://", "postgres://")
_SQLITE_SCHEME = "sqlite://"


class InvalidDatabaseURL(Exception):
    """Raised when `RUNA_DATABASE_URL` is set to something that names no supported backend.

    Its own type, not a bare `ValueError`, so `cli/main.py` can turn an operator's typo into one
    clean line like every other operator error, instead of a traceback through the storage layer.
    """


def shared_url() -> str | None:
    """The Postgres URL this deployment shares, or `None` when its state is a local file.

    `None` is the answer for an unset `RUNA_DATABASE_URL` and for an explicit `sqlite://` one, so
    a caller only ever has to branch on "shared or not", never on which of two local forms said
    so. Raises `InvalidDatabaseURL` for a URL that is neither: a typo in a deployment's database
    URL should stop the app, not silently route a replica's history to its own disk.
    """
    url = os.environ.get(DATABASE_URL_ENV) or None
    if url is None:
        return None
    if url.startswith(_POSTGRES_SCHEMES):
        return url
    if url.startswith(_SQLITE_SCHEME):
        return None
    raise InvalidDatabaseURL(
        f"{DATABASE_URL_ENV} must start with postgresql:// or sqlite://, got {url.split(':')[0]!r}"
    )


def sqlite_path() -> Path:
    """Where the local SQLite file lives: `RUNA_DATABASE_URL`'s path, or `db/runa.db`.

    Three slashes is a relative path and four is absolute, the form SQLAlchemy and every
    `DATABASE_URL` convention already use, so `sqlite:///data/runa.db` relocates the file without
    a `db_path=` argument on five different constructors.
    """
    url = os.environ.get(DATABASE_URL_ENV) or None
    if url is None or not url.startswith(_SQLITE_SCHEME):
        return DEFAULT_DB_PATH
    tail = url.removeprefix(_SQLITE_SCHEME)
    return Path(tail[1:]) if tail.startswith("//") else Path(tail.lstrip("/"))


def session(
    session_id: str, *, user_id: str | None = None, db_path: Path | None = None
) -> SessionABC:
    """This deployment's session store for `session_id`, Postgres-backed or SQLite-backed.

    `user_id` scopes the session's automatic memory, if its agent has any; see `SessionABC`.

    `db_path` overrides where the *local* file lives, for a CLI invoked against another project's
    `--root`. A shared deployment has one database and no such choice to make, so it goes unread.
    """
    if (url := shared_url()) is not None:
        from runa.session.postgres import PostgresSession

        return PostgresSession(session_id, url, user_id=user_id)
    from runa.session.sqlite import SQLiteSession

    return SQLiteSession(session_id, db_path or sqlite_path(), user_id=user_id)


def memory_store(*, dimensions: int) -> MemoryStore:
    """This deployment's `MemoryStore`, for vectors of `dimensions` floats."""
    if (url := shared_url()) is not None:
        from runa.memory.postgres import PostgresMemoryStore

        return PostgresMemoryStore(url, dimensions=dimensions)
    from runa.memory.sqlite import SQLiteMemoryStore

    return SQLiteMemoryStore(sqlite_path(), dimensions=dimensions)


def knowledge_store(*, dimensions: int) -> KnowledgeStore:
    """This deployment's `KnowledgeStore`, for vectors of `dimensions` floats."""
    if (url := shared_url()) is not None:
        from runa.knowledge.postgres import PostgresKnowledgeStore

        return PostgresKnowledgeStore(url, dimensions=dimensions)
    from runa.knowledge.sqlite import SQLiteKnowledgeStore

    return SQLiteKnowledgeStore(sqlite_path(), dimensions=dimensions)


def cache() -> Cache:
    """This deployment's persistent `Cache`: a table in whichever database it already has.

    No second variable and no second service: the cache follows `RUNA_DATABASE_URL` like
    everything else. `MemoryCache()` is still there for a cache that should die with the process,
    and `RedisCache(url)` for a deployment that wants its hot keys off the query path.
    """
    if (url := shared_url()) is not None:
        from runa.cache.postgres import PostgresCache

        return PostgresCache(url)
    from runa.cache.sqlite import SQLiteCache

    return SQLiteCache(sqlite_path())


__all__ = [
    "DATABASE_URL_ENV",
    "DEFAULT_DB_PATH",
    "InvalidDatabaseURL",
    "cache",
    "knowledge_store",
    "memory_store",
    "session",
    "shared_url",
    "sqlite_path",
]
