"""`runa.db`: storage backends -- `sqlite.py` (default, core), `postgres.py`/`redis.py` (extras).

The backends themselves aren't re-exported here; each concern (`session.py`, `memory.py`,
`knowledge.py`, `cache.py`, `tracing/storage.py`, `eval/storage.py`) imports the one it needs
directly. What does live here is the one question every one of them has to ask first: is this
deployment sharing a database, or does it have its own file?
"""

from __future__ import annotations

import os

SHARED_DSN_ENV = "RUNA_POSTGRES_DSN"


def shared_dsn() -> str | None:
    """The Postgres DSN this app shares, or `None` for the default per-process `db/runa.db`.

    Set `RUNA_POSTGRES_DSN` and everything Runa persists (sessions, traces, eval history) goes to
    that database instead of a local SQLite file, with no code change -- the same "it just works
    once the credentials are there" property `LANGFUSE_PUBLIC_KEY` already gives tracing.

    This is what makes more than one replica possible. SQLite is per-process by design, so three
    pods with the default settings keep three disjoint histories and a dashboard that can only
    ever show one of them. Reading this in the storage modules, rather than threading a backend
    choice through every call site, keeps `list_traces(...)` the same function either way.

    Importing this module never imports `asyncpg`: an app without the `postgres` extra has to be
    able to ask the question and get `None`.
    """
    return os.environ.get(SHARED_DSN_ENV) or None


__all__ = ["SHARED_DSN_ENV", "shared_dsn"]
