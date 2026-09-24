"""db/postgres.py: Postgres-backed `SessionABC`/`MemoryStore`/`KnowledgeStore`, via `asyncpg`.

Optional -- the `runa[postgres]` extra, not a core dependency; nothing outside this module
imports it. Same tables and query shapes as the SQLite defaults (`session.py`/`memory.py`/
`knowledge.py`), for a deployment where multiple processes need to share one store instead of
each having their own `db/runa.db`: a `vector` column (the `pgvector` Postgres extension)
replaces `sqlite-vec`'s `vec0`, and Postgres's own MVCC replaces `sqlite3`'s single-writer lock.

One connection pool per DSN, cached at module level and shared by every store built from it, so
`PostgresSession("a", dsn)` and `PostgresSession("b", dsn)` (or a
`PostgresMemoryStore`/`PostgresKnowledgeStore` on the same `dsn`) reuse one pool instead of each
opening their own -- the same "just works" ergonomics as `db/sqlite.py`'s shared `db/runa.db` file,
here applied to pool reuse instead of file reuse.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Coroutine
from typing import Any

import asyncpg

from runa._types import TResponseInputItem
from runa.knowledge import KnowledgeMatch
from runa.memory import MemoryMatch
from runa.session import SessionABC

DEFAULT_POSTGRES_DSN = "postgresql://runa:runa@localhost:5432/runa"

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _background_loop() -> asyncio.AbstractEventLoop:
    """A daemon event loop for the synchronous callers that still need `asyncpg`.

    `TraceExporter.export` and the whole `runa traces`/`runa ui` read path are synchronous, and
    a trace is exported from inside a finishing run, so `asyncio.run()` (which demands there be
    no running loop) is not available. One long-lived loop on its own thread serves them all,
    and `_get_pool`'s per-loop keying gives it its own pool, as it would any other loop.
    """
    global _loop
    if _loop is not None and not _loop.is_closed():
        return _loop
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, name="runa-postgres", daemon=True).start()
    return _loop


def run_sync[T](coro: Coroutine[Any, Any, T], *, timeout: float = 30.0) -> T:
    """Run `coro` on the background loop and wait for it, for a synchronous caller.

    Safe to call from inside a running event loop (which `asyncio.run` is not): the work happens
    on a different loop entirely, so it never tries to re-enter the caller's.
    """
    future = asyncio.run_coroutine_threadsafe(coro, _background_loop())
    return future.result(timeout)


_SESSIONS_TABLE = "agent_sessions"
_MESSAGES_TABLE = "agent_messages"
_MEMORY_ITEMS_TABLE = "memory_items"
_KNOWLEDGE_ITEMS_TABLE = "knowledge_items"

_SESSION_DDL = f"""
CREATE TABLE IF NOT EXISTS {_SESSIONS_TABLE} (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS {_MESSAGES_TABLE} (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES {_SESSIONS_TABLE}(session_id) ON DELETE CASCADE,
    message_data TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_{_MESSAGES_TABLE}_session_id ON {_MESSAGES_TABLE} (session_id, id);
"""


def _memory_ddl(dimensions: int) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {_MEMORY_ITEMS_TABLE} (
        id BIGSERIAL PRIMARY KEY,
        user_id TEXT,
        text TEXT NOT NULL,
        metadata TEXT,
        embedding vector({dimensions}) NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_{_MEMORY_ITEMS_TABLE}_user_id ON {_MEMORY_ITEMS_TABLE} (user_id);
    """


def _knowledge_ddl(dimensions: int) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {_KNOWLEDGE_ITEMS_TABLE} (
        id BIGSERIAL PRIMARY KEY,
        text TEXT NOT NULL,
        source TEXT NOT NULL,
        embedding vector({dimensions}) NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """


_pools: dict[tuple[int, str], asyncpg.Pool] = {}
_pools_lock = asyncio.Lock()


async def _get_pool(dsn: str) -> asyncpg.Pool:
    """Return `dsn`'s shared pool on the *current* event loop, creating it on first use.

    Keyed by `(id(loop), dsn)`, not just `dsn`: an `asyncpg.Pool`'s connections belong to the
    loop that created them, so reusing a pool from a since-closed loop (e.g. a second
    `asyncio.run()` call in the same process, as every test in `tests/test_postgres.py` makes)
    would hang forever acquiring a connection tied to a dead loop. One long-lived event loop
    (a real deployment's) still gets exactly one pool per `dsn`, same as before.

    `vector` has to exist as a type before any connection can register its codec, so a bare
    bootstrap connection creates the extension first -- `init=register_vector` on the pool itself
    would otherwise run on a database that doesn't have the type yet.
    """
    key = (id(asyncio.get_running_loop()), dsn)
    if key in _pools:
        return _pools[key]
    async with _pools_lock:
        if key not in _pools:
            from pgvector.asyncpg import register_vector

            bootstrap = await asyncpg.connect(dsn)
            try:
                await bootstrap.execute("CREATE EXTENSION IF NOT EXISTS vector")
            finally:
                await bootstrap.close()
            _pools[key] = await asyncpg.create_pool(dsn, init=register_vector)
    return _pools[key]


async def _connect(dsn: str, ddl: str) -> asyncpg.Pool:
    """Return `dsn`'s shared pool, applying `ddl` (idempotent) first."""
    pool = await _get_pool(dsn)
    async with pool.acquire() as conn:
        await conn.execute(ddl)
    return pool


class PostgresSession(SessionABC):
    """`SessionABC` backed by Postgres, for a deployment sharing history across processes.

    Same shape as `SQLiteSession`, minus the single-process assumption: every process on the
    same `dsn` sees the same history, since Postgres (unlike `session.py`'s bare
    `sqlite3.connect`) tolerates concurrent writers without corrupting the file.
    """

    def __init__(
        self,
        session_id: str,
        dsn: str = DEFAULT_POSTGRES_DSN,
        *,
        user_id: str | None = None,
    ) -> None:
        """Store `session_id` and which Postgres database its history lives in."""
        self.session_id = session_id
        self.dsn = dsn
        self.user_id = user_id

    async def _pool(self) -> asyncpg.Pool:
        return await _connect(self.dsn, _SESSION_DDL)

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        """Return this session's items, oldest first, capped at the latest `limit` if given."""
        pool = await self._pool()
        if limit is None:
            rows = await pool.fetch(
                f"SELECT message_data FROM {_MESSAGES_TABLE} WHERE session_id = $1 ORDER BY id",
                self.session_id,
            )
        else:
            rows = await pool.fetch(
                f"""
                SELECT message_data FROM {_MESSAGES_TABLE} WHERE session_id = $1
                ORDER BY id DESC LIMIT $2
                """,
                self.session_id,
                limit,
            )
            rows.reverse()
        return [json.loads(row["message_data"]) for row in rows]

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """Append `items`, creating the session row on first write."""
        if not items:
            return
        pool = await self._pool()
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(
                f"INSERT INTO {_SESSIONS_TABLE} (session_id) VALUES ($1) "
                "ON CONFLICT (session_id) DO NOTHING",
                self.session_id,
            )
            await conn.executemany(
                f"INSERT INTO {_MESSAGES_TABLE} (session_id, message_data) VALUES ($1, $2)",
                [(self.session_id, json.dumps(item)) for item in items],
            )
            await conn.execute(
                f"UPDATE {_SESSIONS_TABLE} SET updated_at = now() WHERE session_id = $1",
                self.session_id,
            )

    async def set_items(self, items: list[TResponseInputItem]) -> None:
        """Replace this session's entire history with `items`, in one transaction."""
        pool = await self._pool()
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(
                f"DELETE FROM {_MESSAGES_TABLE} WHERE session_id = $1", self.session_id
            )
            if items:
                await conn.execute(
                    f"INSERT INTO {_SESSIONS_TABLE} (session_id) VALUES ($1) "
                    "ON CONFLICT (session_id) DO NOTHING",
                    self.session_id,
                )
                await conn.executemany(
                    f"INSERT INTO {_MESSAGES_TABLE} (session_id, message_data) VALUES ($1, $2)",
                    [(self.session_id, json.dumps(item)) for item in items],
                )
            await conn.execute(
                f"UPDATE {_SESSIONS_TABLE} SET updated_at = now() WHERE session_id = $1",
                self.session_id,
            )

    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return this session's most recent item, or `None` if it has none."""
        pool = await self._pool()
        row = await pool.fetchrow(
            f"""
            DELETE FROM {_MESSAGES_TABLE}
            WHERE id = (
                SELECT id FROM {_MESSAGES_TABLE} WHERE session_id = $1 ORDER BY id DESC LIMIT 1
            )
            RETURNING message_data
            """,
            self.session_id,
        )
        return json.loads(row["message_data"]) if row else None

    async def clear_session(self) -> None:
        """Delete this session and all of its items."""
        pool = await self._pool()
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(
                f"DELETE FROM {_MESSAGES_TABLE} WHERE session_id = $1", self.session_id
            )
            await conn.execute(
                f"DELETE FROM {_SESSIONS_TABLE} WHERE session_id = $1", self.session_id
            )


class PostgresMemoryStore:
    """`MemoryStore` backed by Postgres + `pgvector`: the `Memory(store=...)` escape hatch.

    `user_id` is a plain filter column here, indexed so per-user nearest-neighbor search stays
    fast -- pgvector has no partition-key primitive the way `sqlite-vec`'s `vec0` does. Distance
    is `<->` (Euclidean/L2), matching `sqlite-vec`'s default and `memory.py`'s
    `_DUPLICATE_DISTANCE` assumption.
    """

    def __init__(self, dsn: str = DEFAULT_POSTGRES_DSN, *, dimensions: int) -> None:
        """Store which Postgres database this store's items/vectors live in, and vector size."""
        self.dsn = dsn
        self.dimensions = dimensions

    async def _pool(self) -> asyncpg.Pool:
        return await _connect(self.dsn, _memory_ddl(self.dimensions))

    async def add(
        self,
        *,
        user_id: str | None,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None,
    ) -> int:
        """Store one already-embedded item for `user_id`, returning its new id."""
        pool = await self._pool()
        row = await pool.fetchrow(
            f"""
            INSERT INTO {_MEMORY_ITEMS_TABLE} (user_id, text, metadata, embedding)
            VALUES ($1, $2, $3, $4)
            RETURNING id
            """,
            user_id,
            text,
            json.dumps(metadata) if metadata is not None else None,
            embedding,
        )
        assert row is not None
        return row["id"]

    async def search(
        self, *, user_id: str | None, embedding: list[float], k: int
    ) -> list[MemoryMatch]:
        """Return `user_id`'s `k` items closest to `embedding`, nearest first."""
        pool = await self._pool()
        rows = await pool.fetch(
            f"""
            SELECT id, text, metadata, embedding <-> $1 AS distance
            FROM {_MEMORY_ITEMS_TABLE}
            WHERE user_id IS NOT DISTINCT FROM $2
            ORDER BY embedding <-> $1
            LIMIT $3
            """,
            embedding,
            user_id,
            k,
        )
        return [
            MemoryMatch(
                id=row["id"],
                text=row["text"],
                metadata=json.loads(row["metadata"]) if row["metadata"] is not None else None,
                distance=row["distance"],
            )
            for row in rows
        ]

    async def delete(self, *, user_id: str | None, memory_id: int) -> None:
        """Delete `user_id`'s item `memory_id`, if it exists."""
        pool = await self._pool()
        await pool.execute(
            f"DELETE FROM {_MEMORY_ITEMS_TABLE} WHERE id = $1 AND user_id IS NOT DISTINCT FROM $2",
            memory_id,
            user_id,
        )


class PostgresKnowledgeStore:
    """`KnowledgeStore` backed by Postgres + `pgvector`: the `Knowledge(store=...)` escape hatch."""

    def __init__(self, dsn: str = DEFAULT_POSTGRES_DSN, *, dimensions: int) -> None:
        """Store which Postgres database this store's chunks/vectors live in, and vector size."""
        self.dsn = dsn
        self.dimensions = dimensions

    async def _pool(self) -> asyncpg.Pool:
        return await _connect(self.dsn, _knowledge_ddl(self.dimensions))

    async def add(self, *, text: str, source: str, embedding: list[float]) -> int:
        """Store one already-embedded chunk, returning its new id."""
        pool = await self._pool()
        row = await pool.fetchrow(
            f"""
            INSERT INTO {_KNOWLEDGE_ITEMS_TABLE} (text, source, embedding)
            VALUES ($1, $2, $3)
            RETURNING id
            """,
            text,
            source,
            embedding,
        )
        assert row is not None
        return row["id"]

    async def search(self, *, embedding: list[float], k: int) -> list[KnowledgeMatch]:
        """Return the `k` chunks closest to `embedding`, nearest first."""
        pool = await self._pool()
        rows = await pool.fetch(
            f"""
            SELECT id, text, source, embedding <-> $1 AS distance
            FROM {_KNOWLEDGE_ITEMS_TABLE}
            ORDER BY embedding <-> $1
            LIMIT $2
            """,
            embedding,
            k,
        )
        return [
            KnowledgeMatch(
                id=row["id"], text=row["text"], source=row["source"], distance=row["distance"]
            )
            for row in rows
        ]

    async def clear(self) -> None:
        """Delete every stored chunk, ahead of a fresh `Knowledge.ingest()`."""
        pool = await self._pool()
        await pool.execute(f"DELETE FROM {_KNOWLEDGE_ITEMS_TABLE}")


__all__ = [
    "DEFAULT_POSTGRES_DSN",
    "PostgresKnowledgeStore",
    "PostgresMemoryStore",
    "PostgresSession",
]
