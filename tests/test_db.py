"""Tests for `runa.db`, the one place that decides where state lives.

The dispatch tests do not need a live Postgres: every adapter stores its URL and connects lazily
on first query, so constructing one is enough to prove the resolver picked it. The live-database
behavior is `tests/test_postgres.py`'s job.
"""

from pathlib import Path

import pytest

from runa import db

_URL = "postgresql://runa:runa@localhost:5432/runa"


def test_an_unset_variable_means_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """The zero-setup default: no variable, no Postgres, one SQLite file."""
    monkeypatch.delenv("RUNA_DATABASE_URL", raising=False)

    assert db.shared_url() is None
    assert db.sqlite_path() == Path("db/runa.db")


def test_a_postgres_url_is_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `postgresql://` URL is what every adapter branches on."""
    monkeypatch.setenv("RUNA_DATABASE_URL", _URL)

    assert db.shared_url() == _URL


def test_a_postgres_scheme_alias_is_shared_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """`postgres://` is the same database; plenty of platforms hand out that spelling."""
    monkeypatch.setenv("RUNA_DATABASE_URL", "postgres://runa@localhost/runa")

    assert db.shared_url() == "postgres://runa@localhost/runa"


def test_an_explicit_sqlite_url_is_not_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    """Saying `sqlite://` out loud must mean the same thing as saying nothing."""
    monkeypatch.setenv("RUNA_DATABASE_URL", "sqlite:///data/runa.db")

    assert db.shared_url() is None
    assert db.sqlite_path() == Path("data/runa.db")


def test_four_slashes_is_an_absolute_sqlite_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `DATABASE_URL` convention: three slashes relative, four absolute."""
    monkeypatch.setenv("RUNA_DATABASE_URL", "sqlite:////tmp/runa.db")

    assert db.sqlite_path() == Path("/tmp/runa.db")


def test_an_unknown_scheme_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo must stop the app, not silently give one replica its own private history."""
    monkeypatch.setenv("RUNA_DATABASE_URL", "mysql://runa@localhost/runa")

    with pytest.raises(db.InvalidDatabaseURL, match="postgresql://, sqlite:// or memory://"):
        db.shared_url()


def test_local_factories_return_the_sqlite_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every concern resolves local, so nothing needs the `postgres` extra installed."""
    monkeypatch.delenv("RUNA_DATABASE_URL", raising=False)

    from runa.cache.sqlite import SQLiteCache
    from runa.eval.sqlite import SQLiteEvalStore
    from runa.knowledge.sqlite import SQLiteKnowledgeStore
    from runa.memory.sqlite import SQLiteMemoryStore
    from runa.session.sqlite import SQLiteSession, SQLiteSessionStore
    from runa.tracing.sqlite import SQLiteTraceStore

    assert isinstance(db.session("s"), SQLiteSession)
    assert isinstance(db.sessions(), SQLiteSessionStore)
    assert isinstance(db.traces(), SQLiteTraceStore)
    assert isinstance(db.evals(), SQLiteEvalStore)
    assert isinstance(db.memory_store(dimensions=4), SQLiteMemoryStore)
    assert isinstance(db.knowledge_store(dimensions=4), SQLiteKnowledgeStore)
    assert isinstance(db.cache(), SQLiteCache)


def test_shared_factories_return_the_postgres_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point: one variable moves every concern, with no call site changed.

    Sessions, memory and knowledge used to need an explicit `PostgresX(...)` while traces and
    eval history followed the variable, so a deployment could read its own traces from Postgres
    and its sessions from a file no other replica had.
    """
    pytest.importorskip("asyncpg")
    monkeypatch.setenv("RUNA_DATABASE_URL", _URL)

    from runa.cache.postgres import PostgresCache
    from runa.eval.postgres import PostgresEvalStore
    from runa.knowledge.postgres import PostgresKnowledgeStore
    from runa.memory.postgres import PostgresMemoryStore
    from runa.session.postgres import PostgresSession, PostgresSessionStore
    from runa.tracing.postgres import PostgresTraceStore

    assert isinstance(db.session("s"), PostgresSession)
    assert isinstance(db.sessions(), PostgresSessionStore)
    assert isinstance(db.traces(), PostgresTraceStore)
    assert isinstance(db.evals(), PostgresEvalStore)
    assert isinstance(db.memory_store(dimensions=4), PostgresMemoryStore)
    assert isinstance(db.knowledge_store(dimensions=4), PostgresKnowledgeStore)
    assert isinstance(db.cache(), PostgresCache)


def test_a_shared_deployment_ignores_the_project(monkeypatch: pytest.MonkeyPatch) -> None:
    """A project only locates the local file, so a shared deployment has nothing to apply it to."""
    pytest.importorskip("asyncpg")
    monkeypatch.setenv("RUNA_DATABASE_URL", _URL)

    from runa.session.postgres import PostgresSession

    db.use_project(Path("/somewhere/else"))
    session = db.session("s")

    assert isinstance(session, PostgresSession)
    assert session.url == _URL


def test_use_project_moves_every_local_concern(monkeypatch: pytest.MonkeyPatch) -> None:
    """All seven concerns land in one file, which is the whole point of resolving them here.

    `root` used to be a per-factory argument, which meant four concerns took it and three could
    not: `Memory` and `Knowledge` are built inside `Agent.__init__`, which has no project root to
    pass. Pointing at another project then split one app's state across two files -- sessions,
    traces and eval history under `root`, memory, knowledge and the cache under the cwd. The
    assertion that matters is that this list has no exceptions left in it.
    """
    monkeypatch.delenv("RUNA_DATABASE_URL", raising=False)
    root = Path("other/project")
    expected = root / "db" / "runa.db"

    db.use_project(root)

    assert db.sqlite_path() == expected
    assert db.session("s").db_path == expected  # type: ignore[attr-defined]
    assert db.sessions().db_path == expected  # type: ignore[attr-defined]
    assert db.traces().db_path == expected  # type: ignore[attr-defined]
    assert db.evals().db_path == expected  # type: ignore[attr-defined]
    assert db.memory_store(dimensions=4).db_path == expected  # type: ignore[attr-defined]
    assert db.knowledge_store(dimensions=4).db_path == expected  # type: ignore[attr-defined]
    assert db.cache().db_path == expected  # type: ignore[attr-defined]


def test_use_project_none_is_the_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default an app running in its own directory wants, and what a test resets to."""
    monkeypatch.delenv("RUNA_DATABASE_URL", raising=False)

    db.use_project(Path("other/project"))
    db.use_project(None)

    assert db.sqlite_path() == Path("db/runa.db")


def test_a_project_applies_an_explicit_sqlite_url_relative_to_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative `sqlite://` path is relative to the project, so `use_project` relocates it."""
    monkeypatch.setenv("RUNA_DATABASE_URL", "sqlite:///data/runa.db")

    db.use_project(Path("other/project"))

    assert db.sqlite_path() == Path("other/project/data/runa.db")


def test_a_project_leaves_an_absolute_sqlite_url_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """An absolute path is already an answer; the project has nothing to add to it."""
    monkeypatch.setenv("RUNA_DATABASE_URL", "sqlite:////var/lib/runa.db")

    db.use_project(Path("other/project"))

    assert db.sqlite_path() == Path("/var/lib/runa.db")


def test_a_directly_built_adapter_honors_the_sqlite_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing a local adapter by name must land on the same file the factories pick.

    These seven used to default to a `db/runa.db` constant of their own, so an app that set
    `sqlite:///data/runa.db` moved six concerns and left whichever one it built by hand behind
    on the old path -- silently, and only in the deployment that set the variable.
    """
    monkeypatch.setenv("RUNA_DATABASE_URL", "sqlite:///data/runa.db")

    from runa.cache.sqlite import SQLiteCache
    from runa.eval.sqlite import SQLiteEvalStore
    from runa.knowledge.sqlite import SQLiteKnowledgeStore
    from runa.memory.sqlite import SQLiteMemoryStore
    from runa.session.sqlite import SQLiteSession, SQLiteSessionStore
    from runa.tracing.sqlite import SQLiteTraceStore

    expected = Path("data/runa.db")
    assert SQLiteSession("user-42").db_path == expected
    assert SQLiteSessionStore().db_path == expected
    assert SQLiteTraceStore().db_path == expected
    assert SQLiteEvalStore().db_path == expected
    assert SQLiteCache().db_path == expected
    assert SQLiteMemoryStore(dimensions=3).db_path == expected
    assert SQLiteKnowledgeStore(dimensions=3).db_path == expected


def test_an_explicit_db_path_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sharp knife survives: `db_path=` points at some other file whatever the URL says."""
    monkeypatch.setenv("RUNA_DATABASE_URL", "sqlite:///data/runa.db")

    from runa.session.sqlite import SQLiteSession

    assert SQLiteSession("user-42", "other/runa.db").db_path == Path("other/runa.db")


def test_memory_url_resolves_every_concern_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """`memory://` is the third answer: no file, no server, all six concerns together.

    The promise `runa.db` exists to keep is that one variable moves everything, so a backend
    that only some concerns honored would be worse than none.
    """
    monkeypatch.setenv("RUNA_DATABASE_URL", "memory://")

    from runa.cache.memory import MemoryCache
    from runa.db.vectors.ephemeral import EphemeralVectorStore
    from runa.eval.ephemeral import EphemeralEvalStore
    from runa.knowledge.vector import VectorKnowledgeStore
    from runa.memory.vector import VectorMemoryStore
    from runa.session.ephemeral import EphemeralSession, EphemeralSessionStore
    from runa.tracing.ephemeral import EphemeralTraceStore

    assert db.ephemeral() is True
    assert db.shared_url() is None
    assert isinstance(db.session("s"), EphemeralSession)
    assert isinstance(db.sessions(), EphemeralSessionStore)
    assert isinstance(db.traces(), EphemeralTraceStore)
    assert isinstance(db.evals(), EphemeralEvalStore)
    assert isinstance(db.cache(), MemoryCache)

    # The two vector concerns have no named in-process adapter to name here: what makes them
    # ephemeral is the `VectorStore` underneath the mapping `runa.db` pairs them with.
    memory = db.memory_store(dimensions=4)
    knowledge = db.knowledge_store(dimensions=4)
    assert isinstance(memory, VectorMemoryStore)
    assert isinstance(knowledge, VectorKnowledgeStore)
    assert isinstance(memory._vectors, EphemeralVectorStore)
    assert isinstance(knowledge._vectors, EphemeralVectorStore)


def test_the_ephemeral_cache_is_one_cache_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two `db.cache()` calls see each other's writes, as two connections to a file would."""
    import asyncio

    monkeypatch.setenv("RUNA_DATABASE_URL", "memory://")

    asyncio.run(db.cache().set("k", "v"))

    assert asyncio.run(db.cache().get("k")) == "v"


def test_reset_ephemeral_empties_every_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """One call clears all six concerns, which is what makes `memory://` usable as a fixture."""
    import asyncio

    from runa.tracing import Trace

    monkeypatch.setenv("RUNA_DATABASE_URL", "memory://")
    db.traces().save(Trace(id="t1", name="A", start_time=0.0))
    asyncio.run(db.session("s").add_items([{"role": "user", "content": "hi"}]))
    asyncio.run(db.cache().set("k", "v"))

    db.reset_ephemeral()

    assert db.traces().list() == []
    assert db.sessions().listing() == []
    assert asyncio.run(db.cache().get("k")) is None


def test_memory_follows_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Memory()` picks up a shared database with no `store=` argument, same as `runa serve`."""
    pytest.importorskip("asyncpg")
    monkeypatch.setenv("RUNA_DATABASE_URL", _URL)

    from runa.memory import Memory
    from runa.memory.postgres import PostgresMemoryStore

    assert isinstance(Memory()._store, PostgresMemoryStore)


def test_a_local_connection_enforces_foreign_keys(tmp_path: Path) -> None:
    """SQLite ignores every `REFERENCES` clause unless the per-connection pragma is on.

    Retention is a `DELETE` on the parent table and nothing else (see `docs/deployment.md`), so
    the cascades have to actually fire.
    """
    from contextlib import closing

    from runa.db.sqlite import connect

    ddl = """
    CREATE TABLE IF NOT EXISTS parent (id TEXT PRIMARY KEY);
    CREATE TABLE IF NOT EXISTS child (
        id TEXT PRIMARY KEY,
        parent_id TEXT NOT NULL REFERENCES parent(id) ON DELETE CASCADE
    );
    """
    with closing(connect(tmp_path / "runa.db", ddl)) as conn:
        conn.execute("INSERT INTO parent VALUES ('p')")
        conn.execute("INSERT INTO child VALUES ('c', 'p')")
        conn.execute("DELETE FROM parent WHERE id = 'p'")

        assert conn.execute("SELECT COUNT(*) FROM child").fetchone()[0] == 0


def test_deleting_a_trace_takes_its_spans(tmp_path: Path) -> None:
    """The retention pass documented for `traces` reaches `spans` without naming them."""
    import sqlite3
    from contextlib import closing

    from runa.tracing import Span, Trace

    db_path = tmp_path / "db" / "runa.db"
    span = Span(
        id="s1",
        trace_id="t1",
        parent_id=None,
        name="SupportAgent",
        type="agent",
        start_time=0.0,
        end_time=1.0,
        status="ok",
    )
    db.use_project(tmp_path)
    db.traces().save(Trace(id="t1", name="SupportAgent", start_time=0.0, spans=[span]))

    from runa.db.sqlite import connect

    with closing(connect(db_path, "")) as conn:
        conn.execute("DELETE FROM traces WHERE id = 't1'")
        conn.commit()

    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM spans WHERE trace_id = 't1'").fetchone()[0] == 0
