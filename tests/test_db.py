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

    with pytest.raises(db.InvalidDatabaseURL, match="must start with postgresql:// or sqlite://"):
        db.shared_url()


def test_local_factories_return_the_sqlite_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every concern resolves local, so nothing needs the `postgres` extra installed."""
    monkeypatch.delenv("RUNA_DATABASE_URL", raising=False)

    from runa.cache.sqlite import SQLiteCache
    from runa.knowledge.sqlite import SQLiteKnowledgeStore
    from runa.memory.sqlite import SQLiteMemoryStore
    from runa.session.sqlite import SQLiteSession

    assert isinstance(db.session("s"), SQLiteSession)
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
    from runa.knowledge.postgres import PostgresKnowledgeStore
    from runa.memory.postgres import PostgresMemoryStore
    from runa.session.postgres import PostgresSession

    assert isinstance(db.session("s"), PostgresSession)
    assert isinstance(db.memory_store(dimensions=4), PostgresMemoryStore)
    assert isinstance(db.knowledge_store(dimensions=4), PostgresKnowledgeStore)
    assert isinstance(db.cache(), PostgresCache)


def test_a_shared_session_ignores_the_local_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """`db_path` is the local file's location, so a shared deployment has nothing to apply it to."""
    pytest.importorskip("asyncpg")
    monkeypatch.setenv("RUNA_DATABASE_URL", _URL)

    from runa.session.postgres import PostgresSession

    session = db.session("s", db_path=Path("/somewhere/else.db"))

    assert isinstance(session, PostgresSession)
    assert session.url == _URL


def test_a_local_session_honors_the_local_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """What `runa chat --root other/project` depends on."""
    monkeypatch.delenv("RUNA_DATABASE_URL", raising=False)

    session = db.session("s", db_path=Path("other/db/runa.db"))

    assert session.db_path == Path("other/db/runa.db")  # type: ignore[attr-defined]


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
    from runa.tracing.storage import save_trace

    db_path = tmp_path / "runa.db"
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
    save_trace(Trace(id="t1", name="SupportAgent", start_time=0.0, spans=[span]), db_path=db_path)

    from runa.db.sqlite import connect

    with closing(connect(db_path, "")) as conn:
        conn.execute("DELETE FROM traces WHERE id = 't1'")
        conn.commit()

    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM spans WHERE trace_id = 't1'").fetchone()[0] == 0
