"""Tests for `runa.db`, the one place that decides where state lives.

The dispatch tests do not need a live Postgres: every adapter stores its URL and connects lazily
on first query, so constructing one is enough to prove the resolver picked it. The live-database
behavior is `tests/test_postgres.py`'s job.
"""

from pathlib import Path

import pytest

from runa import _app, db
from runa._loop import run_sync
from runa.exceptions import UserError
from runa.knowledge import Knowledge
from runa.memory import Memory
from runa.project import use

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


def test_local_factories_return_stores_over_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every concern resolves local, so nothing needs the `postgres` extra installed."""
    monkeypatch.delenv("RUNA_DATABASE_URL", raising=False)

    from runa.db.sqlite import SQLiteDatabase
    from runa.db.vectors.sqlite import SQLiteVectorStore

    for store in (db.session("s"), db.sessions(), db.traces(), db.evals(), db.cache()):
        assert isinstance(store.database, SQLiteDatabase)  # type: ignore[attr-defined]
    assert isinstance(Memory(dimensions=4)._store, SQLiteVectorStore)
    assert isinstance(Knowledge(dimensions=4)._store, SQLiteVectorStore)


def test_shared_factories_return_stores_over_postgres(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point: one variable moves every concern, with no call site changed."""
    pytest.importorskip("asyncpg")
    monkeypatch.setenv("RUNA_DATABASE_URL", _URL)

    from runa.db.pool import PostgresDatabase
    from runa.db.vectors.postgres import PostgresVectorStore

    for store in (db.session("s"), db.sessions(), db.traces(), db.evals(), db.cache()):
        assert isinstance(store.database, PostgresDatabase)  # type: ignore[attr-defined]
    assert isinstance(Memory(dimensions=4)._store, PostgresVectorStore)
    assert isinstance(Knowledge(dimensions=4)._store, PostgresVectorStore)


def test_a_shared_deployment_ignores_the_project(monkeypatch: pytest.MonkeyPatch) -> None:
    """A project only locates the local file, so a shared deployment has nothing to apply it to."""
    pytest.importorskip("asyncpg")
    monkeypatch.setenv("RUNA_DATABASE_URL", _URL)

    from runa.db.pool import PostgresDatabase

    use(Path("/somewhere/else"))
    session = db.session("s")

    assert isinstance(session.database, PostgresDatabase)  # type: ignore[attr-defined]
    assert session.database.url == _URL  # type: ignore[attr-defined]


def test_a_project_moves_every_local_concern(monkeypatch: pytest.MonkeyPatch) -> None:
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

    use(root)

    assert db.sqlite_path() == expected
    assert db.session("s").database.location == expected  # type: ignore[attr-defined]
    assert db.sessions().database.location == expected  # type: ignore[attr-defined]
    assert db.traces().database.location == expected  # type: ignore[attr-defined]
    assert db.evals().database.location == expected  # type: ignore[attr-defined]
    assert Memory(dimensions=4)._store.db_path == expected  # type: ignore[attr-defined]
    assert Knowledge(dimensions=4)._store.db_path == expected  # type: ignore[attr-defined]
    assert db.cache().database.location == expected  # type: ignore[attr-defined]


def test_no_project_is_the_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default an app running in its own directory wants, and what a test resets to."""
    monkeypatch.delenv("RUNA_DATABASE_URL", raising=False)

    use(Path("other/project"))
    use(None)

    assert db.sqlite_path() == Path("db/runa.db")


def test_the_same_project_twice_is_one_project(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two startup paths naming one project is the normal case, not a conflict.

    `runa ui` says it twice by design: `cli/main` from the cwd before dispatching, then
    `web.app.create_app(root)` while building. Saying the same thing again configures nothing new.
    """
    monkeypatch.delenv("RUNA_DATABASE_URL", raising=False)
    root = Path("other/project")

    use(root)
    use(root)

    assert db.sqlite_path() == root / "db" / "runa.db"


def test_a_second_project_leaves_the_first_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process reads one app, so the second one is an error instead of a silent repoint.

    Two Runa apps mounted in one process used to leave both resolving against whichever root was
    built last -- including runs already in flight, whose sessions and memory would land in
    another app's file. The root `runa.db` resolves against is the same value `runa.project`
    imports from, so a refused second app moves neither.
    """
    monkeypatch.delenv("RUNA_DATABASE_URL", raising=False)
    use(Path("one/project"))

    with pytest.raises(UserError):
        use(Path("other/project"))

    assert db.sqlite_path() == Path("one/project/db/runa.db")


def test_a_project_applies_an_explicit_sqlite_url_relative_to_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative `sqlite://` path is relative to the project, so `project.use` relocates it."""
    monkeypatch.setenv("RUNA_DATABASE_URL", "sqlite:///data/runa.db")

    use(Path("other/project"))

    assert db.sqlite_path() == Path("other/project/data/runa.db")


def test_a_project_leaves_an_absolute_sqlite_url_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """An absolute path is already an answer; the project has nothing to add to it."""
    monkeypatch.setenv("RUNA_DATABASE_URL", "sqlite:////var/lib/runa.db")

    use(Path("other/project"))

    assert db.sqlite_path() == Path("/var/lib/runa.db")


def test_a_project_moves_a_file_path_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_app.project_path` answers the same question for state an app keeps as files, not rows.

    `Knowledge`'s source directory follows the root as its vectors do.
    """
    monkeypatch.delenv("RUNA_DATABASE_URL", raising=False)

    use(Path("other/project"))

    assert _app.project_path("app/knowledge") == Path("other/project/app/knowledge")
    assert _app.project_path(Path("/srv/corpus")) == Path("/srv/corpus")


def test_a_shared_deployment_still_places_a_file_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unlike `sqlite_path`, this is read whatever the database is: files are on disk either way."""
    monkeypatch.setenv("RUNA_DATABASE_URL", _URL)

    use(Path("other/project"))

    assert _app.project_path("app/knowledge") == Path("other/project/app/knowledge")


def test_a_bare_memory_or_knowledge_honors_the_sqlite_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """A store built with no arguments lands on the same file the factories pick."""
    monkeypatch.setenv("RUNA_DATABASE_URL", "sqlite:///data/runa.db")

    expected = Path("data/runa.db")
    assert Memory(dimensions=3)._store.db_path == expected  # type: ignore[attr-defined]
    assert Knowledge(dimensions=3)._store.db_path == expected  # type: ignore[attr-defined]


def test_at_puts_every_concern_in_the_database_it_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sharp knife: `db.at(url)` points every concern at some other database, together.

    Whatever the environment says, so a store built by hand for one concern can't split from
    another built by hand beside it.
    """
    monkeypatch.setenv("RUNA_DATABASE_URL", "sqlite:///data/runa.db")
    other = db.at("sqlite:///other/runa.db")

    expected = Path("other/runa.db")
    for store in (other.session("s"), other.sessions(), other.traces(), other.evals()):
        assert store.database.location == expected  # type: ignore[attr-defined]
    assert other.cache().database.location == expected  # type: ignore[attr-defined]
    assert other.paused_runs().database.location == expected  # type: ignore[attr-defined]
    assert db.session("s").database.location == Path("data/runa.db")  # type: ignore[attr-defined]


def test_at_a_shared_url_reaches_postgres_and_an_unknown_one_is_a_user_error() -> None:
    """`at` takes the forms `RUNA_DATABASE_URL` does; a typo in code is the code's mistake."""
    pytest.importorskip("asyncpg")

    shared = db.at("postgresql://example/db")
    assert type(shared.session("s").database).__name__ == "PostgresDatabase"  # type: ignore[attr-defined]
    assert shared.shared_url() == "postgresql://example/db"

    with pytest.raises(UserError, match="postgresql://, sqlite:// or memory://"):
        db.at("mysql://runa@localhost/runa")


def test_memory_url_resolves_every_concern_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """`memory://` is the third answer: no file, no server, all six concerns together.

    The promise `runa.db` exists to keep is that one variable moves everything, so a backend
    that only some concerns honored would be worse than none. Every one of them lands in the same
    in-memory database, through the adapters a local deployment uses.
    """
    monkeypatch.setenv("RUNA_DATABASE_URL", "memory://")

    from runa.db.sqlite import MemoryDatabase

    database = db.sqlite_database()

    assert db.ephemeral() is True
    assert db.shared_url() is None
    assert isinstance(database, MemoryDatabase)
    for store in (db.session("s"), db.sessions(), db.traces(), db.evals(), db.cache()):
        assert store.database.location is database  # type: ignore[attr-defined]
    assert Memory(dimensions=4)._store.db_path is database  # type: ignore[attr-defined]
    assert Knowledge(dimensions=4)._store.db_path is database  # type: ignore[attr-defined]


def test_the_in_memory_cache_is_one_cache_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two `db.cache()` calls see each other's writes, as two connections to a file would."""
    import asyncio

    monkeypatch.setenv("RUNA_DATABASE_URL", "memory://")

    asyncio.run(db.cache().set("k", "v"))

    assert asyncio.run(db.cache().get("k")) == "v"


def test_the_in_memory_database_takes_concurrent_writers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent turns write from worker threads at once, and each waits out the others' locks.

    A shared-cache in-memory database would fail all but one of them with `database table is
    locked` rather than wait; this is the property `memdb` is chosen for.
    """
    import asyncio

    monkeypatch.setenv("RUNA_DATABASE_URL", "memory://")

    async def write(index: int) -> None:
        session = db.session(f"s{index}")
        for turn in range(10):
            await session.add_items([{"role": "user", "content": f"{index}-{turn}"}])

    async def write_all() -> None:
        await asyncio.gather(*(write(index) for index in range(8)))

    asyncio.run(write_all())

    assert len(run_sync(db.sessions().listing())) == 8
    assert all(len(run_sync(db.sessions().messages(f"s{index}"))) == 10 for index in range(8))


def test_reset_ephemeral_empties_every_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """One call clears all six concerns, which is what makes `memory://` usable as a fixture."""
    import asyncio

    from runa.tracing import Trace

    monkeypatch.setenv("RUNA_DATABASE_URL", "memory://")
    run_sync(db.traces().save(Trace(id="t1", name="A", start_time=0.0)))
    asyncio.run(db.session("s").add_items([{"role": "user", "content": "hi"}]))
    asyncio.run(db.cache().set("k", "v"))

    db.reset_ephemeral()

    assert run_sync(db.traces().list()) == []
    assert run_sync(db.sessions().listing()) == []
    assert asyncio.run(db.cache().get("k")) is None


def test_reset_ephemeral_empties_the_database_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An app's own test suite calls it mid-process, so the app and its tracing policy stay put."""
    from runa.tracing import config, observe

    monkeypatch.setenv("RUNA_DATABASE_URL", "memory://")
    use(tmp_path)
    observe(capture_inputs=False)

    db.reset_ephemeral()

    assert _app.project_path("app") == tmp_path / "app"
    assert config.policy().capture_inputs is False


def test_a_process_reset_forgets_every_fact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The app, the tracing policy and the `memory://` tables go together, in one call."""
    import asyncio

    from runa.tracing import config, observe

    monkeypatch.setenv("RUNA_DATABASE_URL", "memory://")
    use(tmp_path)
    observe(capture_inputs=False)
    asyncio.run(db.cache().set("k", "v"))

    _app.reset()

    assert _app.project_path("app") == Path("app")
    assert config.policy().capture_inputs is True
    assert asyncio.run(db.cache().get("k")) is None


def test_memory_follows_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Memory()` picks up a shared database with no `store=` argument, same as `runa serve`."""
    pytest.importorskip("asyncpg")
    monkeypatch.setenv("RUNA_DATABASE_URL", _URL)

    from runa.db.vectors.postgres import PostgresVectorStore

    assert isinstance(Memory()._store, PostgresVectorStore)


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


def test_a_local_connection_grows_a_file_that_predates_a_column(tmp_path: Path) -> None:
    """A `runa.db` outlives the release that created it, so a connect has to catch it up.

    `CREATE TABLE IF NOT EXISTS` alone can't: on the existing table it is a no-op and the next
    query is `no such column`. This is `eval_cases.trace_id`'s story, told with `agent_sessions`
    so it holds for whichever concern's declaration grows next.
    """
    from contextlib import closing
    from dataclasses import replace

    from runa.db.schema import Column
    from runa.db.sqlite import connect
    from runa.session.store import MESSAGES, SESSIONS

    db_path = tmp_path / "db" / "runa.db"
    with closing(connect(db_path, SESSIONS, MESSAGES)) as conn:
        conn.execute(f"INSERT INTO {SESSIONS.name} (session_id) VALUES ('s1')")
        conn.commit()

    grown = replace(SESSIONS, columns=(*SESSIONS.columns, Column("title", "text", null=True)))
    with closing(connect(db_path, grown, MESSAGES)) as conn:
        rows = conn.execute(f"SELECT session_id, title FROM {grown.name}").fetchall()

    assert rows == [("s1", None)]


def test_a_local_connection_applies_a_schema_once_per_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whether a file predates a column is a question about releases, not about queries.

    Every local adapter opens a connection per call -- a session-backed turn opens three -- and
    each one used to re-run the whole `CREATE TABLE IF NOT EXISTS` script and a whole-database
    column scan to re-answer a question whose answer cannot change while the process runs.

    What still has to be asked again is each of the three cases below: another adapter's tables in
    the same file, a declaration that has grown (`test_a_local_connection_grows_a_file_...`), and
    a file that is no longer the file this process applied anything to.
    """
    import asyncio

    from runa.db import sqlite as sqlite_db
    from runa.db.schema import Declaration, Dialect

    applied: list[tuple[str, ...]] = []
    render = sqlite_db.ddl

    def recording(dialect: Dialect, *tables: Declaration) -> str:
        applied.append(tuple("raw" if isinstance(t, str) else t.name for t in tables))
        return render(dialect, *tables)

    monkeypatch.setattr(sqlite_db, "ddl", recording)
    db_path = tmp_path / "db" / "runa.db"
    session = db.at(f"sqlite:///{db_path}").session("conv-1")

    asyncio.run(session.add_items([{"role": "user", "content": "hi"}]))
    for _ in range(5):
        assert len(asyncio.run(session.get_items())) == 1
    assert applied == [("agent_sessions", "agent_messages")]

    asyncio.run(
        db.at(f"sqlite:///{db_path}").cache().set("k", "v")
    )  # another concern's tables, same file
    assert applied[1:] == [("cache_entries",)]

    db_path.unlink()  # a file deleted underneath a live process is a new file, not a known one
    asyncio.run(session.add_items([{"role": "user", "content": "again"}]))
    assert applied[2:] == [("agent_sessions", "agent_messages")]
    assert len(asyncio.run(session.get_items())) == 1


def test_a_local_read_leaves_no_file_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Answering "nothing here yet" is a read, and a read creates nothing.

    One rule, kept by `db/sqlite.py`'s `read` for all four local concerns rather than stated in
    one adapter's docstring: the session listing was the only one that honored it, so a `runa ui`
    page or a cache miss answering the same way still left a `db/runa.db` behind.
    """
    import asyncio

    monkeypatch.delenv("RUNA_DATABASE_URL", raising=False)
    use(tmp_path)

    assert run_sync(db.sessions().listing()) == []
    assert run_sync(db.traces().list()) == []
    assert run_sync(db.traces().get("t1")) is None
    assert run_sync(db.evals().list()) == []
    assert run_sync(db.evals().baseline("support_agent")) is None
    assert asyncio.run(db.cache().get("k")) is None

    assert list(tmp_path.iterdir()) == []


def test_a_local_read_answers_no_rows_only_for_a_table_that_is_absent(tmp_path: Path) -> None:
    """Absence is the one condition a read answers with no rows; a broken table still raises.

    The failure this replaces: a blanket `except sqlite3.OperationalError: return []` around the
    query covered a renamed column as well as a missing table, so `runa chat --list` reported no
    sessions while `runa chat --show` printed the transcript of one of them. Tested as a condition
    now, and a read reconciles like any other connection, so a column the declaration still wants
    is the operator error it always was.
    """
    import asyncio
    import sqlite3
    from contextlib import closing

    from runa.db.schema import SchemaOutOfDate

    db_path = tmp_path / "db" / "runa.db"
    here = db.at(f"sqlite:///{db_path}")
    asyncio.run(here.session("conv-1").add_items([{"role": "user", "content": "hi"}]))
    store = here.sessions()
    assert [summary.id for summary in run_sync(store.listing())] == ["conv-1"]

    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("ALTER TABLE agent_sessions RENAME COLUMN updated_at TO touched_at")
        conn.commit()

    with pytest.raises(SchemaOutOfDate, match="agent_sessions table has no updated_at column"):
        run_sync(store.listing())


def test_a_local_read_grows_a_file_that_predates_a_column(tmp_path: Path) -> None:
    """A read selects the columns this release declares, so it catches an old file up too.

    The other half of opening the file by hand: the session read side was outside
    `db/sqlite.py` and therefore outside `reconcile`, so the one read path that could not be
    brought up to its declaration was also the one that swallowed the `no such column` a stale
    file produced.
    """
    from contextlib import closing
    from dataclasses import replace

    from runa.db.schema import Column
    from runa.db.sqlite import connect, read
    from runa.session.store import MESSAGES, SESSIONS

    db_path = tmp_path / "db" / "runa.db"
    with closing(connect(db_path, SESSIONS, MESSAGES)) as conn:
        conn.execute(f"INSERT INTO {SESSIONS.name} (session_id) VALUES ('s1')")
        conn.commit()

    grown = replace(SESSIONS, columns=(*SESSIONS.columns, Column("agent_name", "text", null=True)))
    with read(db_path, grown, MESSAGES) as conn:
        assert conn is not None
        rows = conn.execute(f"SELECT session_id, agent_name FROM {grown.name}").fetchall()

    assert [tuple(row) for row in rows] == [("s1", None)]


def test_a_local_read_of_another_concerns_file_is_no_rows(tmp_path: Path) -> None:
    """One `runa.db`, four sets of tables: a file is not the same question as a table.

    A project that has traced a run but never used a session has the file and not the tables, and
    "no sessions" is still the answer -- without creating them on the way to saying so.
    """
    from runa.db.sqlite import read
    from runa.session.store import MESSAGES, SESSIONS
    from runa.tracing import Trace

    use(tmp_path)
    run_sync(db.traces().save(Trace(id="t1", name="SupportAgent", start_time=0.0, spans=[])))

    assert run_sync(db.sessions().listing()) == []
    with read(tmp_path / "db" / "runa.db", SESSIONS, MESSAGES) as conn:
        assert conn is None


@pytest.mark.usefixtures("project_database")
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
    use(tmp_path)
    run_sync(db.traces().save(Trace(id="t1", name="SupportAgent", start_time=0.0, spans=[span])))

    from runa.db.sqlite import connect

    with closing(connect(db_path, "")) as conn:
        conn.execute("DELETE FROM traces WHERE id = 't1'")
        conn.commit()

    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM spans WHERE trace_id = 't1'").fetchone()[0] == 0
