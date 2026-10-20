"""What `db/handle.py`'s `Database` promises, held against the SQLite implementation.

The concerns' own contracts (`tests/contracts/`) say what a store does over a database. These are
the statements underneath them: numbered parameters, row counts, transactions, and a read that
creates nothing. `tests/test_postgres.py` runs the concerns over the Postgres handle.
"""

import asyncio
from pathlib import Path

import pytest

from runa.db.schema import Column, Table
from runa.db.sqlite import SQLiteDatabase

_NOTES = Table("notes", columns=(Column("id", "text", primary_key=True), Column("body", "text")))


def _database(tmp_path: Path) -> SQLiteDatabase:
    return SQLiteDatabase(tmp_path / "db" / "runa.db", _NOTES)


def test_numbered_parameters_bind_by_index_and_may_repeat(tmp_path: Path) -> None:
    """`$2` before `$1`, and `$1` twice, as Postgres allows."""
    database = _database(tmp_path)

    async def scenario() -> list[str]:
        await database.execute("INSERT INTO notes (id, body) VALUES ($2, $1)", "text", "a")
        rows = await database.fetch("SELECT body FROM notes WHERE id = $1 OR body = $1", "a")
        return [row["body"] for row in rows]

    assert asyncio.run(scenario()) == ["text"]


def test_execute_reports_how_many_rows_it_changed(tmp_path: Path) -> None:
    """The count a `claim` is decided on."""
    database = _database(tmp_path)

    async def scenario() -> tuple[int, int, int]:
        await database.executemany(
            "INSERT INTO notes (id, body) VALUES ($1, $2)", [("a", "x"), ("b", "x"), ("c", "y")]
        )
        return (
            await database.execute("DELETE FROM notes WHERE body = $1", "x"),
            await database.execute("DELETE FROM notes WHERE body = $1", "x"),
            await database.fetchval("SELECT COUNT(*) FROM notes"),
        )

    assert asyncio.run(scenario()) == (2, 0, 1)


def test_fetchrow_and_fetchval_answer_none_for_no_rows(tmp_path: Path) -> None:
    """A missing row is `None`, whether or not the table exists yet."""
    database = _database(tmp_path)

    async def scenario() -> tuple[object, object]:
        before = await database.fetchrow("SELECT * FROM notes WHERE id = $1", "a")
        await database.execute("INSERT INTO notes (id, body) VALUES ($1, $2)", "b", "x")
        return before, await database.fetchval("SELECT body FROM notes WHERE id = $1", "a")

    assert asyncio.run(scenario()) == (None, None)


def test_a_read_creates_nothing(tmp_path: Path) -> None:
    """Asking a project that has written nothing is an empty answer, not a `db/runa.db`."""
    database = _database(tmp_path)

    assert asyncio.run(database.fetch("SELECT * FROM notes")) == []
    assert not (tmp_path / "db").exists()


def test_a_transaction_commits_together_when_the_block_ends(tmp_path: Path) -> None:
    """Both inserts land, and a statement inside can read what the one before wrote."""
    database = _database(tmp_path)

    async def scenario() -> list[str]:
        async with database.transaction() as tx:
            await tx.execute("INSERT INTO notes (id, body) VALUES ($1, $2)", "a", "x")
            seen = await tx.fetchval("SELECT body FROM notes WHERE id = $1", "a")
            await tx.execute("INSERT INTO notes (id, body) VALUES ($1, $2)", "b", seen)
        return [row["id"] for row in await database.fetch("SELECT id FROM notes ORDER BY id")]

    assert asyncio.run(scenario()) == ["a", "b"]


def test_a_transaction_that_raises_leaves_nothing_behind(tmp_path: Path) -> None:
    """All or none: the insert before the failure is rolled back with it."""
    database = _database(tmp_path)

    async def scenario() -> list[str]:
        await database.execute("INSERT INTO notes (id, body) VALUES ($1, $2)", "kept", "x")
        with pytest.raises(RuntimeError):
            async with database.transaction() as tx:
                await tx.execute("INSERT INTO notes (id, body) VALUES ($1, $2)", "lost", "x")
                raise RuntimeError("boom")
        return [row["id"] for row in await database.fetch("SELECT id FROM notes")]

    assert asyncio.run(scenario()) == ["kept"]


def test_concurrent_writers_wait_for_each_other(tmp_path: Path) -> None:
    """Twenty writers on one file all land: SQLite's one writer per file never fails a caller."""
    database = _database(tmp_path)

    async def scenario() -> int:
        await asyncio.gather(
            *(
                database.execute("INSERT INTO notes (id, body) VALUES ($1, $2)", str(n), "x")
                for n in range(20)
            )
        )
        return await database.fetchval("SELECT COUNT(*) FROM notes")

    assert asyncio.run(scenario()) == 20
