"""Tests for `runa.db.vectors`, the storage `Memory` and `Knowledge` share.

The contract lives in `tests/contracts/vector.py` and is driven here over the backends `runa.db`
can resolve without a live Postgres; `tests/test_postgres.py` drives the same checks against
`PostgresVectorStore`. What stays here is what is one backend's own: the DDL each spec generates,
and the in-process adapter's module-level tables.
"""

import asyncio
from pathlib import Path

import pytest
from contracts.vector import CONTRACT, DIMENSIONS, Build, Check, flat, partitioned

from runa import db
from runa.db.vectors import Column, VectorSpec
from runa.db.vectors.ephemeral import EphemeralVectorStore
from runa.db.vectors.sqlite import SQLiteVectorStore


@pytest.fixture(params=["sqlite", "ephemeral"])
def build(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Build:
    """A `VectorStore` factory, resolved by `runa.db` the way each concern's store gets one.

    Through the environment variable rather than by naming an adapter, since that is the only
    thing that decides: `sqlite://` with four slashes is an absolute path, which is how a test
    keeps the local backend's file in its own `tmp_path`.
    """
    if request.param == "ephemeral":
        monkeypatch.setenv(db.DATABASE_URL_ENV, "memory://")
    else:
        monkeypatch.setenv(db.DATABASE_URL_ENV, f"sqlite:///{tmp_path / 'runa.db'}")
    return db.vector_store


@pytest.mark.parametrize("check", CONTRACT, ids=lambda check: check.__name__)
def test_vector_store_contract(build: Build, check: Check) -> None:
    """Every local backend answers the `VectorStore` contract the same way."""
    asyncio.run(check(build, "owner-1"))


def test_the_local_backend_names_its_two_tables_after_the_spec(tmp_path: Path) -> None:
    """`vec0` stores no payload, so one local store is an items table and a vectors table."""
    store = SQLiteVectorStore(partitioned("memory"), tmp_path / "runa.db")

    assert store.items == "memory_items"
    assert store.vectors == "memory_vectors"


def test_a_partitioned_spec_declares_a_vec0_partition_key(tmp_path: Path) -> None:
    """`partition_by` becomes a `PARTITION KEY`, not a `WHERE` filter, so top-k stays per owner."""
    ddl = SQLiteVectorStore(partitioned(), tmp_path / "runa.db").ddl

    assert "owner TEXT PARTITION KEY" in ddl
    assert f"embedding float[{DIMENSIONS}]" in ddl


def test_an_unpartitioned_spec_declares_no_partition_key(tmp_path: Path) -> None:
    """A `Knowledge`-shaped spec is application-scoped, so its vectors table has only a vector."""
    ddl = SQLiteVectorStore(flat(), tmp_path / "runa.db").ddl

    assert "PARTITION KEY" not in ddl


def test_a_required_column_is_declared_not_null(tmp_path: Path) -> None:
    """`Column(required=True)` is the only thing that puts `NOT NULL` in the generated DDL."""
    spec = VectorSpec(
        name="decl",
        dimensions=DIMENSIONS,
        columns={"needed": Column(required=True), "optional": Column()},
    )

    ddl = SQLiteVectorStore(spec, tmp_path / "runa.db").ddl

    assert "needed TEXT NOT NULL" in ddl
    assert "optional TEXT," in ddl


def test_a_json_column_is_declared_text_locally(tmp_path: Path) -> None:
    """A JSON column holds its value encoded, so SQLite declares it `TEXT` like any other."""
    ddl = SQLiteVectorStore(partitioned(), tmp_path / "runa.db").ddl

    assert "extra TEXT" in ddl


def test_two_in_process_stores_for_one_spec_share_their_rows() -> None:
    """The in-process tables live at module level, the way a database keeps them on disk.

    Two `db.vector_store()` calls have to see each other's writes, so a fresh set of rows per
    instance would make the whole backend a no-op.
    """
    spec = partitioned("shared")

    async def add_then_read() -> int:
        await EphemeralVectorStore(spec).add(
            payload={"owner": "u", "text": "a", "extra": None}, embedding=[1.0, 0.0, 0.0, 0.0]
        )
        found = await EphemeralVectorStore(spec).nearest(
            embedding=[1.0, 0.0, 0.0, 0.0], k=5, partition="u"
        )
        return len(found)

    assert asyncio.run(add_then_read()) == 1


def test_reset_ephemeral_empties_the_in_process_vector_tables() -> None:
    """`db.reset_ephemeral()` reaches this backend too, which is what the test fixture relies on."""
    spec = partitioned("resettable")

    async def add() -> None:
        await EphemeralVectorStore(spec).add(
            payload={"owner": "u", "text": "a", "extra": None}, embedding=[1.0, 0.0, 0.0, 0.0]
        )

    asyncio.run(add())
    db.reset_ephemeral()

    async def count() -> int:
        return len(
            await EphemeralVectorStore(spec).nearest(
                embedding=[1.0, 0.0, 0.0, 0.0], k=5, partition="u"
            )
        )

    assert asyncio.run(count()) == 0
