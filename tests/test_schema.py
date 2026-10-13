"""Tests for `runa.db.schema`: one declaration, two dialects, and an existing database."""

import sqlite3
from dataclasses import replace

from runa.db.schema import (
    POSTGRES,
    SQLITE,
    Column,
    Index,
    SchemaOutOfDate,
    Table,
    ddl,
    reconcile,
)

_EVERY_KIND = Table(
    "widgets",
    columns=(
        Column("id", "serial"),
        Column("name", "text"),
        Column("note", "text", null=True),
        Column("score", "float"),
        Column("count", "int"),
        Column("owner_id", "bigint", references="owners(id)"),
        Column("passed", "bool"),
        Column("created_at", "timestamp", default_now=True),
    ),
    indexes=(Index("name", "name, id DESC"),),
)


def test_sqlite_and_postgres_render_the_same_table_in_their_own_dialect() -> None:
    """Every difference between the two scripts is one of the dialect's own substitutions."""
    assert ddl(SQLITE, _EVERY_KIND) == (
        "CREATE TABLE IF NOT EXISTS widgets (\n"
        "    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
        "    name TEXT NOT NULL,\n"
        "    note TEXT,\n"
        "    score REAL NOT NULL,\n"
        "    count INTEGER NOT NULL,\n"
        "    owner_id INTEGER NOT NULL REFERENCES owners(id) ON DELETE CASCADE,\n"
        "    passed INTEGER NOT NULL,\n"
        "    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP\n"
        ");\n"
        "CREATE INDEX IF NOT EXISTS idx_widgets_name ON widgets (name, id DESC);\n"
    )
    assert ddl(POSTGRES, _EVERY_KIND) == (
        "CREATE TABLE IF NOT EXISTS widgets (\n"
        "    id BIGSERIAL PRIMARY KEY,\n"
        "    name TEXT NOT NULL,\n"
        "    note TEXT,\n"
        "    score DOUBLE PRECISION NOT NULL,\n"
        "    count INTEGER NOT NULL,\n"
        "    owner_id BIGINT NOT NULL REFERENCES owners(id) ON DELETE CASCADE,\n"
        "    passed BOOLEAN NOT NULL,\n"
        "    created_at TIMESTAMPTZ NOT NULL DEFAULT now()\n"
        ");\n"
        "CREATE INDEX IF NOT EXISTS idx_widgets_name ON widgets (name, id DESC);\n"
    )


def test_a_composite_primary_key_is_a_table_constraint() -> None:
    """A table whose identity is two columns declares it once, for both backends."""
    table = Table(
        "pairs",
        columns=(Column("left", "text"), Column("right", "text")),
        primary_key=("left", "right"),
    )

    for dialect in (SQLITE, POSTGRES):
        assert "PRIMARY KEY (left, right)\n);" in ddl(dialect, table)


def test_placeholders_follow_the_dialect_and_the_column_count() -> None:
    """`?` locally, `$1`-numbered on Postgres, one per column of a whole row."""
    assert _EVERY_KIND.placeholders(SQLITE) == ", ".join("?" * 8)
    assert _EVERY_KIND.placeholders(POSTGRES) == "$1, $2, $3, $4, $5, $6, $7, $8"
    assert _EVERY_KIND.column_names[:2] == ("id", "name")


def test_one_placeholder_is_the_whole_of_what_a_predicate_needs_of_a_dialect() -> None:
    """The substitution the three shared `WHERE` fragments are written against, and the only one.

    `session/store.py`'s `agent_filter`, `tracing/store.py`'s `listing_clause` and
    `eval/store.py`'s `baseline_clause` each ask for a numbered placeholder and nothing else, so
    none of them branches on `numbered_placeholders` itself.
    """
    assert (SQLITE.placeholder(1), SQLITE.placeholder(3)) == ("?", "?")
    assert (POSTGRES.placeholder(1), POSTGRES.placeholder(3)) == ("$1", "$3")


def test_the_rendered_script_is_executable_and_idempotent() -> None:
    """SQLite accepts the whole script, and applying it to the same file again is a no-op.

    Both `db/sqlite.py` and `db/pool.py` apply an adapter's DDL on every connect rather than
    tracking whether a database is new, so "run twice" is the normal case, not an edge one.
    """
    conn = sqlite3.connect(":memory:")
    script = ddl(SQLITE, _EVERY_KIND)
    conn.executescript(script)
    conn.executescript(script)

    columns = [row[1] for row in conn.execute("PRAGMA table_info(widgets)")]
    assert tuple(columns) == _EVERY_KIND.column_names


def test_raw_sql_is_rendered_verbatim_and_never_reconciled() -> None:
    """`db/vectors`'s stores hand over `CREATE` text this module can't read; it passes through."""
    raw = "CREATE VIRTUAL TABLE IF NOT EXISTS vecs USING vec0(embedding float[4]);"

    assert ddl(SQLITE, raw) == f"{raw}\n"
    assert reconcile(SQLITE, [("vecs", "rowid")], raw) == []


def test_a_column_added_to_a_declaration_is_added_to_an_existing_database() -> None:
    """The probe `CREATE TABLE IF NOT EXISTS` fails: a release adds a column, the file predates it.

    Applying the new `ddl` to the old file is a no-op, so without `reconcile` the next query is
    `no such column`. `eval_cases.trace_id` is the one this actually happened to.
    """
    grown = replace(
        _EVERY_KIND, columns=(*_EVERY_KIND.columns, Column("trace_id", "text", null=True))
    )
    conn = sqlite3.connect(":memory:")
    conn.executescript(ddl(SQLITE, _EVERY_KIND))  # the database as an earlier release left it
    conn.execute(
        "INSERT INTO widgets (name, score, count, owner_id, passed) VALUES (?,?,?,?,?)",
        ("w", 1.0, 1, 1, 0),
    )

    conn.executescript(ddl(SQLITE, grown))
    statements = reconcile(SQLITE, conn.execute(SQLITE.columns_query), grown)
    assert statements == ["ALTER TABLE widgets ADD COLUMN trace_id TEXT"]
    for statement in statements:
        conn.execute(statement)

    assert conn.execute("SELECT trace_id FROM widgets").fetchall() == [(None,)]
    # And applying it again adds nothing: the second connect is the normal case, not an edge one.
    assert reconcile(SQLITE, conn.execute(SQLITE.columns_query), grown) == []


def test_a_column_that_cannot_be_added_names_itself_instead_of_being_skipped() -> None:
    """The case that needs a human: no value exists for the rows already in the table.

    Loud rather than silent, because silent means the same `no such column` the declaration was
    supposed to prevent -- and because both backends refuse this `ALTER TABLE` anyway.
    """
    existing = [("widgets", column) for column in _EVERY_KIND.column_names]

    for column in (
        Column("tier", "text"),  # NOT NULL, nothing to put in it
        Column("uid", "text", primary_key=True),
        Column("seen_at", "timestamp", null=True, default_now=True),  # non-constant default
        Column("other_id", "serial"),
    ):
        grown = replace(_EVERY_KIND, columns=(*_EVERY_KIND.columns, column))
        for dialect in (SQLITE, POSTGRES):
            try:
                reconcile(dialect, existing, grown)
            except SchemaOutOfDate as exc:
                assert column.name in str(exc)
                assert "widgets" in str(exc)
            else:
                raise AssertionError(f"{column.name} should not be addable")


def test_a_column_the_declaration_no_longer_has_is_left_alone() -> None:
    """An older release reading a newer database is not a failure: every adapter selects by name."""
    existing = [("widgets", name) for name in (*_EVERY_KIND.column_names, "from_the_future")]

    assert reconcile(SQLITE, existing, _EVERY_KIND) == []
