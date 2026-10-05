"""Tests for `runa.db.schema`: one declaration, two dialects."""

import sqlite3

from runa.db.schema import POSTGRES, SQLITE, Column, Index, Table, ddl

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
