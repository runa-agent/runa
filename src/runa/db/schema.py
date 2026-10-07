"""db/schema.py: a table declared once, rendered in whichever dialect the adapter speaks.

Four concerns keep rows in `runa.db`'s database -- sessions, traces, eval history and the cache --
and each has two SQL adapters, one per answer `runa.db` can give. Both adapters need the same
tables with the same columns and the same indexes, and until this module they each wrote their
own `CREATE TABLE` string, with a docstring asking the next person to keep the two in step. They
did not stay in step: the Postgres trace store carried indexes the SQLite one lacked, and the
SQLite eval store carried none at all, which nobody noticed because nothing but prose said they
had to match.

The whole difference between the two dialects is the handful of substitutions in `SQLITE` and
`POSTGRES` below -- `REAL` against `DOUBLE PRECISION`, `AUTOINCREMENT` against `BIGSERIAL`, and
two more. Everything else an adapter was repeating was the same text twice. So a concern declares
its `Table`s once, beside the code that turns a row into an object, and `ddl(dialect, *tables)`
renders them; a column added in the declaration appears in both backends or in neither.

What this is not is a query builder. Adapters still write their own SQL, because that is where
they genuinely differ (`INSERT OR REPLACE` against `ON CONFLICT ... EXCLUDED`, a batched
`ANY($1::text[])`), and a builder general enough to express those would be a shallower module than
the eight adapters it replaced. The one query fragment here is `Table.placeholders`, which is
dialect, not query: `?` against `$1`.

A concern whose adapters must phrase one predicate identically can keep that fragment beside its
own tables -- `session/store.py`'s `agent_filter` is the whole `WHERE` matching an agent's session
ids, placeholders and `ESCAPE` clause included, because half of that rule is not usable on its own.
That is a predicate a concern owns, not a query builder growing here.
"""

from dataclasses import dataclass, field
from typing import Literal

ColumnType = Literal["text", "int", "bigint", "float", "bool", "timestamp", "serial"]
"""The logical column types these four concerns actually store.

Logical, not SQL: `float` is `REAL` locally and `DOUBLE PRECISION` shared, and `bool` is an
integer in a backend that has no boolean. `serial` is an auto-assigned integer primary key, the
one type that renders as a whole column definition rather than a type name.
"""


@dataclass(frozen=True)
class Dialect:
    """One backend's answers to the only questions the two renderings disagree on.

    Adding a backend that speaks SQL means adding one of these, not another copy of every
    `CREATE TABLE`.
    """

    name: str
    types: dict[ColumnType, str]
    serial: str
    """The whole definition of a `serial` column, primary key included."""
    now: str
    """The expression a `default_now` column defaults to."""
    numbered_placeholders: bool
    """`$1, $2, ...` rather than `?, ?, ...`."""


SQLITE = Dialect(
    name="sqlite",
    types={
        "text": "TEXT",
        "int": "INTEGER",
        "bigint": "INTEGER",
        "float": "REAL",
        "bool": "INTEGER",
        "timestamp": "TIMESTAMP",
        "serial": "INTEGER",
    },
    serial="INTEGER PRIMARY KEY AUTOINCREMENT",
    now="CURRENT_TIMESTAMP",
    numbered_placeholders=False,
)

POSTGRES = Dialect(
    name="postgres",
    types={
        "text": "TEXT",
        "int": "INTEGER",
        "bigint": "BIGINT",
        "float": "DOUBLE PRECISION",
        "bool": "BOOLEAN",
        "timestamp": "TIMESTAMPTZ",
        "serial": "BIGSERIAL",
    },
    serial="BIGSERIAL PRIMARY KEY",
    now="now()",
    numbered_placeholders=True,
)


@dataclass(frozen=True)
class Column:
    """One column: its name, what it holds, and the three constraints these tables use.

    Not to be confused with `db/vectors`'s `Column`, which declares how a payload value is
    encoded rather than what a SQL column looks like.

    `NOT NULL` is the default, so a nullable column says `null=True` where it is declared rather
    than being nullable by omission. `references` names the parent as `table(column)` and always
    cascades on delete: every foreign key in Runa's schema is a child row that should not outlive
    its parent, and a second deletion rule would be a second shape for no gain.
    """

    name: str
    type: ColumnType
    null: bool = False
    primary_key: bool = False
    default_now: bool = False
    references: str | None = None

    def sql(self, dialect: Dialect) -> str:
        """This column's definition, in `dialect`."""
        if self.type == "serial":
            return f"{self.name} {dialect.serial}"
        parts = [self.name, dialect.types[self.type]]
        if self.primary_key:
            parts.append("PRIMARY KEY")  # implies NOT NULL in both backends
        elif not self.null:
            parts.append("NOT NULL")
        if self.references is not None:
            parts.append(f"REFERENCES {self.references} ON DELETE CASCADE")
        if self.default_now:
            parts.append(f"DEFAULT {dialect.now}")
        return " ".join(parts)


@dataclass(frozen=True)
class Index:
    """One index, named `idx_<table>_<suffix>` and ordered exactly as written.

    `columns` is raw SQL (`"updated_at DESC, session_id DESC"`) because an index's usefulness is
    in its ordering, and both backends spell that the same way.
    """

    suffix: str
    columns: str


@dataclass(frozen=True)
class Table:
    """One table, declared once for both backends.

    Column order is the row order: `column_names` is what an adapter's `INSERT` lists and what
    the concern's `*_values`/`to_*` marshalling packs and unpacks, so the declaration below is
    the single statement of what a row is.
    """

    name: str
    columns: tuple[Column, ...]
    indexes: tuple[Index, ...] = ()
    primary_key: tuple[str, ...] = field(default_factory=tuple)
    """A composite primary key, for a table whose identity is more than one column."""

    @property
    def column_names(self) -> tuple[str, ...]:
        """Every column, in declaration order, which is also the order rows are written in."""
        return tuple(column.name for column in self.columns)

    def placeholders(self, dialect: Dialect) -> str:
        """The bind placeholders for one whole row: `?, ?, ...` or `$1, $2, ...`."""
        if dialect.numbered_placeholders:
            return ", ".join(f"${index}" for index in range(1, len(self.columns) + 1))
        return ", ".join("?" * len(self.columns))

    def _ddl(self, dialect: Dialect) -> str:
        definitions = [column.sql(dialect) for column in self.columns]
        if self.primary_key:
            definitions.append(f"PRIMARY KEY ({', '.join(self.primary_key)})")
        body = ",\n    ".join(definitions)
        statements = [f"CREATE TABLE IF NOT EXISTS {self.name} (\n    {body}\n);"]
        statements += [
            f"CREATE INDEX IF NOT EXISTS idx_{self.name}_{index.suffix}"
            f" ON {self.name} ({index.columns});"
            for index in self.indexes
        ]
        return "\n".join(statements)


def ddl(dialect: Dialect, *tables: Table) -> str:
    """The `CREATE TABLE`/`CREATE INDEX` script for `tables`, in `dialect`.

    Idempotent (every statement is `IF NOT EXISTS`), which is what lets both `db/sqlite.py` and
    `db/pool.py` apply it on every connect instead of tracking whether a database is new.
    Declaration order is creation order, so a table referencing another is declared after it.
    """
    return "\n".join(table._ddl(dialect) for table in tables) + "\n"


__all__ = ["POSTGRES", "SQLITE", "Column", "Dialect", "Index", "Table", "ddl"]
