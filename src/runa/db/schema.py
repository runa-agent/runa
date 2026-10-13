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
the eight adapters it replaced. The only query fragments here are `Dialect.placeholder` and
`Table.placeholders`, which are dialect, not query: `?` against `$1`.

A concern whose adapters must phrase one predicate identically keeps that fragment beside its own
tables, as a function of the values and the `Dialect` returning the clause and what it binds.
There are three: `session/store.py`'s `agent_filter`, `tracing/store.py`'s `listing_clause` and
`eval/store.py`'s `baseline_clause`. Each returns the whole fragment rather than a piece of one --
`agent_filter` because an escaped `LIKE` pattern is meaningless without the `ESCAPE` clause naming
the escape character, the other two because a numbered placeholder depends on how many values came
before it, so a caller handed half a clause has to re-derive the count. Those are predicates a
concern owns, not a query builder growing here: each states one sentence of policy ("the latest run
for this agent, older than this one"), and the two adapters that must agree on it read it from one
place instead of phrasing it twice.

Declaring a table is only half of owning it: `db/runa.db` outlives installs by design (that is the
whole point of `--continue`, of trace history, of eval baselines), so a column added to a
declaration below has to reach databases that already exist. `CREATE TABLE IF NOT EXISTS` does not
do that -- on an existing table it is a no-op, and the next query fails with `no such column`.
`reconcile` is the other half: it diffs what the declaration says against what the database has
and emits the `ALTER TABLE ... ADD COLUMN` the difference needs, so the two halves answer from the
same declaration. `eval/sqlite.py` used to hand-roll that diff for its one column inside its own
`_connect`, which worked exactly until a second column landed.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from runa.exceptions import OperatorError

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
    columns_query: str
    """Every column this database actually has, as `(table, column)` rows, for `reconcile`.

    One query for the whole database rather than one per table: both backends can answer it that
    way, and it keeps the only connection-aware step in `reconcile`'s plumbing to a single fetch.
    """

    def placeholder(self, index: int) -> str:
        """The bind placeholder for a statement's `index`th value, counting from 1.

        `?` ignores the index; `$1, $2, ...` is the whole reason a fragment has to be told how
        many values were bound before it. Every predicate written for both dialects needs this
        one substitution and nothing else of a dialect, so it lives here rather than as a
        `numbered_placeholders` branch in each of them.
        """
        return f"${index}" if self.numbered_placeholders else "?"


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
    columns_query=(
        "SELECT m.name, p.name FROM sqlite_master AS m "
        "JOIN pragma_table_info(m.name) AS p WHERE m.type = 'table'"
    ),
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
    columns_query=(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema()"
    ),
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

    @property
    def addable(self) -> bool:
        """Whether `reconcile` can add this column to a table that already exists.

        A nullable column with no default, which is every column so far added to a shipped table.
        The other three kinds are what `ALTER TABLE ... ADD COLUMN` refuses outright, in the same
        words in both backends: a `NOT NULL` column has no value for the rows already there, a
        `PRIMARY KEY` (`serial` included) cannot be added at all, and `DEFAULT CURRENT_TIMESTAMP`
        is a non-constant default. Each of those needs a human and a backfill, so `reconcile`
        raises `SchemaOutOfDate` rather than emitting SQL the database will reject anyway.
        """
        return self.null and not self.primary_key and not self.default_now and self.type != "serial"

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
        return ", ".join(dialect.placeholder(index) for index in range(1, len(self.columns) + 1))

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


type Declaration = Table | str
"""One piece of a concern's schema: a declared `Table`, or raw SQL for the few that can't be one.

`db/vectors`'s two SQL stores are the raw case, and the only one: a `vec0` virtual table and a
`pgvector` column have no logical `ColumnType` here, so those adapters build their own `CREATE`
text from a `VectorSpec`. Raw SQL is created like a declared table and reconciled like nothing --
that's the price of staying outside the declaration, and the reason to stay inside it.
"""


class SchemaOutOfDate(OperatorError):
    """Raised when a database is missing a declared column that can't be added additively.

    The one case `reconcile` hands back to a human: a `NOT NULL` column, a primary key or a
    non-constant default has no value for rows that already exist, so there is a backfill to
    decide and nobody but the operator can decide it. Loud rather than skipped, because skipped
    means `no such column` on the next query instead.
    """

    def __init__(self, table: str, column: Column) -> None:
        """Name the table and column, and say what the two ways out are."""
        self.table = table
        self.column = column
        super().__init__(
            f"this database's {table} table has no {column.name} column, and only a nullable "
            f"column with no default can be added to a table that already exists. Add "
            f"{table}.{column.name} by hand (with whatever value its existing rows should get), "
            f"or delete the database to recreate it from the current schema."
        )


def ddl(dialect: Dialect, *tables: Declaration) -> str:
    """The `CREATE TABLE`/`CREATE INDEX` script for `tables`, in `dialect`.

    Idempotent (every statement is `IF NOT EXISTS`), which is what lets both `db/sqlite.py` and
    `db/pool.py` apply it on every connect instead of tracking whether a database is new.
    Declaration order is creation order, so a table referencing another is declared after it.

    Creation only: what this leaves for `reconcile` is a table that already exists and whose
    declaration has since grown a column.
    """
    return (
        "\n".join(table if isinstance(table, str) else table._ddl(dialect) for table in tables)
        + "\n"
    )


def reconcile(
    dialect: Dialect, columns: Iterable[tuple[str, str]], *tables: Declaration
) -> list[str]:
    """The `ALTER TABLE` statements bringing an existing database up to `tables`'s declaration.

    `columns` is `dialect.columns_query`'s rows: one `(table, column)` pair per column the
    database actually has. Pure, like `ddl` -- the caller fetches the rows and executes what comes
    back, which is the whole of what `db/sqlite.py` and `db/pool.py` add to applying a script.

    Only columns need this. A new table and a new index are already covered by `ddl`'s
    `IF NOT EXISTS`, and a table the declaration no longer mentions, or a column it doesn't,
    is left alone: every adapter here selects columns by name, so an older release reading a
    newer database is not a failure. What isn't additive raises `SchemaOutOfDate`; see
    `Column.addable`.
    """
    existing: dict[str, set[str]] = {}
    for table_name, column_name in columns:
        existing.setdefault(table_name, set()).add(column_name)
    statements = []
    for table in tables:
        if isinstance(table, str) or table.name not in existing:
            continue  # raw SQL this module can't read, or a table `ddl` just created
        for column in table.columns:
            if column.name in existing[table.name]:
                continue
            if not column.addable:
                raise SchemaOutOfDate(table.name, column)
            statements.append(f"ALTER TABLE {table.name} ADD COLUMN {column.sql(dialect)}")
    return statements


__all__ = [
    "POSTGRES",
    "SQLITE",
    "Column",
    "Declaration",
    "Dialect",
    "Index",
    "SchemaOutOfDate",
    "Table",
    "ddl",
    "reconcile",
]
