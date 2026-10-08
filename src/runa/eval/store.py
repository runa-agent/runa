"""eval/store.py: `EvalStore`, the history every `agent.evaluate()` run is graded against.

One interface, three adapters: `eval/sqlite.py`, `eval/postgres.py`, `eval/ephemeral.py`. Which
one a caller gets is `runa.db.evals(...)`'s decision, asked once, so a CI job and a laptop
compare against the baseline their deployment actually shares rather than whichever file each
happened to open.

`RUNS`/`CASES` are the two tables both SQL adapters create, declared once here and rendered per
dialect by `db/schema.py`. `EvalRun`/`EvalCaseRow` are what a row becomes, declared here rather
than in an adapter: both
backends read the same two objects back, and `web/evaluations.py` renders them without knowing
which one it got. `case_values`/`to_case`/`to_run` are that mapping in both directions, written
once, so `passed` cannot come back as SQLite's `0`/`1` from one store and a `bool` from the other.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from runa.db.schema import Column, Index, Table
from runa.eval.report import CaseReport, Report

RUNS = Table(
    "eval_runs",
    columns=(
        Column("id", "serial"),
        Column("agent_name", "text"),
        Column("created_at", "text"),
        Column("score", "float"),
        Column("pass_rate", "float"),
    ),
    indexes=(Index("agent", "agent_name, id DESC"),),
)

CASES = Table(
    "eval_cases",
    columns=(
        Column("run_id", "bigint", references=f"{RUNS.name}(id)"),
        Column("case_index", "int"),
        Column("input", "text"),
        Column("output", "text", null=True),
        Column("passed", "bool"),
        Column("results_json", "text"),
        Column("trace_id", "text", null=True),
    ),
    primary_key=("run_id", "case_index"),
)


@dataclass
class EvalCaseRow:
    """One `eval_cases` row, read back: a case's input/output/verdict/per-metric results."""

    index: int
    input: str
    output: str | None
    passed: bool
    results: list[dict[str, Any]]
    trace_id: str | None = None


@dataclass
class EvalRun:
    """One `eval_runs` row, optionally with its `EvalCaseRow`s (empty from `EvalStore.list`)."""

    id: int
    agent_name: str
    created_at: str
    score: float
    pass_rate: float
    cases: list[EvalCaseRow] = field(default_factory=list)


class EvalStore(Protocol):
    """One deployment's eval history: every `agent.evaluate()` run and the cases it graded.

    No inheritance required, every adapter satisfies this by matching shape.
    """

    def save(self, report: Report) -> int:
        """Persist `report` as one run plus one row per case, returning the new run's id."""
        ...

    def get(self, run_id: int) -> EvalRun | None:
        """Look up one run by id, with every case it graded, or `None` if it doesn't exist."""
        ...

    def list(self, *, limit: int = 50) -> list[EvalRun]:
        """Return the most recent `limit` runs, newest first, without their cases."""
        ...

    def baseline(self, agent_name: str, *, before: int | None = None) -> dict[str, bool] | None:
        """Map each input of `agent_name`'s latest run to whether it passed.

        `before` looks at the latest run older than that run id instead, the baseline a past run
        was compared against. `None` when there's no such run. Keyed by input rather than index,
        so reordering, adding, or removing cases between runs still lines the rest up.
        """
        ...


def case_values(run_id: int, case: CaseReport) -> tuple[Any, ...]:
    """One graded `case`'s column values, in `CASES.column_names` order."""
    return (
        run_id,
        case.index,
        case.case.input,
        case.run.output,
        bool(case.passed),
        json.dumps([asdict(result) for result in case.results], default=str),
        case.run.trace.id if case.run.trace else None,
    )


def to_case(row: Any) -> EvalCaseRow:
    """One `eval_cases` row as an `EvalCaseRow`. `row` is anything subscriptable by column name."""
    return EvalCaseRow(
        index=row["case_index"],
        input=row["input"],
        output=row["output"],
        passed=bool(row["passed"]),
        results=json.loads(row["results_json"]),
        trace_id=row["trace_id"],
    )


def to_run(row: Any, case_rows: list[Any]) -> EvalRun:
    """One `eval_runs` row plus its `eval_cases` rows as an `EvalRun`.

    `sqlite3.Row` and `asyncpg.Record` are both subscriptable by column name, which is the only
    thing this needs of either, so the mapping is written once for both.
    """
    return EvalRun(
        id=row["id"],
        agent_name=row["agent_name"],
        created_at=row["created_at"],
        score=row["score"],
        pass_rate=row["pass_rate"],
        cases=[to_case(case_row) for case_row in case_rows],
    )


__all__ = [
    "CASES",
    "RUNS",
    "EvalCaseRow",
    "EvalRun",
    "EvalStore",
    "case_values",
    "to_case",
    "to_run",
]
