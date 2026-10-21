"""eval/store.py: `EvalStore`, the history every `agent.evaluate()` run is graded against.

The `eval_runs`/`eval_cases` tables in whichever database the deployment has, queried through
`db/handle.py`'s `Database`. Every `agent.evaluate()` call writes one `eval_runs` row (one
"experiment") and one `eval_cases` row per case. Which database it is is `runa.db.evals(...)`'s
decision, asked once, so a CI job and a laptop compare against the baseline their deployment
actually shares rather than whichever file each happened to open.

`RUNS`/`CASES` are the two tables, rendered per dialect by `db/schema.py`. `EvalRun`/`EvalCaseRow`
are what a row becomes, and `web/evaluations.py` renders them without knowing which database it
got; the marshalling below the store maps them in both directions, so `passed` cannot come back as
SQLite's `0`/`1` from one database and a `bool` from the other.

`baseline_clause` is here for the same reason: which run is "the baseline" is this module's rule.
It decides what `runa eval` calls a regression, so a laptop and CI have to agree on it.
"""

import builtins
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from runa.db.handle import Database, Row
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


class EvalStore:
    """One deployment's eval history: every `agent.evaluate()` run and the cases it graded.

    The `eval_runs`/`eval_cases` tables of a `Database`, created on the first `save`.
    """

    def __init__(self, database: Database) -> None:
        """Store the database this history lives in."""
        self.database = database

    async def save(self, report: Report) -> int:
        """Persist `report` as one run plus one row per case, returning the new run's id."""
        created_at = datetime.now(UTC).isoformat()
        async with self.database.transaction() as tx:
            run_id: int = await tx.fetchval(
                f"INSERT INTO {RUNS.name} (agent_name, created_at, score, pass_rate) "
                "VALUES ($1, $2, $3, $4) RETURNING id",
                report.agent_name,
                created_at,
                report.score,
                report.pass_rate,
            )
            if report.cases:
                await tx.executemany(
                    f"INSERT INTO {CASES.name} ({', '.join(CASES.column_names)}) "
                    f"VALUES ({CASES.placeholders()})",
                    [_case_values(run_id, case) for case in report.cases],
                )
        return run_id

    async def get(self, run_id: int) -> EvalRun | None:
        """Look up one run by id, with every case it graded, or `None` if it doesn't exist."""
        row = await self.database.fetchrow(f"SELECT * FROM {RUNS.name} WHERE id = $1", run_id)
        if row is None:
            return None
        cases = await self.database.fetch(
            f"SELECT * FROM {CASES.name} WHERE run_id = $1 ORDER BY case_index", run_id
        )
        return _to_run(row, cases)

    async def list(self, *, limit: int = 50) -> builtins.list[EvalRun]:
        """Return the most recent `limit` runs, newest first, without their cases."""
        rows = await self.database.fetch(
            f"SELECT * FROM {RUNS.name} ORDER BY id DESC LIMIT $1", limit
        )
        return [_to_run(row, []) for row in rows]

    async def baseline(
        self, agent_name: str, *, before: int | None = None
    ) -> dict[str, bool] | None:
        """Map each input of `agent_name`'s latest run to whether it passed.

        `before` looks at the latest run older than that run id instead, the baseline a past run
        was compared against -- strictly older, so `before` the newest run skips it. `None` when
        there's no such run. Keyed by input rather than index, so reordering, adding, or removing
        cases between runs still lines the rest up.
        """
        clause, params = baseline_clause(agent_name, before)
        row = await self.database.fetchrow(f"SELECT id FROM {RUNS.name} {clause}", *params)
        if row is None:
            return None
        cases = await self.database.fetch(
            f"SELECT input, passed FROM {CASES.name} WHERE run_id = $1", row["id"]
        )
        return {case["input"]: bool(case["passed"]) for case in cases}


def baseline_clause(agent_name: str, before: int | None) -> tuple[str, tuple[Any, ...]]:
    """Everything after `SELECT id FROM eval_runs` for `EvalStore.baseline`, and its values.

    One sentence of policy: the newest run of `agent_name`, strictly older than `before` when
    there is one. `before=None` drops the comparison rather than widening it to a sentinel, so
    the rule reads the same either way.

    The whole tail rather than the `WHERE` alone, for the reason `tracing/store.py`'s
    `listing_clause` is: the `ORDER BY ... LIMIT 1` is what makes it "the latest", and a numbered
    placeholder has to count the values bound before it.
    """
    clause = "WHERE agent_name = $1"
    params: tuple[Any, ...] = (agent_name,)
    if before is not None:
        clause += " AND id < $2"
        params += (before,)
    return f"{clause} ORDER BY id DESC LIMIT 1", params


def _case_values(run_id: int, case: CaseReport) -> tuple[Any, ...]:
    """One graded `case`'s column values, in `CASES.column_names` order."""
    return (
        run_id,
        case.index,
        case.case.input,
        case.run.output,
        bool(case.passed),
        json.dumps([asdict(result) for result in case.results], default=str),
        case.run.trace.id,
    )


def _to_case(row: Row) -> EvalCaseRow:
    """One `eval_cases` row as an `EvalCaseRow`."""
    return EvalCaseRow(
        index=row["case_index"],
        input=row["input"],
        output=row["output"],
        passed=bool(row["passed"]),
        results=json.loads(row["results_json"]),
        trace_id=row["trace_id"],
    )


def _to_run(row: Row, case_rows: list[Row]) -> EvalRun:
    """One `eval_runs` row plus its `eval_cases` rows as an `EvalRun`."""
    return EvalRun(
        id=row["id"],
        agent_name=row["agent_name"],
        created_at=row["created_at"],
        score=row["score"],
        pass_rate=row["pass_rate"],
        cases=[_to_case(case_row) for case_row in case_rows],
    )


__all__ = [
    "CASES",
    "RUNS",
    "EvalCaseRow",
    "EvalRun",
    "EvalStore",
    "baseline_clause",
]
