"""eval/database.py: `DatabaseEvalStore`, the `EvalStore` over a `Database`.

The `eval_runs`/`eval_cases` tables in whichever database the deployment has. Every
`agent.evaluate()` call writes one `eval_runs` row (one "experiment") and one `eval_cases` row per
case. Local, a CI job and a developer's laptop each compare against a baseline the other cannot
see; over the shared Postgres `runa.db.evals()` returns whenever `RUNA_DATABASE_URL` is a
`postgresql://` one, every replica grades against the same history.

The two tables, the row marshalling and the clause selecting a baseline run are `eval/store.py`'s.
"""

import builtins
from datetime import UTC, datetime

from runa.db.handle import Database
from runa.eval.report import Report
from runa.eval.store import CASES, RUNS, EvalRun, baseline_clause, case_values, to_run


class DatabaseEvalStore:
    """The `EvalStore` over a `Database`: its `eval_runs`/`eval_cases` tables."""

    def __init__(self, database: Database) -> None:
        """Store the database this history lives in; its tables are created on the first `save`."""
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
                    [case_values(run_id, case) for case in report.cases],
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
        return to_run(row, cases)

    async def list(self, *, limit: int = 50) -> builtins.list[EvalRun]:
        """Return the most recent `limit` runs, newest first, without their cases."""
        rows = await self.database.fetch(
            f"SELECT * FROM {RUNS.name} ORDER BY id DESC LIMIT $1", limit
        )
        return [to_run(row, []) for row in rows]

    async def baseline(
        self, agent_name: str, *, before: int | None = None
    ) -> dict[str, bool] | None:
        """Map each input of `agent_name`'s latest run to whether it passed."""
        clause, params = baseline_clause(agent_name, before)
        row = await self.database.fetchrow(f"SELECT id FROM {RUNS.name} {clause}", *params)
        if row is None:
            return None
        cases = await self.database.fetch(
            f"SELECT input, passed FROM {CASES.name} WHERE run_id = $1", row["id"]
        )
        return {case["input"]: bool(case["passed"]) for case in cases}


__all__ = ["DatabaseEvalStore"]
