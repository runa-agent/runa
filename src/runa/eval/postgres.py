"""eval/postgres.py: `PostgresEvalStore`, eval history every replica grades against.

The local `SQLiteEvalStore` keeps eval runs in the per-process `db/runa.db`, which means a CI job
and a developer's laptop each compare against a baseline the other cannot see, and `runa ui` shows
only whichever history it opened. Same two tables (`eval_runs`/`eval_cases`) in Postgres instead,
picked up automatically whenever `RUNA_DATABASE_URL` is a `postgresql://` one.

Optional: part of the `runa[postgres]` extra, like `db/pool.py` and `tracing/postgres.py`. The
two tables, the row marshalling and the clause selecting a baseline run are `eval/store.py`'s,
shared with the SQLite adapter; `baseline` used to branch into two queries here where SQLite
covered both cases with a sentinel, which is one rule written twice.
"""

from datetime import UTC, datetime

from runa.db.pool import Shared, sync
from runa.db.schema import POSTGRES
from runa.eval.report import Report
from runa.eval.store import CASES, RUNS, EvalRun, baseline_clause, case_values, to_run


class PostgresEvalStore(Shared):
    """The shared `EvalStore`: `eval_runs`/`eval_cases` in this deployment's Postgres database."""

    def __init__(self, url: str) -> None:
        """Store which Postgres database this history lives in; connected lazily."""
        super().__init__(url, RUNS, CASES)

    @sync
    async def save(self, report: Report) -> int:
        """Persist `report` as one run plus one row per case, returning the new run's id."""
        pool = await self._pool()
        created_at = datetime.now(UTC).isoformat()
        async with pool.acquire() as conn, conn.transaction():
            run_id: int = await conn.fetchval(
                f"INSERT INTO {RUNS.name} (agent_name, created_at, score, pass_rate) "
                "VALUES ($1, $2, $3, $4) RETURNING id",
                report.agent_name,
                created_at,
                report.score,
                report.pass_rate,
            )
            if report.cases:
                await conn.executemany(
                    f"INSERT INTO {CASES.name} ({', '.join(CASES.column_names)}) "
                    f"VALUES ({CASES.placeholders(POSTGRES)})",
                    [case_values(run_id, case) for case in report.cases],
                )
        return run_id

    @sync
    async def get(self, run_id: int) -> EvalRun | None:
        """Look up one run by id, with every case it graded, or `None` if it doesn't exist."""
        pool = await self._pool()
        row = await pool.fetchrow(f"SELECT * FROM {RUNS.name} WHERE id = $1", run_id)
        if row is None:
            return None
        cases = await pool.fetch(
            f"SELECT * FROM {CASES.name} WHERE run_id = $1 ORDER BY case_index", run_id
        )
        return to_run(row, list(cases))

    @sync
    async def list(self, *, limit: int = 50) -> list[EvalRun]:
        """Return the most recent `limit` runs, newest first, without their cases."""
        pool = await self._pool()
        rows = await pool.fetch(f"SELECT * FROM {RUNS.name} ORDER BY id DESC LIMIT $1", limit)
        return [to_run(row, []) for row in rows]

    @sync
    async def baseline(
        self, agent_name: str, *, before: int | None = None
    ) -> dict[str, bool] | None:
        """Map each input of `agent_name`'s latest run to whether it passed."""
        pool = await self._pool()
        clause, params = baseline_clause(agent_name, before, POSTGRES)
        row = await pool.fetchrow(f"SELECT id FROM {RUNS.name} {clause}", *params)
        if row is None:
            return None
        cases = await pool.fetch(
            f"SELECT input, passed FROM {CASES.name} WHERE run_id = $1", row["id"]
        )
        return {case["input"]: bool(case["passed"]) for case in cases}


__all__ = ["PostgresEvalStore"]
