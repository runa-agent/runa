"""eval/postgres.py: `PostgresEvalStore`, eval history every replica grades against.

The local `SQLiteEvalStore` keeps eval runs in the per-process `db/runa.db`, which means a CI job
and a developer's laptop each compare against a baseline the other cannot see, and `runa ui` shows
only whichever history it opened. Same two tables (`eval_runs`/`eval_cases`) in Postgres instead,
picked up automatically whenever `RUNA_DATABASE_URL` is a `postgresql://` one.

Optional: part of the `runa[postgres]` extra, like `db/pool.py` and `tracing/postgres.py`. The
row marshalling is `eval/store.py`'s, shared with the SQLite adapter.
"""

from datetime import UTC, datetime

from runa.db.pool import connect as _connect
from runa.db.pool import run_sync
from runa.eval.report import Report
from runa.eval.store import CASE_COLUMNS, EvalRun, case_values, to_run

_RUNS_TABLE = "eval_runs"
_CASES_TABLE = "eval_cases"

_DDL = f"""
CREATE TABLE IF NOT EXISTS {_RUNS_TABLE} (
    id BIGSERIAL PRIMARY KEY,
    agent_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    pass_rate DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_{_RUNS_TABLE}_agent ON {_RUNS_TABLE} (agent_name, id DESC);
CREATE TABLE IF NOT EXISTS {_CASES_TABLE} (
    run_id BIGINT NOT NULL REFERENCES {_RUNS_TABLE}(id) ON DELETE CASCADE,
    case_index INTEGER NOT NULL,
    input TEXT NOT NULL,
    output TEXT,
    passed BOOLEAN NOT NULL,
    results_json TEXT NOT NULL,
    trace_id TEXT,
    PRIMARY KEY (run_id, case_index)
);
"""

_CASE_PLACEHOLDERS = ", ".join(f"${index}" for index in range(1, len(CASE_COLUMNS) + 1))


class PostgresEvalStore:
    """The shared `EvalStore`: `eval_runs`/`eval_cases` in this deployment's Postgres database."""

    def __init__(self, url: str) -> None:
        """Store which Postgres database this history lives in; connected lazily."""
        self.url = url

    def save(self, report: Report) -> int:
        """Persist `report` as one run plus one row per case, returning the new run's id."""
        return run_sync(self._save(report))

    def get(self, run_id: int) -> EvalRun | None:
        """Look up one run by id, with every case it graded, or `None` if it doesn't exist."""
        return run_sync(self._get(run_id))

    def list(self, *, limit: int = 50) -> list[EvalRun]:
        """Return the most recent `limit` runs, newest first, without their cases."""
        return run_sync(self._list(limit))

    def baseline(self, agent_name: str, *, before: int | None = None) -> dict[str, bool] | None:
        """Map each input of `agent_name`'s latest run to whether it passed."""
        return run_sync(self._baseline(agent_name, before))

    async def _save(self, report: Report) -> int:
        pool = await _connect(self.url, _DDL)
        created_at = datetime.now(UTC).isoformat()
        async with pool.acquire() as conn, conn.transaction():
            run_id: int = await conn.fetchval(
                f"INSERT INTO {_RUNS_TABLE} (agent_name, created_at, score, pass_rate) "
                "VALUES ($1, $2, $3, $4) RETURNING id",
                report.agent_name,
                created_at,
                report.score,
                report.pass_rate,
            )
            if report.cases:
                await conn.executemany(
                    f"INSERT INTO {_CASES_TABLE} ({', '.join(CASE_COLUMNS)}) "
                    f"VALUES ({_CASE_PLACEHOLDERS})",
                    [case_values(run_id, case) for case in report.cases],
                )
        return run_id

    async def _get(self, run_id: int) -> EvalRun | None:
        pool = await _connect(self.url, _DDL)
        row = await pool.fetchrow(f"SELECT * FROM {_RUNS_TABLE} WHERE id = $1", run_id)
        if row is None:
            return None
        cases = await pool.fetch(
            f"SELECT * FROM {_CASES_TABLE} WHERE run_id = $1 ORDER BY case_index", run_id
        )
        return to_run(row, list(cases))

    async def _list(self, limit: int) -> list[EvalRun]:
        pool = await _connect(self.url, _DDL)
        rows = await pool.fetch(f"SELECT * FROM {_RUNS_TABLE} ORDER BY id DESC LIMIT $1", limit)
        return [to_run(row, []) for row in rows]

    async def _baseline(self, agent_name: str, before: int | None) -> dict[str, bool] | None:
        pool = await _connect(self.url, _DDL)
        if before is None:
            row = await pool.fetchrow(
                f"SELECT id FROM {_RUNS_TABLE} WHERE agent_name = $1 ORDER BY id DESC LIMIT 1",
                agent_name,
            )
        else:
            row = await pool.fetchrow(
                f"SELECT id FROM {_RUNS_TABLE} WHERE agent_name = $1 AND id < $2 "
                "ORDER BY id DESC LIMIT 1",
                agent_name,
                before,
            )
        if row is None:
            return None
        cases = await pool.fetch(
            f"SELECT input, passed FROM {_CASES_TABLE} WHERE run_id = $1", row["id"]
        )
        return {case["input"]: bool(case["passed"]) for case in cases}


__all__ = ["PostgresEvalStore"]
