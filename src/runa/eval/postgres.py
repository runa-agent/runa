"""eval/postgres.py: eval history in Postgres, so every replica grades against one baseline.

`eval/storage.py` keeps eval runs in the same per-process `db/runa.db` as everything else, which
means a CI job and a developer's laptop each compare against a baseline the other cannot see, and
`runa ui` shows only whichever history it opened. Same two tables (`eval_runs`/`eval_cases`) in
Postgres instead, picked up automatically whenever `RUNA_POSTGRES_DSN` is set.

Optional: part of the `runa[postgres]` extra, like `db/postgres.py` and `tracing/postgres.py`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from runa.db.postgres import DEFAULT_POSTGRES_DSN, _connect, run_sync
from runa.eval.report import Report
from runa.eval.storage import EvalCaseRow, EvalRun

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


async def _save(report: Report, dsn: str) -> int:
    pool = await _connect(dsn, _DDL)
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
                f"""
                INSERT INTO {_CASES_TABLE}
                    (run_id, case_index, input, output, passed, results_json, trace_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                [
                    (
                        run_id,
                        case.index,
                        case.run.input,
                        case.run.final_output,
                        bool(case.passed),
                        json.dumps([asdict(result) for result in case.results], default=str),
                        case.run.trace.id or None,
                    )
                    for case in report.cases
                ],
            )
    return run_id


def save_report(report: Report, *, dsn: str = DEFAULT_POSTGRES_DSN) -> int:
    """Persist `report`, returning the new `eval_runs.id`."""
    return run_sync(_save(report, dsn))


def _row_to_case(row: Any) -> EvalCaseRow:
    return EvalCaseRow(
        index=row["case_index"],
        input=row["input"],
        output=row["output"],
        passed=bool(row["passed"]),
        results=json.loads(row["results_json"]),
        trace_id=row["trace_id"],
    )


def _row_to_run(row: Any, cases: list[Any]) -> EvalRun:
    return EvalRun(
        id=row["id"],
        agent_name=row["agent_name"],
        created_at=row["created_at"],
        score=row["score"],
        pass_rate=row["pass_rate"],
        cases=[_row_to_case(case) for case in cases],
    )


async def _list(dsn: str, limit: int) -> list[EvalRun]:
    pool = await _connect(dsn, _DDL)
    rows = await pool.fetch(f"SELECT * FROM {_RUNS_TABLE} ORDER BY id DESC LIMIT $1", limit)
    return [_row_to_run(row, []) for row in rows]


def list_eval_runs(*, limit: int = 50, dsn: str = DEFAULT_POSTGRES_DSN) -> list[EvalRun]:
    """Return the most recent `limit` eval runs, newest first, without their cases."""
    return run_sync(_list(dsn, limit))


async def _get(run_id: int, dsn: str) -> EvalRun | None:
    pool = await _connect(dsn, _DDL)
    row = await pool.fetchrow(f"SELECT * FROM {_RUNS_TABLE} WHERE id = $1", run_id)
    if row is None:
        return None
    cases = await pool.fetch(
        f"SELECT * FROM {_CASES_TABLE} WHERE run_id = $1 ORDER BY case_index", run_id
    )
    return _row_to_run(row, list(cases))


def get_eval_run(run_id: int, *, dsn: str = DEFAULT_POSTGRES_DSN) -> EvalRun | None:
    """Look up one eval run by id, with every case it graded, or `None` if it doesn't exist."""
    return run_sync(_get(run_id, dsn))


async def _baseline(agent_name: str, before: int | None, dsn: str) -> dict[str, bool] | None:
    pool = await _connect(dsn, _DDL)
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


def load_baseline(
    agent_name: str, *, before: int | None = None, dsn: str = DEFAULT_POSTGRES_DSN
) -> dict[str, bool] | None:
    """Map each input of `agent_name`'s latest eval run to whether it passed.

    Same contract as the SQLite backend, including `before` and the keyed-by-input behavior.
    """
    return run_sync(_baseline(agent_name, before, dsn))


__all__ = ["get_eval_run", "list_eval_runs", "load_baseline", "save_report"]
