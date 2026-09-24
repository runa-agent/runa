"""eval/storage.py: the `eval_runs`/`eval_cases` tables inside `runa.db`.

Every `agent.evaluate()` call writes one row to `eval_runs` (one "experiment") and one row
per case to `eval_cases`, in the same `runa.db` file `SQLiteSession` already uses, so a
local app accumulates one database with no setup.
"""

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from runa.db.sqlite import DEFAULT_DB_PATH
from runa.db.sqlite import connect as _connect_db
from runa.eval.report import Report

_RUNS_TABLE = "eval_runs"
_CASES_TABLE = "eval_cases"

_DDL = f"""
CREATE TABLE IF NOT EXISTS {_RUNS_TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    score REAL NOT NULL,
    pass_rate REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS {_CASES_TABLE} (
    run_id INTEGER NOT NULL REFERENCES {_RUNS_TABLE}(id),
    case_index INTEGER NOT NULL,
    input TEXT NOT NULL,
    output TEXT,
    passed INTEGER NOT NULL,
    results_json TEXT NOT NULL,
    PRIMARY KEY (run_id, case_index)
);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    return _connect_db(db_path, _DDL)


def save_report(report: Report, *, db_path: Path = DEFAULT_DB_PATH) -> int:
    """Persist `report` to `db_path`, returning the new `eval_runs.id`."""
    created_at = datetime.now(UTC).isoformat()
    with closing(_connect(db_path)) as conn:
        cursor = conn.execute(
            f"INSERT INTO {_RUNS_TABLE} (agent_name, created_at, score, pass_rate) "
            "VALUES (?, ?, ?, ?)",
            (report.agent_name, created_at, report.score, report.pass_rate),
        )
        run_id = cursor.lastrowid
        assert run_id is not None
        conn.executemany(
            f"""
            INSERT INTO {_CASES_TABLE}
                (run_id, case_index, input, output, passed, results_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    case.index,
                    case.run.input,
                    case.run.final_output,
                    int(case.passed),
                    json.dumps([asdict(result) for result in case.results], default=str),
                )
                for case in report.cases
            ],
        )
        conn.commit()
    return run_id


@dataclass
class EvalCaseRow:
    """One `eval_cases` row, read back: a case's input/output/verdict/per-metric results."""

    index: int
    input: str
    output: str | None
    passed: bool
    results: list[dict[str, Any]]


@dataclass
class EvalRun:
    """One `eval_runs` row, optionally with its `EvalCaseRow`s (empty from `list_eval_runs`)."""

    id: int
    agent_name: str
    created_at: str
    score: float
    pass_rate: float
    cases: list[EvalCaseRow] = field(default_factory=list)


def _row_to_case(row: sqlite3.Row) -> EvalCaseRow:
    return EvalCaseRow(
        index=row["case_index"],
        input=row["input"],
        output=row["output"],
        passed=bool(row["passed"]),
        results=json.loads(row["results_json"]),
    )


def list_eval_runs(*, limit: int = 50, db_path: Path = DEFAULT_DB_PATH) -> list[EvalRun]:
    """Return the most recent `limit` eval runs, newest first, without their cases."""
    with closing(_connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM {_RUNS_TABLE} ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            EvalRun(
                id=row["id"],
                agent_name=row["agent_name"],
                created_at=row["created_at"],
                score=row["score"],
                pass_rate=row["pass_rate"],
            )
            for row in rows
        ]


def get_eval_run(run_id: int, *, db_path: Path = DEFAULT_DB_PATH) -> EvalRun | None:
    """Look up one eval run by id, with every case it graded, or `None` if it doesn't exist."""
    with closing(_connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(f"SELECT * FROM {_RUNS_TABLE} WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        case_rows = conn.execute(
            f"SELECT * FROM {_CASES_TABLE} WHERE run_id = ? ORDER BY case_index", (run_id,)
        ).fetchall()
        return EvalRun(
            id=row["id"],
            agent_name=row["agent_name"],
            created_at=row["created_at"],
            score=row["score"],
            pass_rate=row["pass_rate"],
            cases=[_row_to_case(case_row) for case_row in case_rows],
        )


def load_baseline(agent_name: str, *, db_path: Path = DEFAULT_DB_PATH) -> dict[str, bool] | None:
    """Map each input of `agent_name`'s latest eval run to whether it passed.

    `None` when the agent was never evaluated. Keyed by input rather than index, so reordering,
    adding, or removing cases between runs still lines the rest up.
    """
    with closing(_connect(db_path)) as conn:
        row = conn.execute(
            f"SELECT id FROM {_RUNS_TABLE} WHERE agent_name = ? ORDER BY id DESC LIMIT 1",
            (agent_name,),
        ).fetchone()
        if row is None:
            return None
        rows = conn.execute(
            f"SELECT input, passed FROM {_CASES_TABLE} WHERE run_id = ?", (row[0],)
        ).fetchall()
        return {input: bool(passed) for input, passed in rows}


__all__ = [
    "EvalCaseRow",
    "EvalRun",
    "get_eval_run",
    "list_eval_runs",
    "load_baseline",
    "save_report",
]
