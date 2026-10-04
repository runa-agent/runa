"""eval/sqlite.py: `SQLiteEvalStore`, the local `EvalStore`.

`db/runa.db`'s `eval_runs`/`eval_cases` tables, in the same connect-and-create-if-missing file
every other local adapter writes to (`db/sqlite.py`). Every `agent.evaluate()` call writes one
`eval_runs` row (one "experiment") and one `eval_cases` row per case.

Same tables, same columns and the same index as `eval/postgres.py`. This side carried no index at
all until the schemas were brought together, while `baseline` filters and orders on `agent_name`
and `id` on every call.
"""

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from runa.db import DEFAULT_DB_PATH
from runa.db.sqlite import connect as _connect_db
from runa.eval.report import Report
from runa.eval.store import CASE_COLUMNS, EvalRun, case_values, to_run

_RUNS_TABLE = "eval_runs"
_CASES_TABLE = "eval_cases"
_NO_LIMIT = 2**63 - 1  # SQLite's largest integer, above every `eval_runs.id`

_DDL = f"""
CREATE TABLE IF NOT EXISTS {_RUNS_TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    score REAL NOT NULL,
    pass_rate REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_{_RUNS_TABLE}_agent ON {_RUNS_TABLE} (agent_name, id DESC);
CREATE TABLE IF NOT EXISTS {_CASES_TABLE} (
    run_id INTEGER NOT NULL REFERENCES {_RUNS_TABLE}(id) ON DELETE CASCADE,
    case_index INTEGER NOT NULL,
    input TEXT NOT NULL,
    output TEXT,
    passed INTEGER NOT NULL,
    results_json TEXT NOT NULL,
    trace_id TEXT,
    PRIMARY KEY (run_id, case_index)
);
"""

_CASE_PLACEHOLDERS = ", ".join("?" * len(CASE_COLUMNS))


class SQLiteEvalStore:
    """The local `EvalStore`: `db/runa.db`'s `eval_runs`/`eval_cases` tables."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        """Store which SQLite file this history lives in; the tables are created on first use."""
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        """Open the file, adding `eval_cases.trace_id` to a `runa.db` from before it existed.

        The one migration either backend needs, and only this one can need it: a shared Postgres
        is created by whichever replica connects first and has had the column from the start,
        where a local file can be older than the release that linked a case to its trace.
        """
        conn = _connect_db(self.db_path, _DDL)
        conn.row_factory = sqlite3.Row
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({_CASES_TABLE})")}
        if "trace_id" not in columns:
            conn.execute(f"ALTER TABLE {_CASES_TABLE} ADD COLUMN trace_id TEXT")
            conn.commit()
        return conn

    def save(self, report: Report) -> int:
        """Persist `report` as one run plus one row per case, returning the new run's id."""
        created_at = datetime.now(UTC).isoformat()
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                f"INSERT INTO {_RUNS_TABLE} (agent_name, created_at, score, pass_rate) "
                "VALUES (?, ?, ?, ?)",
                (report.agent_name, created_at, report.score, report.pass_rate),
            )
            run_id = cursor.lastrowid
            assert run_id is not None
            conn.executemany(
                f"INSERT INTO {_CASES_TABLE} ({', '.join(CASE_COLUMNS)}) "
                f"VALUES ({_CASE_PLACEHOLDERS})",
                [case_values(run_id, case) for case in report.cases],
            )
            conn.commit()
        return run_id

    def get(self, run_id: int) -> EvalRun | None:
        """Look up one run by id, with every case it graded, or `None` if it doesn't exist."""
        with closing(self._connect()) as conn:
            row = conn.execute(f"SELECT * FROM {_RUNS_TABLE} WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            cases = conn.execute(
                f"SELECT * FROM {_CASES_TABLE} WHERE run_id = ? ORDER BY case_index", (run_id,)
            ).fetchall()
            return to_run(row, cases)

    def list(self, *, limit: int = 50) -> list[EvalRun]:
        """Return the most recent `limit` runs, newest first, without their cases."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT * FROM {_RUNS_TABLE} ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [to_run(row, []) for row in rows]

    def baseline(self, agent_name: str, *, before: int | None = None) -> dict[str, bool] | None:
        """Map each input of `agent_name`'s latest run to whether it passed."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"SELECT id FROM {_RUNS_TABLE} WHERE agent_name = ? AND id < ? "
                "ORDER BY id DESC LIMIT 1",
                (agent_name, before if before is not None else _NO_LIMIT),
            ).fetchone()
            if row is None:
                return None
            cases = conn.execute(
                f"SELECT input, passed FROM {_CASES_TABLE} WHERE run_id = ?", (row["id"],)
            ).fetchall()
            return {case["input"]: bool(case["passed"]) for case in cases}


__all__ = ["SQLiteEvalStore"]
