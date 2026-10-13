"""eval/sqlite.py: `SQLiteEvalStore`, the local `EvalStore`.

`db/runa.db`'s `eval_runs`/`eval_cases` tables, in the same connect-and-create-if-missing file
every other local adapter writes to (`db/sqlite.py`). Every `agent.evaluate()` call writes one
`eval_runs` row (one "experiment") and one `eval_cases` row per case.

The tables are `eval/store.py`'s `RUNS`/`CASES`, rendered here in SQLite's dialect. This side
carried no index at all while the two schemas were written out separately, though `baseline`
filters and orders on `agent_name` and `id` on every call.

It also carried this repository's one hand-rolled migration, a `PRAGMA table_info` and an
`ALTER TABLE` adding `eval_cases.trace_id` to a `runa.db` from before that column existed. That
is `db/schema.py`'s `reconcile` now, applied by `db/sqlite.py` to every local adapter's tables:
the same diff, taken from the declaration rather than written out for one known column.
"""

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from runa.db import sqlite_path
from runa.db.schema import SQLITE
from runa.db.sqlite import connect as _connect_db
from runa.db.sqlite import read as _read_db
from runa.eval.report import Report
from runa.eval.store import CASES, RUNS, EvalRun, baseline_clause, case_values, to_run


class SQLiteEvalStore:
    """The local `EvalStore`: `db/runa.db`'s `eval_runs`/`eval_cases` tables."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Store which SQLite file this history lives in, defaulting to `db.sqlite_path()`.

        The tables are created on first use.
        """
        self.db_path = Path(db_path) if db_path is not None else sqlite_path()

    def _connect(self) -> sqlite3.Connection:
        return _connect_db(self.db_path, RUNS, CASES)

    def save(self, report: Report) -> int:
        """Persist `report` as one run plus one row per case, returning the new run's id."""
        created_at = datetime.now(UTC).isoformat()
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                f"INSERT INTO {RUNS.name} (agent_name, created_at, score, pass_rate) "
                "VALUES (?, ?, ?, ?)",
                (report.agent_name, created_at, report.score, report.pass_rate),
            )
            run_id = cursor.lastrowid
            assert run_id is not None
            conn.executemany(
                f"INSERT INTO {CASES.name} ({', '.join(CASES.column_names)}) "
                f"VALUES ({CASES.placeholders(SQLITE)})",
                [case_values(run_id, case) for case in report.cases],
            )
            conn.commit()
        return run_id

    def get(self, run_id: int) -> EvalRun | None:
        """Look up one run by id, with every case it graded, or `None` if it doesn't exist."""
        with _read_db(self.db_path, RUNS, CASES) as conn:
            if conn is None:
                return None
            row = conn.execute(f"SELECT * FROM {RUNS.name} WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            cases = conn.execute(
                f"SELECT * FROM {CASES.name} WHERE run_id = ? ORDER BY case_index", (run_id,)
            ).fetchall()
            return to_run(row, cases)

    def list(self, *, limit: int = 50) -> list[EvalRun]:
        """Return the most recent `limit` runs, newest first, without their cases."""
        with _read_db(self.db_path, RUNS, CASES) as conn:
            if conn is None:
                return []
            rows = conn.execute(
                f"SELECT * FROM {RUNS.name} ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [to_run(row, []) for row in rows]

    def baseline(self, agent_name: str, *, before: int | None = None) -> dict[str, bool] | None:
        """Map each input of `agent_name`'s latest run to whether it passed."""
        clause, params = baseline_clause(agent_name, before, SQLITE)
        with _read_db(self.db_path, RUNS, CASES) as conn:
            if conn is None:
                return None
            row = conn.execute(f"SELECT id FROM {RUNS.name} {clause}", params).fetchone()
            if row is None:
                return None
            cases = conn.execute(
                f"SELECT input, passed FROM {CASES.name} WHERE run_id = ?", (row["id"],)
            ).fetchall()
            return {case["input"]: bool(case["passed"]) for case in cases}


__all__ = ["SQLiteEvalStore"]
