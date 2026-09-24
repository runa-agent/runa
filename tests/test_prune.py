"""Tests for `runa.db.prune`, the retention pass over `db/runa.db`.

Everything Runa persists is append-only, so without this a long-lived deployment's database grows
until the disk runs out. These cover the cutoff, the parent/child deletes, and the safety rails.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from runa.db.prune import prune
from runa.tracing.spans import Span
from runa.tracing.storage import save_trace
from runa.tracing.traces import Trace


def _days_ago(days: float) -> float:
    return (datetime.now(UTC) - timedelta(days=days)).timestamp()


def _iso_days_ago(days: float) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


def _write_trace(db: Path, trace_id: str, age_days: float, spans: int = 2) -> None:
    start = _days_ago(age_days)
    trace = Trace(id=trace_id, name="Agent", start_time=start, end_time=start + 1)
    for i in range(spans):
        trace.spans.append(
            Span(
                id=f"{trace_id}-s{i}",
                trace_id=trace_id,
                parent_id=None,
                name="llm",
                type="llm",
                start_time=start,
                end_time=start + 1,
                status="ok",
            )
        )
    save_trace(trace, db_path=db)


def _write_session(db: Path, session_id: str, age_days: float, messages: int = 3) -> None:
    with closing(sqlite3.connect(db)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS agent_sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS agent_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES agent_sessions(session_id),
                message_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        stamp = _iso_days_ago(age_days)
        conn.execute(
            "INSERT INTO agent_sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
            (session_id, stamp, stamp),
        )
        for i in range(messages):
            conn.execute(
                "INSERT INTO agent_messages (session_id, message_data) VALUES (?, ?)",
                (session_id, f'{{"role": "user", "content": "m{i}"}}'),
            )
        conn.commit()


def _write_eval_run(db: Path, age_days: float, cases: int = 2) -> None:
    with closing(sqlite3.connect(db)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                score REAL NOT NULL,
                pass_rate REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS eval_cases (
                run_id INTEGER NOT NULL REFERENCES eval_runs(id),
                case_index INTEGER NOT NULL,
                input TEXT NOT NULL,
                output TEXT,
                passed INTEGER NOT NULL,
                results_json TEXT NOT NULL,
                trace_id TEXT,
                PRIMARY KEY (run_id, case_index)
            );
        """)
        cursor = conn.execute(
            "INSERT INTO eval_runs (agent_name, created_at, score, pass_rate) VALUES (?, ?, ?, ?)",
            ("a", _iso_days_ago(age_days), 1.0, 1.0),
        )
        for i in range(cases):
            conn.execute(
                "INSERT INTO eval_cases "
                "(run_id, case_index, input, output, passed, results_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (cursor.lastrowid, i, "in", "out", 1, "{}"),
            )
        conn.commit()


def _rows(db: Path, table: str) -> int:
    with closing(sqlite3.connect(db)) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_old_traces_and_their_spans_go(tmp_path: Path) -> None:
    """A trace past the cutoff takes its spans with it; the foreign key is respected."""
    db = tmp_path / "runa.db"
    _write_trace(db, "old", age_days=90)

    pruned = prune(older_than_days=30, db_path=db)

    assert (pruned.traces, pruned.spans) == (1, 2)
    assert _rows(db, "traces") == 0
    assert _rows(db, "spans") == 0


def test_recent_traces_stay(tmp_path: Path) -> None:
    """The cutoff is a cutoff: anything newer is untouched."""
    db = tmp_path / "runa.db"
    _write_trace(db, "new", age_days=2)
    _write_trace(db, "old", age_days=90)

    pruned = prune(older_than_days=30, db_path=db)

    assert pruned.traces == 1
    assert _rows(db, "traces") == 1


def test_old_sessions_and_their_messages_go(tmp_path: Path) -> None:
    """A session is pruned by `updated_at`, so an idle conversation ages out with its messages."""
    db = tmp_path / "runa.db"
    _write_session(db, "old", age_days=60)
    _write_session(db, "fresh", age_days=1)

    pruned = prune(older_than_days=30, db_path=db)

    assert (pruned.sessions, pruned.messages) == (1, 3)
    assert _rows(db, "agent_sessions") == 1
    assert _rows(db, "agent_messages") == 3


def test_old_eval_runs_and_their_cases_go(tmp_path: Path) -> None:
    """Eval history ages out the same way, by `created_at`."""
    db = tmp_path / "runa.db"
    _write_eval_run(db, age_days=200)

    pruned = prune(older_than_days=30, db_path=db)

    assert (pruned.eval_runs, pruned.eval_cases) == (1, 2)
    assert _rows(db, "eval_runs") == 0
    assert _rows(db, "eval_cases") == 0


def test_dry_run_counts_without_deleting(tmp_path: Path) -> None:
    """An operator can see the damage before agreeing to it."""
    db = tmp_path / "runa.db"
    _write_trace(db, "old", age_days=90)

    pruned = prune(older_than_days=30, db_path=db, dry_run=True)

    assert pruned.traces == 1
    assert _rows(db, "traces") == 1  # still there


def test_kinds_narrows_what_is_touched(tmp_path: Path) -> None:
    """`--only traces` leaves sessions and evals alone."""
    db = tmp_path / "runa.db"
    _write_trace(db, "old", age_days=90)
    _write_session(db, "old", age_days=90)

    pruned = prune(older_than_days=30, db_path=db, kinds=("traces",))

    assert pruned.traces == 1
    assert pruned.sessions == 0
    assert _rows(db, "agent_sessions") == 1


def test_an_unknown_kind_is_rejected(tmp_path: Path) -> None:
    """A typo should not silently prune nothing and report success."""
    db = tmp_path / "runa.db"
    _write_trace(db, "old", age_days=90)

    with pytest.raises(ValueError, match="unknown kind"):
        prune(older_than_days=30, db_path=db, kinds=("tracez",))


def test_a_database_with_only_some_tables_is_fine(tmp_path: Path) -> None:
    """An app that has never run an eval has no `eval_runs`; pruning it is not an error."""
    db = tmp_path / "runa.db"
    _write_trace(db, "old", age_days=90)

    pruned = prune(older_than_days=30, db_path=db)

    assert pruned.traces == 1
    assert pruned.eval_runs == 0


def test_a_missing_database_is_a_no_op(tmp_path: Path) -> None:
    """Pruning before anything has been written must not create or crash on the file."""
    db = tmp_path / "never-written.db"

    pruned = prune(older_than_days=30, db_path=db)

    assert pruned.total == 0
    assert not db.exists()


def test_pruning_reclaims_disk(tmp_path: Path) -> None:
    """`VACUUM` runs after a real prune: reclaiming the file is the point of the exercise."""
    db = tmp_path / "runa.db"
    for i in range(200):
        _write_trace(db, f"t{i}", age_days=90, spans=5)
    before = db.stat().st_size

    prune(older_than_days=30, db_path=db)

    assert db.stat().st_size < before


def test_nothing_old_enough_changes_nothing(tmp_path: Path) -> None:
    """The common case: a young database, a no-op prune."""
    db = tmp_path / "runa.db"
    _write_trace(db, "new", age_days=1)

    assert prune(older_than_days=30, db_path=db).total == 0
    assert _rows(db, "traces") == 1
