"""db/prune.py: deleting what `runa.db` has outgrown.

Everything Runa persists is append-only: every run adds a trace and its spans, every turn adds
session messages, every `runa eval` adds a run and its cases. Nothing removed any of it, so a
long-lived deployment's `runa.db` grew without bound and the only cure was deleting the file,
which also threw away the history worth keeping.

One cutoff, applied to all three, because they are the same decision ("how far back do we care?")
rather than three policies to configure separately. Each table's own timestamp column decides what
is old: `traces.start_time` (epoch seconds), `agent_sessions.updated_at` and `eval_runs.created_at`
(ISO text), so a session stays alive as long as it is still being talked to.

Prunes whichever store the app actually uses: the shared Postgres when `RUNA_DATABASE_URL` is set
(where unbounded growth matters most, since that database outlives every replica writing to it),
the local `db/runa.db` otherwise.
"""

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from runa.db import DEFAULT_DB_PATH, shared_url


@dataclass
class Pruned:
    """How many rows of each kind a prune removed (or would remove, for a dry run)."""

    traces: int = 0
    spans: int = 0
    sessions: int = 0
    messages: int = 0
    eval_runs: int = 0
    eval_cases: int = 0

    @property
    def total(self) -> int:
        """Every row across every kind, for a caller that just wants "did anything go?"."""
        return (
            self.traces
            + self.spans
            + self.sessions
            + self.messages
            + self.eval_runs
            + self.eval_cases
        )


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """Whether `name` is a table in this database.

    A `runa.db` only has the tables whatever wrote to it first created, so an app that has never
    run an eval has no `eval_runs`. Pruning what was never written is not an error.
    """
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _count(conn: sqlite3.Connection, sql: str, params: tuple[object, ...]) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0]) if row else 0


def _prune_traces(conn: sqlite3.Connection, cutoff: float, pruned: Pruned, commit: bool) -> None:
    if not _table_exists(conn, "traces"):
        return
    ids = [r[0] for r in conn.execute("SELECT id FROM traces WHERE start_time < ?", (cutoff,))]
    if not ids:
        return
    marks = ",".join("?" * len(ids))
    pruned.traces = len(ids)
    pruned.spans = _count(
        conn, f"SELECT COUNT(*) FROM spans WHERE trace_id IN ({marks})", tuple(ids)
    )
    if commit:
        # Spans first: they carry the foreign key into `traces`.
        conn.execute(f"DELETE FROM spans WHERE trace_id IN ({marks})", tuple(ids))
        conn.execute(f"DELETE FROM traces WHERE id IN ({marks})", tuple(ids))


def _prune_sessions(conn: sqlite3.Connection, cutoff: str, pruned: Pruned, commit: bool) -> None:
    if not _table_exists(conn, "agent_sessions"):
        return
    ids = [
        r[0]
        for r in conn.execute(
            "SELECT session_id FROM agent_sessions WHERE updated_at < ?", (cutoff,)
        )
    ]
    if not ids:
        return
    marks = ",".join("?" * len(ids))
    pruned.sessions = len(ids)
    pruned.messages = _count(
        conn, f"SELECT COUNT(*) FROM agent_messages WHERE session_id IN ({marks})", tuple(ids)
    )
    if commit:
        conn.execute(f"DELETE FROM agent_messages WHERE session_id IN ({marks})", tuple(ids))
        conn.execute(f"DELETE FROM agent_sessions WHERE session_id IN ({marks})", tuple(ids))


def _prune_evals(conn: sqlite3.Connection, cutoff: str, pruned: Pruned, commit: bool) -> None:
    if not _table_exists(conn, "eval_runs"):
        return
    ids = [r[0] for r in conn.execute("SELECT id FROM eval_runs WHERE created_at < ?", (cutoff,))]
    if not ids:
        return
    marks = ",".join("?" * len(ids))
    pruned.eval_runs = len(ids)
    pruned.eval_cases = _count(
        conn, f"SELECT COUNT(*) FROM eval_cases WHERE run_id IN ({marks})", tuple(ids)
    )
    if commit:
        conn.execute(f"DELETE FROM eval_cases WHERE run_id IN ({marks})", tuple(ids))
        conn.execute(f"DELETE FROM eval_runs WHERE id IN ({marks})", tuple(ids))


def prune(
    *,
    older_than_days: int,
    db_path: Path = DEFAULT_DB_PATH,
    dry_run: bool = False,
    vacuum: bool = True,
) -> Pruned:
    """Delete traces, sessions and eval runs older than `older_than_days`, reporting the counts.

    `dry_run` counts without deleting, so an operator can see the damage before agreeing to it.
    `vacuum` reclaims the freed pages afterward: SQLite does not shrink the file on `DELETE`
    alone, and reclaiming disk is usually the whole reason for pruning.

    `db_path` is where the *local* file lives, for a CLI invoked against another project's
    `--root`; a shared deployment has one database and leaves it unread.
    """
    pruned = Pruned()
    now = datetime.now(UTC)
    if (url := shared_url()) is not None:
        from runa.db import prune_postgres

        return prune_postgres.prune(
            url=url, now=now, older_than_days=older_than_days, dry_run=dry_run
        )
    if not Path(db_path).exists():
        return pruned

    epoch_cutoff = (now - timedelta(days=older_than_days)).timestamp()
    iso_cutoff = (now - timedelta(days=older_than_days)).isoformat()

    with closing(sqlite3.connect(db_path)) as conn:
        _prune_traces(conn, epoch_cutoff, pruned, not dry_run)
        _prune_sessions(conn, iso_cutoff, pruned, not dry_run)
        _prune_evals(conn, iso_cutoff, pruned, not dry_run)
        if not dry_run:
            conn.commit()
            if vacuum and pruned.total:
                conn.execute("VACUUM")

    return pruned


__all__ = ["Pruned", "prune"]
