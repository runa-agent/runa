"""db/prune_postgres.py: the retention pass, against a shared Postgres.

Same cutoff and the same `Pruned` counts as `db/prune.py`'s SQLite path, so `runa prune` reports
the same thing whichever store the app is on. Kept beside it rather than inside it so the core
package never imports `asyncpg`: `prune.py` is part of a plain install, this is the
`runa[postgres]` extra.

No `VACUUM` here. Postgres autovacuum reclaims the space itself, and a manual `VACUUM FULL` takes
an exclusive lock on the table, which is not something a retention job should do to a database
other replicas are actively writing to.
"""

from datetime import datetime, timedelta
from typing import Any

from runa.db.pool import get_pool, run_sync
from runa.db.prune import Pruned


async def _table_exists(conn: Any, name: str) -> bool:
    """Whether `name` is a table in this database; an app may never have run an eval."""
    return bool(await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", f"public.{name}"))


async def _delete(
    conn: Any,
    *,
    parent: str,
    child: str,
    parent_key: str,
    child_key: str,
    where: str,
    cutoff: object,
    dry_run: bool,
) -> tuple[int, int]:
    """Count (and unless `dry_run`, delete) rows of `parent` past `cutoff`, plus their children."""
    if not await _table_exists(conn, parent):
        return 0, 0
    ids = [
        row[0]
        for row in await conn.fetch(f"SELECT {parent_key} FROM {parent} WHERE {where} < $1", cutoff)
    ]
    if not ids:
        return 0, 0
    children = 0
    if await _table_exists(conn, child):
        children = int(
            await conn.fetchval(f"SELECT COUNT(*) FROM {child} WHERE {child_key} = ANY($1)", ids)
        )
        if not dry_run:
            await conn.execute(f"DELETE FROM {child} WHERE {child_key} = ANY($1)", ids)
    if not dry_run:
        await conn.execute(f"DELETE FROM {parent} WHERE {parent_key} = ANY($1)", ids)
    return len(ids), children


async def _prune(url: str, now: datetime, older_than_days: int, dry_run: bool) -> Pruned:
    pruned = Pruned()
    cutoff = now - timedelta(days=older_than_days)
    epoch_cutoff = cutoff.timestamp()
    iso_cutoff = cutoff.isoformat()

    pool = await get_pool(url)
    async with pool.acquire() as conn, conn.transaction():
        pruned.traces, pruned.spans = await _delete(
            conn,
            parent="traces",
            child="spans",
            parent_key="id",
            child_key="trace_id",
            where="start_time",
            cutoff=epoch_cutoff,
            dry_run=dry_run,
        )
        pruned.sessions, pruned.messages = await _delete(
            conn,
            parent="agent_sessions",
            child="agent_messages",
            parent_key="session_id",
            child_key="session_id",
            where="updated_at",
            cutoff=cutoff,
            dry_run=dry_run,
        )
        pruned.eval_runs, pruned.eval_cases = await _delete(
            conn,
            parent="eval_runs",
            child="eval_cases",
            parent_key="id",
            child_key="run_id",
            where="created_at",
            cutoff=iso_cutoff,
            dry_run=dry_run,
        )
    return pruned


def prune(*, url: str, now: datetime, older_than_days: int, dry_run: bool) -> Pruned:
    """Delete traces, sessions and eval runs older than `older_than_days` from `url`."""
    return run_sync(_prune(url, now, older_than_days, dry_run))


__all__ = ["prune"]
