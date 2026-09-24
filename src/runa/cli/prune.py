"""cli/prune.py: `runa prune`, the retention command over this app's `db/runa.db`.

Thin formatting over `runa.db.prune`, the same way `cli/traces.py` only formats what
`runa.tracing.storage` already exposes.
"""

from __future__ import annotations

from pathlib import Path

from runa.cli._project import resolve_db_path
from runa.db.prune import Pruned, prune


def _lines(pruned: Pruned) -> list[str]:
    """One line per kind that actually had something to remove."""
    rows = (
        ("traces", pruned.traces, "spans", pruned.spans),
        ("sessions", pruned.sessions, "messages", pruned.messages),
        ("eval runs", pruned.eval_runs, "cases", pruned.eval_cases),
    )
    return [
        f"  {count} {label} ({child_count} {child})"
        for label, count, child, child_count in rows
        if count
    ]


def prune_cli(
    *,
    root: Path,
    older_than_days: int,
    kinds: tuple[str, ...],
    dry_run: bool = False,
) -> str:
    """Prune `root`'s `db/runa.db` and describe what went (or, for `--dry-run`, what would)."""
    pruned = prune(
        older_than_days=older_than_days,
        db_path=resolve_db_path(root),
        kinds=kinds,
        dry_run=dry_run,
    )
    if not pruned.total:
        return f"nothing older than {older_than_days} days"

    verb = "would remove" if dry_run else "removed"
    return "\n".join([f"{verb}, older than {older_than_days} days:", *_lines(pruned)])


__all__ = ["prune_cli"]
