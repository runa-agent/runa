"""tracing/postgres.py: `PostgresExporter`, traces exported to one specific Postgres.

The `runa[postgres]` extra, not a core dependency. The default exporter is a plain `StoreExporter`
over whatever `runa.db.traces()` resolved, so this name is for an app that wants to export to a
database other than this deployment's.
"""

from runa.db.pool import PostgresDatabase
from runa.tracing.database import DatabaseTraceStore
from runa.tracing.export import StoreExporter
from runa.tracing.store import SPANS, TRACES


class PostgresExporter(StoreExporter):
    """A `TraceExporter` that persists every finished trace to Postgres."""

    def __init__(self, url: str) -> None:
        """Export to `url`'s `traces`/`spans` tables."""
        super().__init__(DatabaseTraceStore(PostgresDatabase(url, TRACES, SPANS)))


__all__ = ["PostgresExporter"]
