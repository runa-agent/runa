"""tracing/export.py: where a finished trace goes: the exporters, and the queue that feeds them.

`export_trace` hands a trace to every active exporter on the calling thread, `export_trace_later`
queues it off a run's loop, and `flush_exports` waits for the queue. The active list is a
process-wide default (`observe(exporter=...)`, `add_exporter`) with a per-task override
(`observe.scoped(exporter=...)`), held in a `ContextVar`. What a trace captures is `config.py`.
"""

import contextvars
import os
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Protocol

from runa import _app
from runa._log import logger
from runa.db.handle import run_sync
from runa.tracing.store import TraceStore
from runa.tracing.traces import Trace


class TraceExporter(Protocol):
    """Exports a finished `Trace` somewhere: SQLite, the console, or a caller's own backend.

    `export` is synchronous and may block: a finishing run's trace is exported on tracing's own
    thread, and a synchronous caller (a `tracing.trace` block) exports inline.
    """

    def export(self, trace: Trace) -> None:
        """Export `trace`. Exceptions are caught by the caller; tracing must never fail open."""
        ...


_exporting = ThreadPoolExecutor(max_workers=1, thread_name_prefix="runa-export")
"""The thread every queued export runs on, created on the first one.

One worker, so an exporter's `export` is never called from two threads at once and traces are
exported in the order runs finished. Its own pool, so slow exports never take the slots sync
`@tool`/`@guardrail` calls run in. Drained before the interpreter exits.
"""


def export_trace(trace: Trace) -> None:
    """Hand `trace` to every exporter active here, in order, on the calling thread.

    An exporter that raises is logged and the next one still runs; nothing raises to the caller.
    """
    for exporter in exporters():
        try:
            exporter.export(trace)
        except Exception:  # noqa: BLE001 -- tracing must never break the caller
            logger.warning("tracing: exporter %r failed", exporter, exc_info=True)


def export_trace_later(trace: Trace) -> None:
    """Queue `export_trace(trace)` on `_exporting` and return without waiting for it.

    So a slow exporter never holds up a run's loop, a `Run` can reach its caller before its trace
    reaches the store; `flush_exports` waits for it. The task's context is copied across, so the
    worker sees `observe.scoped(...)` overrides.
    """
    _exporting.submit(contextvars.copy_context().run, export_trace, trace)


def flush_exports() -> None:
    """Block until every export queued before this call has finished.

    The one worker runs exports in the order they were queued, so waiting for a no-op queued
    behind them is waiting for all of them. Blocks the calling thread: call it from one with no
    loop to stall, or through `asyncio.to_thread`.
    """
    _exporting.submit(lambda: None).result()


class StoreExporter:
    """A `TraceExporter` that writes every finished trace to a `TraceStore`.

    Every persistent exporter: `SQLiteExporter` and `PostgresExporter` are this class with a store
    already chosen, and `_local_exporter` builds one over `runa.db.traces()`.
    """

    def __init__(self, store: TraceStore) -> None:
        """Store where finished traces are written."""
        self.store = store

    def export(self, trace: Trace) -> None:
        """Persist `trace`. Exceptions are caught by the caller; tracing never fails a run."""
        run_sync(self.store.save(trace))


class SQLiteExporter(StoreExporter):
    """The local exporter: persists every finished trace to `db/runa.db`."""

    def __init__(self, path: Path | None = None) -> None:
        """Export to `path`, or to wherever `runa.db` says this deployment's local file is."""
        from runa.db import sqlite_database
        from runa.db.sqlite import SQLiteDatabase
        from runa.tracing.database import DatabaseTraceStore
        from runa.tracing.store import SPANS, TRACES

        location = path if path is not None else sqlite_database()
        super().__init__(DatabaseTraceStore(SQLiteDatabase(location, TRACES, SPANS)))


class ConsoleExporter:
    """Prints every finished trace's human-readable tree (`Trace.__str__`) to stdout."""

    def export(self, trace: Trace) -> None:
        """Print `trace`."""
        print(trace)  # noqa: T201 -- this exporter's entire job is printing


def _default_exporters() -> list[TraceExporter]:
    """This deployment's store, plus `LangfuseExporter` if its credentials are already in the env.

    Langfuse is on whenever `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set, with no code
    change; without the `runa[langfuse]` extra it is silently skipped.
    """
    exporters: list[TraceExporter] = [_local_exporter()]
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        try:
            from runa.tracing.langfuse import LangfuseExporter
        except ImportError:
            pass
        else:
            exporters.append(LangfuseExporter())
    return exporters


def _local_exporter() -> TraceExporter:
    """An exporter over whatever store `runa.db` resolved: one decision, asked once.

    No fallback to SQLite if the `postgres` extra is missing: traces follow the other concerns.
    """
    from runa import db

    return StoreExporter(db.traces())


_default: list[TraceExporter] | None = None
"""The process-wide list, `None` until first read.

Resolved on first use rather than at import, so the environment deciding it is the one in place
when a trace first needs exporting: resolving at import would read `RUNA_DATABASE_URL`/`LANGFUSE_*`
before a generated `main.py` has run its `load_dotenv()`, and an app whose `.env` names a shared
database would write its traces to SQLite while its sessions went to Postgres.
"""

_scoped: ContextVar[list[TraceExporter] | None] = ContextVar("runa_tracing_exporters", default=None)
"""The `observe.scoped(exporter=...)` list for the current task, or `None` to use `_default`."""


def _forget_default() -> None:
    global _default
    _default = None


_app.on_reset(_forget_default)


def _process_exporters() -> list[TraceExporter]:
    global _default
    if _default is None:
        _default = _default_exporters()
    return _default


def exporters() -> list[TraceExporter]:
    """Return the active list of `TraceExporter`s a finished trace is sent to.

    This task's `observe.scoped(exporter=...)` list, else the process-wide default, resolved on
    first call.
    """
    scoped = _scoped.get()
    return scoped if scoped is not None else _process_exporters()


def set_exporters(replacement: list[TraceExporter]) -> None:
    """Replace the process-wide list, as startup configuration (`observe(exporter=...)`)."""
    global _default
    _default = replacement


def add_exporter(exporter: TraceExporter) -> None:
    """Add `exporter` alongside whatever's already installed, instead of replacing it.

    `observe(exporter=...)` replaces the list; this keeps the default store exporter. Writes the
    process-wide default, as startup configuration.
    """
    set_exporters([*_process_exporters(), exporter])


def scope_exporters(replacement: list[TraceExporter] | None) -> Token[list[TraceExporter] | None]:
    """Make `replacement` this task's list (`None` keeps the one in force); undo with `unscope`."""
    return _scoped.set(replacement if replacement is not None else _scoped.get())


def unscope(token: Token[list[TraceExporter] | None]) -> None:
    """Restore the list this task saw before the `scope_exporters` call that made `token`."""
    _scoped.reset(token)
