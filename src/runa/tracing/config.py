"""tracing/config.py: the privacy policy every captured span/trace goes through, plus `observe()`.

Also how a finished trace reaches the exporters: `export_trace` on the calling thread,
`export_trace_later` queued off a run's loop, and `flush_exports` to wait for the queue.

`capture_inputs`/`capture_outputs` gate whether input/output are kept at all; `redact`/`redactor`
scrub what's kept; the `max_*_bytes` limits truncate what's left. Spans and `lifecycle.py`'s DEBUG
log lines both read it through `policy()`. It has a process-wide default (`observe(...)`) and a
per-task override (`with observe.scoped(...)`), held in a `ContextVar`.
"""

import contextvars
import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, Token
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Protocol

from runa import _app
from runa._log import logger
from runa.tracing.store import TraceStore
from runa.tracing.traces import Trace

_REDACTED = "[REDACTED]"


class TraceExporter(Protocol):
    """Exports a finished `Trace` somewhere: SQLite, the console, or a caller's own backend.

    `export` is synchronous and may block: a finishing run's trace is exported on tracing's own
    thread, and a synchronous caller (a `tracing.trace` block, `runa ui`) exports inline.
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
        self.store.save(trace)


class SQLiteExporter(StoreExporter):
    """The local exporter: persists every finished trace to `db/runa.db` (`tracing/sqlite.py`)."""

    def __init__(self, path: Path | None = None) -> None:
        """Export to `path`, or to wherever `runa.db` says this deployment's local file is."""
        from runa.tracing.sqlite import SQLiteTraceStore

        super().__init__(SQLiteTraceStore(path))


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


@dataclass(frozen=True)
class Policy:
    """The settings in force for one captured span: what `policy()` hands a caller.

    Frozen: a caller holding one can't change the policy others read. `observe(...)` and
    `observe.scoped(...)` replace it instead.
    """

    capture_inputs: bool = True
    capture_outputs: bool = True
    redact: list[str] | None = None
    redactor: Callable[[Any], Any] | None = None
    max_input_bytes: int = 32_000
    max_output_bytes: int = 32_000
    max_tool_result_bytes: int = 32_000
    exporters: list[TraceExporter] | None = None
    """`None` until first read, when `exporters()` resolves `_default_exporters()`.

    The one field `policy()`'s caller doesn't read directly: `exporters()` is what resolves this,
    so that the environment deciding it is the one in place when tracing is first used.

    Not a `default_factory`: this dataclass is instantiated at import, and resolving the default
    there would read `RUNA_DATABASE_URL`/`LANGFUSE_*` before a generated `main.py` has run its
    `load_dotenv()`. An app whose `.env` names a shared database would then write its traces to
    SQLite while its sessions went to Postgres, which is the exact split `runa.db` exists to
    prevent. Deciding on first use instead means the environment is whatever the app has set up
    by the time a trace actually needs exporting.
    """


_DEFAULT = "tracing.policy"

_scoped: ContextVar[Policy | None] = ContextVar("runa_tracing_policy", default=None)
"""The `observe.scoped(...)` policy for the current task, or `None` to fall back to `_default`."""


def policy() -> Policy:
    """The policy in force here: this task's `observe.scoped(...)` block, else the default.

    The whole read side of this module. Read the fields a caller needs off the result
    (`policy().max_input_bytes`); there is no per-field accessor to import.
    """
    scoped = _scoped.get()
    return scoped if scoped is not None else _default()


def _merge(base: Policy, updates: dict[str, Any]) -> Policy:
    """`base` with `updates` on top, as a new `Policy`; nothing is mutated in place."""
    return replace(base, **updates)


def _default() -> Policy:
    """The process-wide policy, which is what an `observe(...)` call and `add_exporter` write.

    It is a process fact held in `runa._app`, so whatever resets the process also resets this.
    """
    return _app.held(_DEFAULT, Policy)


def _set_default(default: Policy) -> None:
    """Install `default` as the process-wide policy."""
    _app.hold(_DEFAULT, default)


def _redact_dict(value: dict[str, Any], redact: list[str]) -> dict[str, Any]:
    return {k: (_REDACTED if k in redact else v) for k, v in value.items()}


def _truncate(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore") + "... [truncated]"


def apply_policy(value: Any, *, max_bytes: int) -> Any:
    """Apply the active redact/size policy to one span's `input`/`output`/tool result.

    Callers check `policy().capture_inputs`/`.capture_outputs` themselves before calling this:
    this function only redacts and truncates a value that's already been decided worth keeping.
    """
    if value is None:
        return None
    active = policy()
    if active.redactor is not None:
        value = active.redactor(value)
    if active.redact and isinstance(value, dict):
        value = _redact_dict(value, active.redact)
    if isinstance(value, dict | list):
        try:
            value = json.loads(_truncate(json.dumps(value, default=str), max_bytes))
        except TypeError, ValueError:
            value = _truncate(str(value), max_bytes)
    elif isinstance(value, str):
        value = _truncate(value, max_bytes)
    return value


def _resolved_default_exporters() -> list[TraceExporter]:
    """The process-wide exporter list, resolving `_default_exporters()` on first use.

    Memoized on `_default`, so a scoped override of another field never re-resolves it.
    """
    if (installed := _default().exporters) is not None:
        return installed
    resolved = _default_exporters()
    _set_default(_merge(_default(), {"exporters": resolved}))
    return resolved


def exporters() -> list[TraceExporter]:
    """Return the active list of `TraceExporter`s a finished trace is sent to.

    Resolves the default on first call, so the environment that decides it is the one in place
    when tracing is first used rather than when `runa.tracing` was imported.
    """
    active = policy().exporters
    return active if active is not None else _resolved_default_exporters()


def add_exporter(exporter: TraceExporter) -> None:
    """Add `exporter` alongside whatever's already installed, instead of replacing it.

    `observe(exporter=...)` replaces the list; this keeps the default store exporter. Writes the
    process-wide default, as startup configuration.
    """
    _set_default(_merge(_default(), {"exporters": [*_resolved_default_exporters(), exporter]}))


_OVERRIDABLE = tuple(f.name for f in fields(Policy) if f.name != "exporters")
"""Every `Policy` field an `observe(...)` call overrides under the field's own name.

`exporters` is left out: it is set through `exporter=`, which also takes a bare exporter.
"""


def _updates(named: dict[str, Any]) -> dict[str, Any]:
    """The overrides that were actually named, as the `Policy` field names `_merge` takes.

    `named` is the calling signature's `locals()`. A parameter renamed out of step with its
    field raises here rather than overriding nothing.
    """
    updates: dict[str, Any] = {n: named[n] for n in _OVERRIDABLE if named[n] is not None}
    if (exporter := named["exporter"]) is not None:
        updates["exporters"] = exporter if isinstance(exporter, list) else [exporter]
    return updates


class _ScopedPolicy:
    """What `observe.scoped(...)` returns: its overrides, installed on the current task only.

    Reusable and reentrant: each entry restores the policy in force where it was entered.
    """

    def __init__(self, updates: dict[str, Any]) -> None:
        """Hold `updates` until a `with` block installs them."""
        self._updates = updates
        self._tokens: list[Token[Policy | None]] = []

    def __enter__(self) -> _ScopedPolicy:
        """Install these overrides on top of whatever policy is in force here."""
        self._tokens.append(_scoped.set(_merge(policy(), self._updates)))
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Restore whatever tracing policy this task saw before the block."""
        _scoped.reset(self._tokens.pop())


class observe:
    """Configure what tracing captures: this call for the process, `.scoped(...)` for one task.

    `observe(capture_inputs=False)` applies the setting to the process-wide default and leaves it
    there: it is what an application configures itself with, once, at startup.
    `with observe.scoped(capture_inputs=False):` applies it to the current task only, and restores
    whatever that task saw before on exit.

    `.scoped` is a `ContextVar`, so one of `runa serve`'s concurrent requests never decides what
    another captures. A task doesn't see an override installed by a task started after it, so
    the plain call, not a block held open around the application, configures a deployment.
    """

    def __init__(
        self,
        *,
        capture_inputs: bool | None = None,
        capture_outputs: bool | None = None,
        redact: list[str] | None = None,
        redactor: Callable[[Any], Any] | None = None,
        max_input_bytes: int | None = None,
        max_output_bytes: int | None = None,
        max_tool_result_bytes: int | None = None,
        exporter: TraceExporter | list[TraceExporter] | None = None,
    ) -> None:
        """Apply the given overrides to the process-wide tracing policy, permanently.

        Not a context manager: `with observe(...)` is a `TypeError`; use `observe.scoped(...)`.
        """
        _set_default(_merge(_default(), _updates(locals())))

    @staticmethod
    def scoped(
        *,
        capture_inputs: bool | None = None,
        capture_outputs: bool | None = None,
        redact: list[str] | None = None,
        redactor: Callable[[Any], Any] | None = None,
        max_input_bytes: int | None = None,
        max_output_bytes: int | None = None,
        max_tool_result_bytes: int | None = None,
        exporter: TraceExporter | list[TraceExporter] | None = None,
    ) -> _ScopedPolicy:
        """A `with` block that applies the given overrides to the current task only.

        Nothing is applied until the block is entered, and everything is restored when it exits.
        An inner block overrides what it names and inherits the rest from the block around it.
        """
        return _ScopedPolicy(_updates(locals()))


__all__ = [
    "ConsoleExporter",
    "SQLiteExporter",
    "StoreExporter",
    "TraceExporter",
    "observe",
]
