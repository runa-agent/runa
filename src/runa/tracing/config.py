"""tracing/config.py: the privacy policy every captured span/trace goes through, plus `observe()`.

One small, global, mutable policy, not a policy engine. `capture_inputs`/`capture_outputs` gate
whether input/output are kept at all; `redact`/`redactor` scrub what's kept; the `max_*_bytes`
limits truncate what's left. `run_internal/spans.py` applies it as each span opens and closes,
and `lifecycle.py` applies the same policy to its DEBUG log lines, so one setting covers both.
"""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from runa.tracing.store import TraceStore
from runa.tracing.traces import Trace

_REDACTED = "[REDACTED]"


class TraceExporter(Protocol):
    """Exports a finished `Trace` somewhere: SQLite, the console, or a caller's own backend.

    `export` is synchronous because it's called from `_export` (`run_internal/spans.py`, or
    `tracing/manual.py` for a manual `trace` block) as a run finishes, from inside whatever loop
    that run was already on: there is nothing to await from there, and `asyncio.run()` demands
    there be no running loop. `TraceStore` is synchronous for the same reason, plus the
    `runa traces`/`runa ui` read path. An exporter over an async backend reaches a loop through
    `db/pool.run_sync`, as `PostgresTraceStore` does.
    """

    def export(self, trace: Trace) -> None:
        """Export `trace`. Exceptions are caught by the caller; tracing must never fail open."""
        ...


class StoreExporter:
    """A `TraceExporter` that writes every finished trace to a `TraceStore`.

    The whole implementation, for every persistent exporter: which store it holds is decided once,
    when the exporter is built, instead of per write. `SQLiteExporter` and `PostgresExporter` are
    this class with a store already chosen, and `_local_exporter` builds one over whatever
    `runa.db.traces()` resolved.

    Deciding once is a fix, not a tidy-up. The old `SQLiteExporter.export` called a module-level
    `save_trace` that asked `runa.db` for the backend all over again, so the exporter a
    deployment had installed and the store its traces landed in were two separate answers to one
    question.
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

    Checking `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` here (rather than requiring an explicit
    `add_exporter(LangfuseExporter())` call) is what lets a Langfuse project just work the moment
    its keys are set, no code change needed -- the same "no separate setup step" property the
    rest of tracing already has. The `runa[langfuse]` extra not being installed is the normal
    case for most apps even with those env vars unset; silently skipping it here, rather than
    raising, keeps that normal.
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

    No fallback if the `postgres` extra is missing. Quietly writing traces to SQLite while
    sessions and eval history go to Postgres splits one deployment's story across two stores to
    avoid an error message, and the next session call would raise the same `ImportError` anyway.
    """
    from runa import db

    return StoreExporter(db.traces())


@dataclass
class _Config:
    capture_inputs: bool = True
    capture_outputs: bool = True
    redact: list[str] | None = None
    redactor: Callable[[Any], Any] | None = None
    max_input_bytes: int = 32_000
    max_output_bytes: int = 32_000
    max_tool_result_bytes: int = 32_000
    exporters: list[TraceExporter] | None = None
    """`None` until first read, when `exporters()` resolves `_default_exporters()`.

    Not a `default_factory`: this dataclass is instantiated at import, and resolving the default
    there would read `RUNA_DATABASE_URL`/`LANGFUSE_*` before a generated `main.py` has run its
    `load_dotenv()`. An app whose `.env` names a shared database would then write its traces to
    SQLite while its sessions went to Postgres, which is the exact split `runa.db` exists to
    prevent. Deciding on first use instead means the environment is whatever the app has set up
    by the time a trace actually needs exporting.
    """


_config = _Config()


def _redact_dict(value: dict[str, Any], redact: list[str]) -> dict[str, Any]:
    return {k: (_REDACTED if k in redact else v) for k, v in value.items()}


def _truncate(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore") + "... [truncated]"


def apply_policy(value: Any, *, max_bytes: int) -> Any:
    """Apply the active redact/size policy to one span's `input`/`output`/tool result.

    Callers check `capture_inputs()`/`capture_outputs()` themselves before calling this: this
    function only redacts and truncates a value that's already been decided worth keeping.
    """
    if value is None:
        return None
    if _config.redactor is not None:
        value = _config.redactor(value)
    if _config.redact and isinstance(value, dict):
        value = _redact_dict(value, _config.redact)
    if isinstance(value, dict | list):
        try:
            value = json.loads(_truncate(json.dumps(value, default=str), max_bytes))
        except TypeError, ValueError:
            value = _truncate(str(value), max_bytes)
    elif isinstance(value, str):
        value = _truncate(value, max_bytes)
    return value


def capture_inputs() -> bool:
    """Return whether span/trace input should be captured under the active policy."""
    return _config.capture_inputs


def capture_outputs() -> bool:
    """Return whether span/trace output should be captured under the active policy."""
    return _config.capture_outputs


def input_limit() -> int:
    """Return the active `max_input_bytes` limit."""
    return _config.max_input_bytes


def output_limit() -> int:
    """Return the active `max_output_bytes` limit."""
    return _config.max_output_bytes


def tool_result_limit() -> int:
    """Return the active `max_tool_result_bytes` limit."""
    return _config.max_tool_result_bytes


def exporters() -> list[TraceExporter]:
    """Return the active list of `TraceExporter`s a finished trace is sent to.

    Resolves the default on first call, so the environment that decides it is the one in place
    when tracing is first used rather than when `runa.tracing` was imported.
    """
    if _config.exporters is None:
        _config.exporters = _default_exporters()
    return _config.exporters


def add_exporter(exporter: TraceExporter) -> None:
    """Add `exporter` alongside whatever's active, instead of replacing it.

    `observe(exporter=...)` is a full override, since it's also how a bare `with observe(...)`
    block temporarily swaps exporters for its duration. That makes it the wrong tool for adding
    one more exporter (e.g. a second `LangfuseExporter`, pointed at a different project, or a
    `ConsoleExporter` for local debugging) without disabling the default store exporter and
    losing trace history. `LangfuseExporter` itself doesn't need this: `_default_exporters`
    already adds one automatically once `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set.
    """
    _config.exporters = [*exporters(), exporter]


class observe:
    """Configure the tracing privacy policy: usable as a plain call or as a `with` block.

    A bare `observe(capture_inputs=False)` call applies the setting immediately and leaves it in
    place. `with observe(capture_inputs=False):` applies it for the duration of the block and
    restores whatever was active before on exit. Both forms are supported per the design's
    "should be supported, but should not be required" guidance for the context-manager form.
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
        """Apply the given overrides to the global tracing policy immediately."""
        global _config
        self._previous = _config
        updates: dict[str, Any] = {
            k: v
            for k, v in {
                "capture_inputs": capture_inputs,
                "capture_outputs": capture_outputs,
                "redact": redact,
                "redactor": redactor,
                "max_input_bytes": max_input_bytes,
                "max_output_bytes": max_output_bytes,
                "max_tool_result_bytes": max_tool_result_bytes,
            }.items()
            if v is not None
        }
        if exporter is not None:
            updates["exporters"] = exporter if isinstance(exporter, list) else [exporter]
        _config = _Config(**{**self._previous.__dict__, **updates})

    def __enter__(self) -> observe:
        """Return self; the policy was already applied in `__init__`."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Restore whatever tracing policy was active before this `observe(...)` was created."""
        global _config
        _config = self._previous


__all__ = [
    "ConsoleExporter",
    "SQLiteExporter",
    "StoreExporter",
    "TraceExporter",
    "observe",
]
