"""tracing/config.py: the privacy policy every captured span/trace goes through, plus `observe()`.

Also the one place that knows how a finished trace reaches the exporters: `export_trace` for a
synchronous caller, `export_trace_off_loop` for a run that has a loop to protect. Which thread an
exporter runs on is this module's decision to make -- see `TraceExporter`.

One small policy, not a policy engine. `capture_inputs`/`capture_outputs` gate whether
input/output are kept at all; `redact`/`redactor` scrub what's kept; the `max_*_bytes`
limits truncate what's left. `run_internal/spans.py` applies it as each span opens and closes,
and `lifecycle.py` applies the same policy to its DEBUG log lines, so one setting covers both.
Both reach it through `policy()`, the module's one read accessor, which hands over the whole
`Policy` rather than spelling one function per field.

The policy lives in two places because its two call sites mean two different things by it: a
process-wide default an application sets once at startup (`observe(...)`), and a per-task
override that one `with observe.scoped(...)` block installs for its own duration. The override is
a `ContextVar`, the way `tracing/manual.py` already holds the current trace, so a block opened
while serving one request can't change what tracing captures for every other request in flight.
"""

import asyncio
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

    `export` is synchronous, and blocking in it is expected: an exporter is plain code doing
    plain I/O -- a file write, an HTTP POST, `db/pool.run_sync` over an async driver as
    `PostgresTraceStore` does -- and which thread it runs on is this module's decision rather
    than the exporter's. A finishing run awaits `export_trace_off_loop`, so the write happens on
    tracing's own thread while the loop goes on serving every other request; a caller that is
    synchronous all the way down (a `tracing.trace` block, `runa traces`, `runa ui`) calls
    `export_trace` inline on the thread it already has. `TraceStore` is synchronous for the same
    reason.
    """

    def export(self, trace: Trace) -> None:
        """Export `trace`. Exceptions are caught by the caller; tracing must never fail open."""
        ...


_exporting = ThreadPoolExecutor(max_workers=1, thread_name_prefix="runa-export")
"""The thread every off-loop export runs on, created on the first one.

One worker, deliberately. An exporter is usually a user's own class, so "your `export` is never
called from two threads at once" is a promise worth more than exporting two traces in parallel,
and it keeps traces exported in the order runs finished. It is also why this is not
`asyncio.to_thread`: that pool is shared with the sync `@tool` and `@guardrail` offloads, where a
burst of slow trace writes would sit in the slots a tool call needs.
"""


def export_trace(trace: Trace) -> None:
    """Hand `trace` to every exporter active here, in order, on the calling thread.

    Fail-open is the whole shape of this function: an exporter that raises is logged and the next
    one still runs, because losing a trace is a worse outcome than losing one backend's copy of
    it, and failing the run that produced it is worse than both.
    """
    for exporter in exporters():
        try:
            exporter.export(trace)
        except Exception:  # noqa: BLE001 -- tracing must never break the caller
            logger.warning("tracing: exporter %r failed", exporter, exc_info=True)


async def export_trace_off_loop(trace: Trace) -> None:
    """`export_trace`, moved onto `_exporting` so a slow exporter can't stall the event loop.

    The caller is a finishing run, which under `runa serve` shares one loop with every other
    request in flight. A blocking exporter called there -- a Langfuse OTLP POST with retries
    behind it, a `PostgresTraceStore.save` that waits up to 30s on a struggling database -- would
    stop that loop dead, turning one run's observability hiccup into every concurrent run's
    latency.

    The run still waits for its own export, which is what keeps `Run` and the trace store
    consistent for the caller: once `run_sync` returns, the trace is written. The context is
    copied across because `exporters()` reads `observe.scoped(...)`'s `ContextVar`, which
    `run_in_executor` would otherwise leave behind.
    """
    context = contextvars.copy_context()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_exporting, lambda: context.run(export_trace, trace))


class StoreExporter:
    """A `TraceExporter` that writes every finished trace to a `TraceStore`.

    The whole implementation, for every persistent exporter: which store it holds is decided once,
    when the exporter is built, instead of per write. `SQLiteExporter` and `PostgresExporter` are
    this class with a store already chosen, and `_local_exporter` builds one over whatever
    `runa.db.traces()` resolved. The exporter a deployment installed and the store its traces land
    in are therefore one answer, not two.
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


@dataclass(frozen=True)
class Policy:
    """The settings in force for one captured span: what `policy()` hands a caller.

    Handed over whole rather than one field at a time. A caller that keeps a value and the limit
    that applies to it (`run_internal/spans.py`, `lifecycle.py`) wants two fields of one policy,
    not two functions that each read a different global, and a new setting here costs its readers
    nothing. Frozen for that reason: the object outlives the call that asked for it, so a caller
    holding one must not be able to write the policy every other caller is reading. `observe(...)`
    and `observe.scoped(...)` replace it instead (`_merge`), which is also what makes a
    `ContextVar` override safe to share.
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

    Memoized on `_default` rather than on whatever `policy()` returned, so an
    `observe.scoped(...)` block that overrides something other than `exporter` doesn't re-resolve
    the list -- a second store, a second `LangfuseExporter` -- once per task that opens one.
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

    `observe(exporter=...)` is a full override, since it's also how an `observe.scoped(...)`
    block temporarily swaps exporters for its duration. That makes it the wrong tool for adding one
    more exporter (e.g. a second `LangfuseExporter`, pointed at a different project, or a
    `ConsoleExporter` for local debugging) without disabling the default store exporter and
    losing trace history. `LangfuseExporter` itself doesn't need this: `_default_exporters`
    already adds one automatically once `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set.

    Writes the process-wide default, like an `observe(...)` call: adding an exporter is
    startup configuration, not something one task does for its own duration.
    """
    _set_default(_merge(_default(), {"exporters": [*_resolved_default_exporters(), exporter]}))


_OVERRIDABLE = tuple(f.name for f in fields(Policy) if f.name != "exporters")
"""Every `Policy` field an `observe(...)` call overrides under the field's own name.

Read off `Policy` rather than written out again, so a ninth setting is a field plus the two
public signatures that are its API -- not a third and fourth copy of the same list. `exporters`
is the one field left out: it is set through `exporter=`, singular, which also takes a bare
exporter rather than a one-item list.
"""


def _updates(named: dict[str, Any]) -> dict[str, Any]:
    """The overrides that were actually named, as the `Policy` field names `_merge` takes.

    `named` is the calling signature's own `locals()`, which holds exactly one entry per
    parameter. Both call sites spell their options out as keyword-only parameters, so a
    misspelled one is a `TypeError` where it was written; this is only the shared translation of
    what they collected, reading the names off `_OVERRIDABLE` rather than forwarding them one by
    one (five of the eight options share a type with a neighbour, so a swapped pair would
    type-check). A parameter renamed out of step with its field raises here on the first call
    rather than overriding nothing. Anything else `named` carries (`self`) is never read.
    """
    updates: dict[str, Any] = {n: named[n] for n in _OVERRIDABLE if named[n] is not None}
    if (exporter := named["exporter"]) is not None:
        updates["exporters"] = exporter if isinstance(exporter, list) else [exporter]
    return updates


class _ScopedPolicy:
    """What `observe.scoped(...)` returns: its overrides, installed on the current task only.

    It holds the overrides and nothing else -- no snapshot of the policy it will replace -- so it
    is reusable and order-independent: a module-level constant, entered from two tasks at once or
    nested inside itself, each entry reading the policy in force where it is entered and restoring
    exactly that on exit. The tokens are a stack for that reason, one per live entry.
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

    The two scopes are two call sites rather than one name that infers which was meant from
    whether a `with` follows it, the way `Guardrail.input`/`.output` and `Agent.handoff`/
    `.delegate` name the choice they make. The inference needed a constructor that wrote the
    process-wide default for `__enter__` to then undo, which made the object single-use (it
    captured the default it would restore) and the write itself invisible to anyone reading the
    call site. Naming the scope instead means nothing is captured and nothing is undone.

    The scopes differ in reach, deliberately. A block that changed the policy process-wide would
    mean one of `runa.serve`'s concurrent requests deciding what every other in-flight request
    captures, so `.scoped` is a `ContextVar`. That costs the opposite: a task does not see an
    override installed by a task that started after it, so the plain call -- not a block held open
    around the whole application -- is what configures a whole deployment.
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

        There is nothing to hold on to afterwards: the call *is* the configuration. An instance
        is not a context manager, so a `with observe(...)` written out of habit is a `TypeError`
        at that line -- pointing at `observe.scoped(...)` -- rather than a block whose policy
        silently outlives it.
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
