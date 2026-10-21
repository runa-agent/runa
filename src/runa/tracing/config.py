"""tracing/config.py: the privacy policy every captured span/trace goes through, plus `observe()`.

`capture_inputs`/`capture_outputs` gate whether input/output are kept at all; `redact`/`redactor`
scrub what's kept; the `max_*_bytes` limits truncate what's left. Spans and `lifecycle.py`'s DEBUG
log lines both read it through `policy()`. It has a process-wide default (`observe(...)`) and a
per-task override (`with observe.scoped(...)`), held in a `ContextVar`. Where a finished trace
goes is `export.py`; `observe(exporter=...)` forwards there.
"""

import json
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, fields, replace
from typing import Any

from runa import _app
from runa.tracing import export

_REDACTED = "[REDACTED]"


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


_default_policy: Policy | None = None

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

    It is process state, so `runa._app.reset` forgets it with the app.
    """
    global _default_policy
    if _default_policy is None:
        _default_policy = Policy()
    return _default_policy


def _set_default(default: Policy) -> None:
    """Install `default` as the process-wide policy."""
    global _default_policy
    _default_policy = default


def _forget_default() -> None:
    global _default_policy
    _default_policy = None


_app.on_reset(_forget_default)


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


def _updates(named: dict[str, Any]) -> dict[str, Any]:
    """The policy overrides that were actually named, as the `Policy` field names `_merge` takes.

    `named` is the calling signature's `locals()`. A parameter renamed out of step with its
    field raises here rather than overriding nothing.
    """
    return {f.name: named[f.name] for f in fields(Policy) if named[f.name] is not None}


def _exporter_list(named: dict[str, Any]) -> list[export.TraceExporter] | None:
    """The `exporter=` argument as a list, or `None` if it wasn't named."""
    exporter = named["exporter"]
    if exporter is None:
        return None
    return exporter if isinstance(exporter, list) else [exporter]


class _ScopedPolicy:
    """What `observe.scoped(...)` returns: its overrides, installed on the current task only.

    Reusable and reentrant: each entry restores the policy in force where it was entered.
    """

    def __init__(
        self, updates: dict[str, Any], exporters: list[export.TraceExporter] | None
    ) -> None:
        """Hold `updates` and `exporters` until a `with` block installs them."""
        self._updates = updates
        self._exporters = exporters
        self._tokens: list[
            tuple[Token[Policy | None], Token[list[export.TraceExporter] | None]]
        ] = []

    def __enter__(self) -> _ScopedPolicy:
        """Install these overrides on top of whatever policy is in force here."""
        self._tokens.append(
            (
                _scoped.set(_merge(policy(), self._updates)),
                export.scope_exporters(self._exporters),
            )
        )
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Restore whatever tracing policy this task saw before the block."""
        policy_token, exporters_token = self._tokens.pop()
        export.unscope(exporters_token)
        _scoped.reset(policy_token)


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
        exporter: export.TraceExporter | list[export.TraceExporter] | None = None,
    ) -> None:
        """Apply the given overrides to the process-wide tracing policy, permanently.

        Not a context manager: `with observe(...)` is a `TypeError`; use `observe.scoped(...)`.
        """
        named = locals()
        _set_default(_merge(_default(), _updates(named)))
        if (exporters := _exporter_list(named)) is not None:
            export.set_exporters(exporters)

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
        exporter: export.TraceExporter | list[export.TraceExporter] | None = None,
    ) -> _ScopedPolicy:
        """A `with` block that applies the given overrides to the current task only.

        Nothing is applied until the block is entered, and everything is restored when it exits.
        An inner block overrides what it names and inherits the rest from the block around it.
        """
        named = locals()
        return _ScopedPolicy(_updates(named), _exporter_list(named))


__all__ = ["observe"]
