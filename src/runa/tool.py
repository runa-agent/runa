"""`@tool` decorator for exposing functions to agents.

`tool` derives a JSON schema for the model to call a plain Python function by, purely by
reflection over its signature and docstring.
"""

import enum
import inspect
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, Literal, get_args, get_origin, get_type_hints, overload

from runa._calling import RESERVED_PARAMS, by_name
from runa._types import RunContext
from runa.approval import _NeedsApproval as _ApprovalPredicate
from runa.approval import predicate
from runa.guardrail import BoundGuardrail, GuardrailsList, Phase, flatten_guardrails

_NeedsApproval = bool | _ApprovalPredicate


@dataclass(kw_only=True)
class Tool:
    """What the model is offered and a run gates a call by: a schema, and the call's policy.

    It is what a hook is handed as a call's `tool`. A `FunctionTool` is one that also runs
    itself; a delegate's `tool` is a plain `Tool`, because its call is a nested run of the
    caller's (see `runa.handoff.Delegate`).
    """

    name: str
    description: str
    params_json_schema: dict[str, Any]
    guardrails: dict[Phase, list[BoundGuardrail]] = field(default_factory=dict)
    """This tool's guardrails by `Phase`, checked against the call's arguments
    (`Phase.TOOL_INPUT`) and its return value (`Phase.TOOL_OUTPUT`)."""
    needs_approval: _ApprovalPredicate = field(default=predicate(False))
    """Called with `(ctx, args, call_id)`: whether this call waits on a human before it runs."""


@dataclass(kw_only=True)
class FunctionTool(Tool):
    """A tool the model can call: its schema and policy, and the callable that actually runs it."""

    on_invoke_tool: Callable[[RunContext, dict[str, Any], str], Awaitable[Any]]
    """Called with `(ctx, args, call_id)`, `args` the model's arguments already parsed into a
    dict; returns what the tool does."""


@dataclass(frozen=True)
class ToolCall:
    """One tool call a run answered: the tool's name, its arguments, its result, its failure.

    A call the model made is recorded whether it ran or failed: the tool raised, the delegate's
    run errored, the arguments didn't parse, or no tool has that name. A call a human rejected
    is not, since nothing was attempted.

    Recorded by the turn loop as the call runs (`run_internal/tool_execution.py`) and kept on the
    `Run`, deliberately not derived from the run's `"tool"` trace spans: a span's input/output
    goes through the tracing privacy policy (`runa.tracing.observe`), so it may be withheld,
    redacted, or truncated. What an eval grades can't depend on an observability setting, so the
    facts live here and the spans stay the display copy.
    """

    name: str
    arguments: str
    """The call's arguments as JSON, exactly the string the model produced."""
    output: str
    """The tool's return value, exactly the string that went back to the model. For a failed
    call that is `"error: "` and its `error`."""
    children: tuple[ToolCall, ...] = ()
    """For a `.delegate` call, every tool call the delegate's own run answered, in order (its
    delegate calls with their own `children`); empty for any other call."""
    error: str | None = None
    """Why the call failed, or `None` if it succeeded: the one way to tell a failure from a tool
    that returned text starting with `"error: "`."""

    def walk(self) -> Iterator[ToolCall]:
        """This call, then every call under it, depth-first in the order they finished."""
        yield self
        for child in self.children:
            yield from child.walk()


def _json_type(annotation: Any) -> dict[str, Any]:
    """Map a Python type annotation to a JSON Schema fragment, best-effort.

    Recognizes `str`/`int`/`float`/`bool`, `list[T]`, `dict`, `Literal[...]`, `Enum` subclasses,
    and `T | None` (unwrapped to `T`'s schema, JSON Schema has no first-class optional). Anything
    else (a dataclass, a `TypedDict`, ...) falls back to an unconstrained `{}`, which still works,
    just without the model getting a shape hint for it.
    """
    if annotation is inspect.Signature.empty or annotation is Any:
        return {}
    origin = get_origin(annotation)
    if origin is UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        return _json_type(args[0]) if len(args) == 1 else {"anyOf": [_json_type(a) for a in args]}
    if origin is Literal:
        return {"enum": list(get_args(annotation))}
    if origin is list:
        (item,) = get_args(annotation) or (Any,)
        return {"type": "array", "items": _json_type(item)}
    if origin is dict:
        return {"type": "object"}
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return {"enum": [member.value for member in annotation]}
    json_type = {str: "string", int: "integer", float: "number", bool: "boolean"}.get(annotation)
    return {"type": json_type} if json_type else {}


def _split_docstring(doc: str, param_names: list[str]) -> tuple[str, dict[str, str]]:
    """Split a docstring into the tool's description and each of `param_names`' (see `tool`).

    A line is a parameter's only if its name is one of `param_names` and text follows the colon;
    any other line (`Note: ...`) stays in the description. A parameter's first line wins.
    """
    kept: list[str] = []
    described: dict[str, str] = {}
    for line in doc.splitlines():
        name, sep, text = line.strip().partition(":")
        if sep and name in param_names and text.strip() and name not in described:
            described[name] = text.strip()
        else:
            kept.append(line)
    return "\n".join(kept).strip(), described


def _schema_from_signature(func: Callable[..., Any]) -> tuple[dict[str, Any], list[str]]:
    """Build a `{"type": "object", "properties": {...}}` schema from `func`'s parameters.

    `ctx`/`call_id` are Runa's reserved parameter names (see `runa.approval`) for the run context
    and the tool call's id; they're never part of the model-facing schema. Returns the schema
    alongside the ordered list of the remaining (model-facing) parameter names. A parameter
    described in the docstring (see `tool`) carries that text as its `description`.
    """
    signature = inspect.signature(func)
    hints = get_type_hints(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    param_names: list[str] = []
    for param_name, param in signature.parameters.items():
        if param_name in RESERVED_PARAMS:
            continue
        param_names.append(param_name)
        properties[param_name] = _json_type(hints.get(param_name, param.annotation))
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    _, described = _split_docstring(inspect.getdoc(func) or "", param_names)
    for param_name, text in described.items():
        properties[param_name] = {**properties[param_name], "description": text}
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema, param_names


@overload
def tool(func: Callable[..., Any]) -> FunctionTool: ...


@overload
def tool(
    func: None = None,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    guardrails: GuardrailsList | None = None,
    needs_approval: _NeedsApproval = False,
) -> Callable[[Callable[..., Any]], FunctionTool]: ...


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    guardrails: GuardrailsList | None = None,
    needs_approval: _NeedsApproval = False,
) -> FunctionTool | Callable[[Callable[..., Any]], FunctionTool]:
    """Wrap a plain function as a `FunctionTool`, deriving its schema from its signature.

    The docstring is the tool's description, except a `<param>: <text>` line, which describes
    that parameter in the schema instead.

    `guardrails=[...]` wires `@guardrail` predicates, bare (wired as both sides)
    or bound via `.input`/`.output`, against the tool call's parsed arguments and its return
    value, respectively.
    """
    bound = flatten_guardrails(guardrails or [], tool=True)

    def decorator(fn: Callable[..., Any]) -> FunctionTool:
        schema, param_names = _schema_from_signature(fn)
        description, _ = _split_docstring(inspect.getdoc(fn) or "", param_names)

        return FunctionTool(
            name=name_override or fn.__name__,
            description=description_override or description,
            params_json_schema=schema,
            on_invoke_tool=by_name(fn),
            guardrails=bound,
            needs_approval=predicate(needs_approval),
        )

    return decorator(func) if func is not None else decorator


__all__ = ["FunctionTool", "Tool", "ToolCall", "tool"]
