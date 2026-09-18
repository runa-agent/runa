"""`@tool` decorator for exposing functions to agents.

`tool` derives a JSON schema for the model to call a plain Python function by, purely by
reflection over its signature (`_schema_from_signature` below), the same "let types speak for
themselves" approach the rest of Runa follows, and the one the `pyright` override comment in
`pyproject.toml` already documents as this codebase's intended design for tool schemas.
"""

from __future__ import annotations

import asyncio
import enum
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import UnionType
from typing import Any, Literal, get_args, get_origin, get_type_hints, overload

from runa._types import RunContextWrapper
from runa.approval import _NeedsApproval as _ApprovalPredicate
from runa.guardrail import (
    ToolGuardrailsDict,
    ToolGuardrailsList,
    ToolInputGuardrail,
    ToolOutputGuardrail,
    flatten_tool_guardrails,
)

_RESERVED_PARAMS = ("ctx", "call_id")
_NeedsApproval = bool | _ApprovalPredicate


@dataclass
class FunctionTool:
    """A tool the model can call: its schema, and the callable that actually runs it."""

    name: str
    description: str
    params_json_schema: dict[str, Any]
    on_invoke_tool: Callable[[RunContextWrapper, str, str], Awaitable[Any]]
    """Called with `(ctx, arguments_json, call_id)`; returns whatever the wrapped function does."""
    tool_input_guardrails: list[ToolInputGuardrail[Any]] | None = None
    tool_output_guardrails: list[ToolOutputGuardrail[Any]] | None = None
    needs_approval: _NeedsApproval = False


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


def _schema_from_signature(func: Callable[..., Any]) -> tuple[dict[str, Any], list[str]]:
    """Build a `{"type": "object", "properties": {...}}` schema from `func`'s parameters.

    `ctx`/`call_id` are Runa's reserved parameter names (see `runa.approval`) for the run context
    and the tool call's id; they're never part of the model-facing schema. Returns the schema
    alongside the ordered list of the remaining (model-facing) parameter names.
    """
    signature = inspect.signature(func)
    hints = get_type_hints(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    param_names: list[str] = []
    for param_name, param in signature.parameters.items():
        if param_name in _RESERVED_PARAMS:
            continue
        param_names.append(param_name)
        properties[param_name] = _json_type(hints.get(param_name, param.annotation))
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema, param_names


def _bind_arguments(
    func: Callable[..., Any], args: dict[str, Any], ctx: RunContextWrapper, call_id: str
) -> dict[str, Any]:
    """Build `func`'s call kwargs from the model's parsed `args`, plus any reserved parameters."""
    reserved = {"ctx": ctx, "call_id": call_id}
    bound: dict[str, Any] = {}
    for name in inspect.signature(func).parameters:
        if name in reserved:
            bound[name] = reserved[name]
        elif name in args:
            bound[name] = args[name]
    return bound


@overload
def tool(func: Callable[..., Any]) -> FunctionTool: ...


@overload
def tool(
    func: None = None,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    guardrails: ToolGuardrailsList | ToolGuardrailsDict | None = None,
    needs_approval: _NeedsApproval = False,
) -> Callable[[Callable[..., Any]], FunctionTool]: ...


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    guardrails: ToolGuardrailsList | ToolGuardrailsDict | None = None,
    needs_approval: _NeedsApproval = False,
) -> FunctionTool | Callable[[Callable[..., Any]], FunctionTool]:
    """Wrap a plain function as a `FunctionTool`, deriving its schema from its signature.

    `guardrails=[...]` (or `{...}`) wires `@guardrail` predicates, bare (wired as both sides)
    or bound via `.input`/`.output`, against the tool call's parsed arguments and its return
    value, respectively.
    """
    input_guardrails, output_guardrails = flatten_tool_guardrails(guardrails or [])

    def decorator(fn: Callable[..., Any]) -> FunctionTool:
        schema, _ = _schema_from_signature(fn)

        async def on_invoke_tool(ctx: RunContextWrapper, arguments_json: str, call_id: str) -> Any:
            args = json.loads(arguments_json) if arguments_json else {}
            kwargs = _bind_arguments(fn, args, ctx, call_id)
            if inspect.iscoroutinefunction(fn):
                result = await fn(**kwargs)
            else:
                # Off the event loop: a sync tool that does blocking I/O (a sync HTTP call, a
                # blocking DB driver) would otherwise stall every other concurrent run/tool call.
                result = await asyncio.to_thread(fn, **kwargs)
            return await result if inspect.isawaitable(result) else result

        return FunctionTool(
            name=name_override or fn.__name__,
            description=description_override or (inspect.getdoc(fn) or "").strip(),
            params_json_schema=schema,
            on_invoke_tool=on_invoke_tool,
            tool_input_guardrails=input_guardrails or None,
            tool_output_guardrails=output_guardrails or None,
            needs_approval=needs_approval,
        )

    return decorator(func) if func is not None else decorator


__all__ = ["FunctionTool", "tool"]
