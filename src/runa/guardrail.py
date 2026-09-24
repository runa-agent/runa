"""`@guardrail` decorator that turns a plain predicate into an input/output guardrail."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from runa._types import RunContextWrapper, TResponseInputItem

_Predicate = Callable[[Any], bool | Awaitable[bool]]
_GuardrailFunction = Callable[[Any, Any, Any], Awaitable["GuardrailFunctionOutput"]]
_ToolGuardrailFunction = Callable[[Any], Awaitable["ToolGuardrailFunctionOutput"]]


@dataclass
class GuardrailFunctionOutput:
    """What a guardrail function returns: whatever it wants recorded, plus trip/no-trip."""

    output_info: Any
    tripwire_triggered: bool


@dataclass
class GuardrailResult:
    """A guardrail plus the verdict it returned, whether or not it tripped.

    Every guardrail run this run is recorded (see `RunContextWrapper.input_guardrail_results`
    etc.), not just the one that stopped the run; `tripped` distinguishes the two.
    """

    guardrail: Any
    output: Any
    tripped: bool


class GuardrailResults(TypedDict):
    """A run's guardrail audit trail: every guardrail that ran, by where it ran."""

    input_guardrail_results: list[GuardrailResult]
    output_guardrail_results: list[GuardrailResult]
    tool_input_guardrail_results: list[GuardrailResult]
    tool_output_guardrail_results: list[GuardrailResult]


def guardrail_results(context_wrapper: RunContextWrapper) -> GuardrailResults:
    """A snapshot of the run's four audit lists, as keyword arguments for a result or state."""
    return GuardrailResults(
        input_guardrail_results=list(context_wrapper.input_guardrail_results),
        output_guardrail_results=list(context_wrapper.output_guardrail_results),
        tool_input_guardrail_results=list(context_wrapper.tool_input_guardrail_results),
        tool_output_guardrail_results=list(context_wrapper.tool_output_guardrail_results),
    )


@dataclass
class InputGuardrail[TContext]:
    """Checks an agent's input before the model ever sees it; trips the run if it should stop."""

    guardrail_function: _GuardrailFunction
    name: str | None = None


@dataclass
class OutputGuardrail[TContext]:
    """Checks an agent's final output before a run returns it; trips the run if it should stop."""

    guardrail_function: _GuardrailFunction
    name: str | None = None


@dataclass
class ToolGuardrailFunctionOutput:
    """What a tool guardrail function returns: whatever it wants recorded, plus its verdict."""

    output_info: Any
    behavior: dict[str, Any]

    @classmethod
    def raise_exception(cls, output_info: Any = None) -> ToolGuardrailFunctionOutput:
        """Build a verdict that halts the tool call and raises a tripwire exception."""
        return cls(output_info=output_info, behavior={"type": "raise_exception"})

    @classmethod
    def allow(cls, output_info: Any = None) -> ToolGuardrailFunctionOutput:
        """Build a verdict that lets the tool call proceed."""
        return cls(output_info=output_info, behavior={"type": "allow"})


@dataclass
class ToolInputGuardrailContext:
    """What a tool input/output guardrail's `data.context` carries: the raw call it's checking."""

    tool_arguments: str
    tool_name: str = ""
    call_id: str = ""


@dataclass
class ToolInputGuardrailData:
    """What a tool guardrail function receives: the call's context, and its output once it ran."""

    context: ToolInputGuardrailContext
    output: Any = None


@dataclass
class ToolInputGuardrail[TContext]:
    """Checks a tool call's arguments before the tool runs; trips the call if it should stop."""

    guardrail_function: _ToolGuardrailFunction
    name: str | None = None

    def get_name(self) -> str:
        """Return this guardrail's name, defaulting to its wrapped function's name."""
        return self.name or self.guardrail_function.__name__


@dataclass
class ToolOutputGuardrail[TContext]:
    """Checks a tool's return value after it runs; trips the call if it should stop."""

    guardrail_function: _ToolGuardrailFunction
    name: str | None = None

    def get_name(self) -> str:
        """Return this guardrail's name, defaulting to its wrapped function's name."""
        return self.name or self.guardrail_function.__name__


def _latest_text(value: str | list[TResponseInputItem]) -> str:
    """Reduce a guardrail's raw input (a string, or the running item list) to the latest text."""
    if isinstance(value, str):
        return value
    content = value[-1].get("content", "") if value else ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def _wrap(
    func: _Predicate, *, reduce_input: bool
) -> Callable[..., Awaitable[GuardrailFunctionOutput]]:
    """Wrap a `(value) -> bool` predicate into the SDK's async `(ctx, agent, value)` shape."""

    async def wrapper(ctx: Any, agent: Any, value: Any) -> GuardrailFunctionOutput:
        checked = _latest_text(value) if reduce_input else value
        if inspect.iscoroutinefunction(func):
            result = await func(checked)
        else:
            # Off the event loop: a sync predicate that does blocking I/O (a moderation API
            # call, ...) would otherwise stall every other concurrent run/tool call.
            result = await asyncio.to_thread(func, checked)
        return GuardrailFunctionOutput(output_info=func.__doc__, tripwire_triggered=bool(result))

    return wrapper


def _tool_args(data: ToolInputGuardrailData) -> Any:
    """Parse a tool call's raw JSON arguments into a dict, falling back to the raw string."""
    try:
        return json.loads(data.context.tool_arguments)
    except (TypeError, ValueError):
        return data.context.tool_arguments


def _wrap_tool(
    func: _Predicate, *, on_output: bool
) -> Callable[[Any], Awaitable[ToolGuardrailFunctionOutput]]:
    """Wrap a `(value) -> bool` predicate into a tool guardrail's async `(data)` shape."""

    async def wrapper(data: Any) -> ToolGuardrailFunctionOutput:
        checked = data.output if on_output else _tool_args(data)
        if inspect.iscoroutinefunction(func):
            result = await func(checked)
        else:
            # Off the event loop: a sync predicate that does blocking I/O (a moderation API
            # call, ...) would otherwise stall every other concurrent run/tool call.
            result = await asyncio.to_thread(func, checked)
        return (
            ToolGuardrailFunctionOutput.raise_exception(output_info=func.__doc__)
            if result
            else ToolGuardrailFunctionOutput.allow(output_info=func.__doc__)
        )

    return wrapper


def _tool_input_guardrail(func: _Predicate) -> ToolInputGuardrail[Any]:
    """Bind a predicate as a `ToolInputGuardrail`."""
    return ToolInputGuardrail(
        guardrail_function=_wrap_tool(func, on_output=False), name=func.__name__
    )


def _tool_output_guardrail(func: _Predicate) -> ToolOutputGuardrail[Any]:
    """Bind a predicate as a `ToolOutputGuardrail`."""
    return ToolOutputGuardrail(
        guardrail_function=_wrap_tool(func, on_output=True), name=func.__name__
    )


@dataclass
class _AgentInputGuardrail(InputGuardrail[Any]):
    """An `InputGuardrail` that remembers its raw predicate, for reuse in `@tool(guardrails=)`."""

    predicate: _Predicate = field(kw_only=True)


@dataclass
class _AgentOutputGuardrail(OutputGuardrail[Any]):
    """An `OutputGuardrail` that remembers its raw predicate, for reuse in `@tool(guardrails=)`."""

    predicate: _Predicate = field(kw_only=True)


class Guardrail:
    """A predicate bound to neither side yet; `.input`/`.output` (or `.i`/`.o`) picks which.

    `@guardrail` wraps a plain `(value) -> bool` predicate (tripping the guardrail on `True`)
    into this, the same way `@tool` wraps a plain function into a `FunctionTool`. Read `.input`
    to bind it to the input side, `.output` for the output side; `.i`/`.o` are the exact same
    binding under a shorter name, there is no third spelling. The same bound object works in
    both places it's listed:

    - In an `Agent.guardrails` list, it's an `InputGuardrail`/`OutputGuardrail`: the predicate
      sees the latest user message as plain text (regardless of whether the SDK passed a string
      or the running list of input items) on `.input`, or the agent's final output on `.output`.
    - In a `@tool(guardrails=[...])` list, the same object is reinterpreted as a
      `ToolInputGuardrail`/`ToolOutputGuardrail`: the predicate sees the tool call's arguments
      (parsed from JSON into a dict) on `.input`, or the tool's raw return value on `.output`.

    The predicate's docstring becomes `output_info`. Listed bare (no `.input`/`.output`), it's
    wired as both sides of whichever pair applies.
    """

    def __init__(self, func: _Predicate) -> None:
        """Store the predicate to bind on `.input`/`.output` access."""
        self._func = func

    @property
    def input(self) -> InputGuardrail[Any]:
        """Bind this predicate to the input side."""
        return _AgentInputGuardrail(
            guardrail_function=_wrap(self._func, reduce_input=True),
            name=self._func.__name__,
            predicate=self._func,
        )

    @property
    def output(self) -> OutputGuardrail[Any]:
        """Bind this predicate to the output side."""
        return _AgentOutputGuardrail(
            guardrail_function=_wrap(self._func, reduce_input=False),
            name=self._func.__name__,
            predicate=self._func,
        )

    @property
    def i(self) -> InputGuardrail[Any]:
        """Shorthand for `.input`."""
        return self.input

    @property
    def o(self) -> OutputGuardrail[Any]:
        """Shorthand for `.output`."""
        return self.output


def guardrail(func: _Predicate) -> Guardrail:
    """Turn a `(value) -> bool` predicate into a `Guardrail`; bind it via `.input`/`.output`."""
    return Guardrail(func)


GuardrailsList = list["InputGuardrail[Any] | OutputGuardrail[Any] | Guardrail"]
GuardrailsDict = dict[Literal["input", "output"], GuardrailsList]

ToolGuardrailsList = list["InputGuardrail[Any] | OutputGuardrail[Any] | Guardrail"]
ToolGuardrailsDict = dict[Literal["input", "output"], ToolGuardrailsList]


def _entries(guardrails: Any) -> list[Any]:
    """Normalize a flat list or `{"input": [...], "output": [...]}` dict to a flat entry list.

    A bare `Guardrail` nested in a dict bucket binds to that bucket's side; an already-bound
    entry passes through untouched.
    """
    if not isinstance(guardrails, dict):
        return list(guardrails)
    return [
        getattr(sub, mode) if isinstance(sub, Guardrail) else sub
        for mode, subs in guardrails.items()
        for sub in subs
    ]


def flatten_agent_guardrails(
    guardrails: GuardrailsList | GuardrailsDict,
) -> tuple[list[InputGuardrail[Any]], list[OutputGuardrail[Any]]]:
    """Split an `Agent.guardrails` list/dict into input/output lists; bare entries wire as both."""
    input_guardrails: list[InputGuardrail[Any]] = []
    output_guardrails: list[OutputGuardrail[Any]] = []
    for entry in _entries(guardrails):
        if isinstance(entry, Guardrail):
            input_guardrails.append(entry.input)
            output_guardrails.append(entry.output)
        elif isinstance(entry, _AgentInputGuardrail):
            input_guardrails.append(entry)
        elif isinstance(entry, _AgentOutputGuardrail):
            output_guardrails.append(entry)
        else:
            raise TypeError(
                f"guardrails entries must be @guardrail predicates bound via "
                f".input/.output, got {type(entry).__name__}"
            )
    return input_guardrails, output_guardrails


def flatten_tool_guardrails(
    guardrails: ToolGuardrailsList | ToolGuardrailsDict,
) -> tuple[list[ToolInputGuardrail[Any]], list[ToolOutputGuardrail[Any]]]:
    """Split a `@tool(guardrails=...)` list/dict into input/output lists; bare entries wire as both.

    Accepts a bare `@guardrail` predicate, or the same `.input`/`.output`-bound object used for
    `Agent.guardrails`, reused here against the tool call's arguments/return value instead of
    the agent's input/output.
    """
    input_guardrails: list[ToolInputGuardrail[Any]] = []
    output_guardrails: list[ToolOutputGuardrail[Any]] = []
    for entry in _entries(guardrails):
        if isinstance(entry, Guardrail):
            input_guardrails.append(_tool_input_guardrail(entry._func))
            output_guardrails.append(_tool_output_guardrail(entry._func))
        elif isinstance(entry, _AgentInputGuardrail):
            input_guardrails.append(_tool_input_guardrail(entry.predicate))
        elif isinstance(entry, _AgentOutputGuardrail):
            output_guardrails.append(_tool_output_guardrail(entry.predicate))
        else:
            raise TypeError(
                f"guardrail entries must be @guardrail predicates bound via "
                f".input/.output, got {type(entry).__name__}"
            )
    return input_guardrails, output_guardrails


__all__ = [
    "Guardrail",
    "GuardrailResult",
    "GuardrailResults",
    "GuardrailsDict",
    "GuardrailsList",
    "ToolGuardrailsDict",
    "ToolGuardrailsList",
    "flatten_agent_guardrails",
    "flatten_tool_guardrails",
    "guardrail",
    "guardrail_results",
]
