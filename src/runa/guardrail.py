"""`@guardrail` decorator that turns a plain predicate into an input/output guardrail.

Agent-or-tool by input-or-output is a fact about guardrails, so it lives here as data -- one
`Phase` -- rather than as structure spread across the codebase. One entry type carries its phase,
one runner (`run_internal.guardrails`) takes it as an argument, one exception reports it, and a
run's audit trail is one mapping keyed by it. A fifth phase is a member of `Phase`, not a new
field, class and list in nine modules.
"""

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from runa._items import ConversationItem, latest_text, parsed_arguments

_Predicate = Callable[[Any], bool | Awaitable[bool]]
_GuardrailFunction = Callable[..., Awaitable[Any]]


class Phase(StrEnum):
    """Where a guardrail runs: agent-or-tool by input-or-output, the 2x2 named once.

    The phase decides everything the four used to spell out separately: which list an entry
    belongs to, which call shape its function takes (an agent's `(context, agent, value)`, a
    tool's `(data)`), which key its results are recorded under, and how a tripwire names it.
    """

    INPUT = "input"
    OUTPUT = "output"
    TOOL_INPUT = "tool_input"
    TOOL_OUTPUT = "tool_output"

    @property
    def on_tool(self) -> bool:
        """Whether this phase guards one tool call rather than the agent's own turn."""
        return self in (Phase.TOOL_INPUT, Phase.TOOL_OUTPUT)

    @property
    def on_output(self) -> bool:
        """Whether this phase checks what came back rather than what is about to go in."""
        return self in (Phase.OUTPUT, Phase.TOOL_OUTPUT)

    @property
    def title(self) -> str:
        """How a tripwire message names this phase, e.g. `"Tool input guardrail"`."""
        return f"{self.value.replace('_', ' ')} guardrail".capitalize()


@dataclass
class GuardrailFunctionOutput:
    """What an agent guardrail function returns: whatever it wants recorded, plus trip/no-trip."""

    output_info: Any
    tripwire_triggered: bool

    @property
    def tripped(self) -> bool:
        """Whether this verdict stops the run; the one question the runner asks a verdict."""
        return self.tripwire_triggered


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

    @property
    def tripped(self) -> bool:
        """Whether this verdict stops the tool call; `behavior` is the only place that says so."""
        return self.behavior["type"] == "raise_exception"


@dataclass
class GuardrailResult:
    """A guardrail plus the verdict it returned and the `Phase` it returned it in.

    Every guardrail run this run is recorded (see `GuardrailResults`), not just the one that
    stopped the run; `tripped` distinguishes the two.
    """

    guardrail: Any
    output: Any
    tripped: bool
    phase: Phase


@dataclass
class GuardrailResults:
    """A run's guardrail audit trail: every `GuardrailResult` it produced, keyed by `Phase`.

    One mapping rather than a list per phase, so the four names RUNA.md documents are written
    once, on `GuardrailAudit`, and a phase nothing ran in simply has no key.
    """

    by_phase: dict[Phase, list[GuardrailResult]] = field(default_factory=dict)

    def record(self, result: GuardrailResult) -> None:
        """Append `result` under its own phase, starting that phase's list on first use."""
        self.by_phase.setdefault(result.phase, []).append(result)

    def __getitem__(self, phase: Phase) -> list[GuardrailResult]:
        """Everything that ran in `phase`, an empty list if nothing did."""
        return self.by_phase.get(phase, [])

    def snapshot(self) -> GuardrailResults:
        """A copy, so a finished `Run` isn't still growing as a shared context records more."""
        return GuardrailResults({phase: list(ran) for phase, ran in self.by_phase.items()})


class GuardrailAudit:
    """The four `*_guardrail_results` names RUNA.md documents, over one phase-keyed mapping.

    Mixed into `Run`, `RunState` and `RunErrorDetails`, each of which holds the trail as a single
    `guardrail_results` field: the four public names are written here once instead of once per
    holder, and a fifth phase adds nothing to any of the three.
    """

    if TYPE_CHECKING:
        # Declared for the properties to read, not contributed as a dataclass field: a base
        # class's field sorts ahead of every field the holder declares itself.
        guardrail_results: GuardrailResults

    @property
    def input_guardrail_results(self) -> list[GuardrailResult]:
        """Every agent input guardrail that ran, tripped or not."""
        return self.guardrail_results[Phase.INPUT]

    @property
    def output_guardrail_results(self) -> list[GuardrailResult]:
        """Every agent output guardrail that ran, tripped or not."""
        return self.guardrail_results[Phase.OUTPUT]

    @property
    def tool_input_guardrail_results(self) -> list[GuardrailResult]:
        """Every tool input guardrail that ran, tripped or not."""
        return self.guardrail_results[Phase.TOOL_INPUT]

    @property
    def tool_output_guardrail_results(self) -> list[GuardrailResult]:
        """Every tool output guardrail that ran, tripped or not."""
        return self.guardrail_results[Phase.TOOL_OUTPUT]


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

    @classmethod
    def of(cls, tool_name: str, args_json: str, call_id: str) -> ToolInputGuardrailData:
        """Build what a call's guardrails see: which tool, the call, its arguments.

        The only place a `ToolInputGuardrailContext` is constructed, so a field it carries can't
        be populated for the input side and forgotten for the output side.
        """
        return cls(
            context=ToolInputGuardrailContext(
                tool_arguments=args_json, tool_name=tool_name, call_id=call_id
            )
        )

    def returning(self, output: Any) -> ToolInputGuardrailData:
        """The same call as the output side sees it: this context, plus what the tool returned."""
        return ToolInputGuardrailData(context=self.context, output=output)


@dataclass
class BoundGuardrail:
    """A predicate bound to one `Phase`: what an `Agent`'s or a `@tool`'s guardrail list holds.

    One entry type for all four cells of the 2x2, which used to be four dataclasses (six, with
    the two that added nothing but `predicate`) carrying the same two fields and the same
    `get_name()`. `phase` is the entire difference between them. `predicate` is the raw
    `(value) -> bool` that was wrapped, kept so the same bound object can be rebound against a
    tool call when it's listed in `@tool(guardrails=[...])` instead of on an agent.
    """

    guardrail_function: _GuardrailFunction
    phase: Phase
    name: str | None = None
    predicate: _Predicate | None = None

    def get_name(self) -> str:
        """This guardrail's name: the one `@guardrail` set, or its wrapped function's."""
        return self.name or getattr(self.guardrail_function, "__name__", "guardrail")


def _checked_input(value: str | list[ConversationItem]) -> str:
    """Reduce a guardrail's raw input (a string, or the running item list) to the latest text."""
    return value if isinstance(value, str) else latest_text(value)


def _tool_args(data: ToolInputGuardrailData) -> Any:
    """Parse a tool call's raw JSON arguments into a dict, falling back to the raw string."""
    try:
        return parsed_arguments(data.context.tool_arguments)
    except ValueError:
        return data.context.tool_arguments


def _checked(phase: Phase, args: tuple[Any, ...]) -> Any:
    """The value `phase`'s predicate actually sees, dug out of its call shape's arguments.

    An agent guardrail is called `(context, agent, value)` and a tool's `(data)`; an input phase
    checks what is going in (the latest user text, the call's parsed arguments), an output phase
    what came back (the agent's final output, the tool's return value).
    """
    if not phase.on_tool:
        value = args[-1]
        return value if phase.on_output else _checked_input(value)
    data = args[0]
    return data.output if phase.on_output else _tool_args(data)


def _verdict(phase: Phase, tripped: bool, output_info: Any) -> Any:
    """Spell a predicate's `bool` the way `phase`'s caller expects to read it back."""
    if not phase.on_tool:
        return GuardrailFunctionOutput(output_info=output_info, tripwire_triggered=tripped)
    if tripped:
        return ToolGuardrailFunctionOutput.raise_exception(output_info=output_info)
    return ToolGuardrailFunctionOutput.allow(output_info=output_info)


def _wrap(func: _Predicate, phase: Phase) -> _GuardrailFunction:
    """Wrap a `(value) -> bool` predicate into the call shape `phase` is invoked with."""

    async def wrapper(*args: Any) -> Any:
        checked = _checked(phase, args)
        if inspect.iscoroutinefunction(func):
            tripped = await func(checked)
        else:
            # Off the event loop: a sync predicate that does blocking I/O (a moderation API
            # call, ...) would otherwise stall every other concurrent run/tool call.
            tripped = await asyncio.to_thread(func, checked)
        return _verdict(phase, bool(tripped), func.__doc__)

    return wrapper


class Guardrail:
    """A predicate bound to neither side yet; `.input`/`.output` (or `.i`/`.o`) picks which.

    `@guardrail` wraps a plain `(value) -> bool` predicate (tripping the guardrail on `True`)
    into this, the same way `@tool` wraps a plain function into a `FunctionTool`. Read `.input`
    to bind it to the input side, `.output` for the output side; `.i`/`.o` are the exact same
    binding under a shorter name, there is no third spelling. The same bound object works in
    both places it's listed:

    - In an `Agent.guardrails` list, it's bound to `Phase.INPUT`/`Phase.OUTPUT`: the predicate
      sees the latest user message as plain text (regardless of whether the run was passed a
      string or the running list of input items) on `.input`, or the agent's final output on
      `.output`.
    - In a `@tool(guardrails=[...])` list, the same object is rebound to `Phase.TOOL_INPUT`/
      `Phase.TOOL_OUTPUT`: the predicate sees the tool call's arguments (parsed from JSON into a
      dict) on `.input`, or the tool's raw return value on `.output`.

    The predicate's docstring becomes `output_info`. Listed bare (no `.input`/`.output`), it's
    wired as both sides of whichever pair applies.
    """

    def __init__(self, func: _Predicate) -> None:
        """Store the predicate to bind on `.input`/`.output` access."""
        self._func = func

    def _bind(self, phase: Phase) -> BoundGuardrail:
        """This predicate as a `phase` entry, wrapped into the call shape that phase is given."""
        return BoundGuardrail(
            guardrail_function=_wrap(self._func, phase),
            phase=phase,
            name=self._func.__name__,
            predicate=self._func,
        )

    @property
    def input(self) -> BoundGuardrail:
        """Bind this predicate to the input side."""
        return self._bind(Phase.INPUT)

    @property
    def output(self) -> BoundGuardrail:
        """Bind this predicate to the output side."""
        return self._bind(Phase.OUTPUT)

    @property
    def i(self) -> BoundGuardrail:
        """Shorthand for `.input`."""
        return self.input

    @property
    def o(self) -> BoundGuardrail:
        """Shorthand for `.output`."""
        return self.output


def guardrail(func: _Predicate) -> Guardrail:
    """Turn a `(value) -> bool` predicate into a `Guardrail`; bind it via `.input`/`.output`."""
    return Guardrail(func)


GuardrailsList = list["BoundGuardrail | Guardrail"]
GuardrailsDict = dict[Literal["input", "output"], GuardrailsList]


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


def _rebound(entry: BoundGuardrail, phase: Phase) -> BoundGuardrail:
    """`entry` as a `phase` entry: itself if it already is one, its predicate rebound if not.

    Rebinding is what lets one `@guardrail` object mean "check the agent" on an agent and "check
    the tool call" on a tool. It needs the raw predicate the entry wrapped, so a hand-built entry
    -- which carries a function in one call shape and nothing to re-wrap -- can't cross over.
    """
    if entry.phase is phase:
        return entry
    if entry.predicate is None:
        raise TypeError(
            f"guardrails entries must be @guardrail predicates bound via .input/.output, "
            f"got one bound to {entry.phase.value!r} with no predicate to rebind"
        )
    return Guardrail(entry.predicate)._bind(phase)


def flatten_guardrails(
    guardrails: GuardrailsList | GuardrailsDict, *, tool: bool = False
) -> dict[Phase, list[BoundGuardrail]]:
    """Sort a `guardrails=` list/dict into entries by `Phase`; a bare entry wires as both sides.

    One function for both lists Runa accepts -- an `Agent`'s, checked against its input and its
    final output, and a `@tool`'s, checked against the call's arguments and its return value --
    because they differ only in which pair of phases they bind to, which is what `tool` picks.
    """
    sides = (Phase.TOOL_INPUT, Phase.TOOL_OUTPUT) if tool else (Phase.INPUT, Phase.OUTPUT)
    bound: dict[Phase, list[BoundGuardrail]] = {phase: [] for phase in sides}
    for entry in _entries(guardrails):
        if isinstance(entry, Guardrail):
            for phase in sides:
                bound[phase].append(entry._bind(phase))
        elif isinstance(entry, BoundGuardrail):
            phase = sides[1] if entry.phase.on_output else sides[0]
            bound[phase].append(_rebound(entry, phase))
        else:
            raise TypeError(
                f"guardrails entries must be @guardrail predicates bound via "
                f".input/.output, got {type(entry).__name__}"
            )
    return bound


__all__ = [
    "BoundGuardrail",
    "Guardrail",
    "GuardrailAudit",
    "GuardrailResult",
    "GuardrailResults",
    "GuardrailsDict",
    "GuardrailsList",
    "Phase",
    "flatten_guardrails",
    "guardrail",
]
