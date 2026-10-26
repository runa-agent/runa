"""`@guardrail` decorator that turns a plain predicate into an input/output guardrail.

Agent-or-tool by input-or-output is a fact about guardrails, so it lives here as data -- one
`Phase` -- rather than as structure spread across the codebase. One entry type carries its phase,
one verdict type comes back from every phase, one runner (the turn loop's) takes it as
an argument, one exception reports it, and a run's audit trail is one mapping keyed by it. A fifth
phase is a member of `Phase`, not a new field, class and list in nine modules.
"""

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from runa._items import ConversationItem, latest_text
from runa.exceptions import UserError

_Predicate = Callable[[Any], bool | Awaitable[bool]]
_GuardrailFunction = Callable[[Any], Awaitable["GuardrailVerdict"]]


class Phase(StrEnum):
    """Where a guardrail runs: agent-or-tool by input-or-output, the 2x2 named once.

    The phase decides everything about an entry: which list it belongs to, what the one value its
    function is handed means (the agent's own input or final output, a tool call's arguments or
    return value), which key its results are recorded under, and how a tripwire names it.
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
class GuardrailVerdict:
    """What a guardrail function returns: the predicate's docstring, plus trip/no-trip.

    `output_info` is the `@guardrail` predicate's `__doc__` (or `None`), never anything it
    computed: a predicate returns a bool, so what it checks for is all the audit trail can say.

    One shape for all four phases, because `tripped` is the only question anyone asks a verdict.
    Whether the guardrail watched an agent's turn or one tool call is `Phase`'s to say -- the
    wrapper reads `phase` to know what the one value it was handed is, `GuardrailResult.phase`
    records it for the audit trail -- so the verdict doesn't re-encode it as a second type.
    """

    output_info: Any
    tripped: bool


@dataclass
class GuardrailResult:
    """A guardrail plus the verdict it returned and the `Phase` it returned it in.

    Every guardrail run this run is recorded (see `GuardrailResults`), not just the one that
    stopped the run; `tripped` distinguishes the two.
    """

    guardrail: Any
    output: GuardrailVerdict
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

    Mixed into what `Run` and `RunState` share, which exposes the trail as a single
    `guardrail_results`: the four public names are written here once instead of once per holder,
    and a fifth phase adds nothing to either.
    """

    if TYPE_CHECKING:

        @property
        def guardrail_results(self) -> GuardrailResults:
            """The trail, keyed by `Phase`: the holder's own, never a dataclass field here.

            A base class's field sorts ahead of every field the holder declares itself, and a
            read-only declaration lets a holder answer it with a property.
            """
            ...

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
class BoundGuardrail:
    """A predicate bound to one `Phase`: what an `Agent`'s or a `@tool`'s guardrail list holds.

    One entry type for all four cells of the 2x2: `phase` is the entire difference between them.
    `predicate` is the raw `(value) -> bool` that was wrapped, kept so the same bound object can be
    rebound against a tool call when it's listed in `@tool(guardrails=[...])` instead of on an
    agent.
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


def _checked(phase: Phase, value: Any) -> Any:
    """The value `phase`'s predicate actually sees, reduced from the one the runner passed.

    Every guardrail is called with exactly one argument, whatever its phase; what that argument
    is, is this 2x2's only statement. An output phase is handed what came back (the agent's
    final output, the tool's return value) and checks it as-is, and so does a tool's input phase,
    handed the call's arguments already parsed into a dict. The agent's input phase is handed
    the run's input (a string, or the running item list) and checks the latest user text.
    """
    if phase.on_output or phase.on_tool:
        return value
    return _checked_input(value)


def _wrap(func: _Predicate, phase: Phase) -> _GuardrailFunction:
    """Wrap a `(value) -> bool` predicate into the guardrail function the runner calls."""

    async def wrapper(value: Any) -> GuardrailVerdict:
        checked = _checked(phase, value)
        if inspect.iscoroutinefunction(func):
            tripped = await func(checked)
        else:
            # Off the event loop: a sync predicate that does blocking I/O (a moderation API
            # call, ...) would otherwise stall every other concurrent run/tool call.
            tripped = await asyncio.to_thread(func, checked)
        return GuardrailVerdict(output_info=func.__doc__, tripped=bool(tripped))

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


GuardrailsList = Sequence["BoundGuardrail | Guardrail"]


def _entries(guardrails: Any) -> list[Any]:
    """The `guardrails=` declaration as a list, the one shape it takes.

    A dict grouping entries by side is a `UserError` naming the list it should be, rather than
    its keys reaching `flatten_guardrails` as entries.
    """
    if not isinstance(guardrails, list | tuple):
        raise UserError(
            f"guardrails must be a list, e.g. [block_empty.input, contains_pii.output], "
            f"got {type(guardrails).__name__}"
        )
    return list(guardrails)


def _rebound(entry: BoundGuardrail, phase: Phase) -> BoundGuardrail:
    """`entry` as a `phase` entry: itself if it already is one, its predicate rebound if not.

    Rebinding is what lets one `@guardrail` object mean "check the agent" on an agent and "check
    the tool call" on a tool. It needs the raw predicate the entry wrapped, so a hand-built entry
    -- which reads its value the way its own phase does, and carries nothing to re-wrap -- can't
    cross over.
    """
    if entry.phase is phase:
        return entry
    if entry.predicate is None:
        raise UserError(
            f"guardrails entries must be @guardrail predicates bound via .input/.output, "
            f"got one bound to {entry.phase.value!r} with no predicate to rebind"
        )
    return Guardrail(entry.predicate)._bind(phase)


def flatten_guardrails(
    guardrails: GuardrailsList, *, tool: bool = False
) -> dict[Phase, list[BoundGuardrail]]:
    """Sort a `guardrails=` list into entries by `Phase`; a bare entry wires as both sides.

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
            raise UserError(
                f"guardrails entries must be @guardrail predicates bound via "
                f".input/.output, got {type(entry).__name__}"
            )
    return bound


def _entry_name(entry: Any) -> str:
    """One declared entry's name: a bound entry's own, or a bare `Guardrail`'s predicate's."""
    if isinstance(entry, Guardrail):
        return entry._func.__name__
    if isinstance(entry, BoundGuardrail):
        return entry.get_name()
    return str(entry)


def declared_names(guardrails: GuardrailsList) -> list[str]:
    """Name every guardrail a `guardrails=` declaration lists, in the order it lists them.

    What such a declaration may hold -- bound entries or bare `Guardrail`s -- is this module's
    own knowledge, so naming them for display lives here beside `flatten_guardrails` rather than
    in whichever caller (`runa ui`'s Agents page, today) needs it next.
    """
    return [_entry_name(entry) for entry in _entries(guardrails)]


__all__ = [
    "BoundGuardrail",
    "Guardrail",
    "GuardrailAudit",
    "GuardrailResult",
    "GuardrailResults",
    "GuardrailVerdict",
    "GuardrailsList",
    "Phase",
    "declared_names",
    "flatten_guardrails",
    "guardrail",
]
