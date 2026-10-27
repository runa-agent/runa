"""`@guardrail` decorator that turns a plain predicate into an input/output guardrail.

Agent-or-tool by input-or-output is one `Phase`. One entry type carries its phase, one verdict
type comes back from every phase, one runner (the turn loop's) takes it as an argument, one
exception reports it, and a run's audit trail is one mapping keyed by it, so a phase is added by
adding a member to `Phase`.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from runa._calling import call_app
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
    """A predicate wrapped for one `Phase`: what an `Agent`'s or a `@tool`'s resolved lists hold.

    One entry type for all four cells of the 2x2: `phase` is the entire difference between them.
    Built by `flatten_guardrails` from a declared `Guardrail`, never declared itself.
    """

    guardrail_function: _GuardrailFunction
    phase: Phase
    name: str | None = None

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
        tripped = await call_app(func, _checked(phase, value))
        return GuardrailVerdict(output_info=func.__doc__, tripped=bool(tripped))

    return wrapper


class Guardrail:
    """A predicate and the side it guards; `.input`/`.output` (or `.i`/`.o`) picks the side.

    `@guardrail` wraps a plain `(value) -> bool` predicate (tripping the guardrail on `True`)
    into this, the same way `@tool` wraps a plain function into a `FunctionTool`. Read `.input`
    for the input side, `.output` for the output side; `.i`/`.o` are the exact same binding under
    a shorter name, there is no third spelling. Listed bare, it guards both sides. Whether a side
    means the agent or a tool call is the list's to say, so the same object works in both:

    - In an `Agent.guardrails` list, it runs in `Phase.INPUT`/`Phase.OUTPUT`: the predicate sees
      the latest user message as plain text (regardless of whether the run was passed a string or
      the running list of input items) on `.input`, or the agent's final output on `.output`.
    - In a `@tool(guardrails=[...])` list, it runs in `Phase.TOOL_INPUT`/`Phase.TOOL_OUTPUT`: the
      predicate sees the tool call's arguments (parsed from JSON into a dict) on `.input`, or the
      tool's raw return value on `.output`.

    The predicate's docstring becomes `output_info`.
    """

    def __init__(self, func: _Predicate, sides: tuple[bool, ...] = (False, True)) -> None:
        """Store the predicate and the sides it guards, each `True` for the output side."""
        self._func = func
        self._sides = sides

    def bind(self, phases: tuple[Phase, Phase]) -> list[BoundGuardrail]:
        """This predicate wrapped for each of its sides, given the `(input, output)` phase pair."""
        return [
            BoundGuardrail(
                guardrail_function=_wrap(self._func, phases[on_output]),
                phase=phases[on_output],
                name=self._func.__name__,
            )
            for on_output in self._sides
        ]

    @property
    def input(self) -> Guardrail:
        """This predicate on the input side."""
        return Guardrail(self._func, (False,))

    @property
    def output(self) -> Guardrail:
        """This predicate on the output side."""
        return Guardrail(self._func, (True,))

    @property
    def i(self) -> Guardrail:
        """Shorthand for `.input`."""
        return self.input

    @property
    def o(self) -> Guardrail:
        """Shorthand for `.output`."""
        return self.output

    @property
    def name(self) -> str:
        """The predicate's name, which every entry bound from it carries."""
        return self._func.__name__


def guardrail(func: _Predicate) -> Guardrail:
    """Turn a `(value) -> bool` predicate into a `Guardrail`; bind it via `.input`/`.output`."""
    return Guardrail(func)


GuardrailsList = Sequence[Guardrail]


def _entries(guardrails: Any) -> list[Guardrail]:
    """The `guardrails=` declaration as a list of `Guardrail`s, the one shape it takes.

    A dict grouping entries by side is a `UserError` naming the list it should be, rather than
    its keys reaching `flatten_guardrails` as entries, and so is an entry `@guardrail` didn't make.
    """
    if not isinstance(guardrails, list | tuple):
        raise UserError(
            f"guardrails must be a list, e.g. [block_empty.input, contains_pii.output], "
            f"got {type(guardrails).__name__}"
        )
    for entry in guardrails:
        if not isinstance(entry, Guardrail):
            raise UserError(
                f"guardrails entries must be @guardrail predicates, bare or bound via "
                f".input/.output, got {type(entry).__name__}"
            )
    return list(guardrails)


def flatten_guardrails(
    guardrails: GuardrailsList, *, tool: bool = False
) -> dict[Phase, list[BoundGuardrail]]:
    """Wrap a `guardrails=` list into entries by `Phase`, in declared order within each.

    One function for both lists Runa accepts -- an `Agent`'s, checked against its input and its
    final output, and a `@tool`'s, checked against the call's arguments and its return value --
    because they differ only in which pair of phases they bind to, which is what `tool` picks.
    """
    phases = (Phase.TOOL_INPUT, Phase.TOOL_OUTPUT) if tool else (Phase.INPUT, Phase.OUTPUT)
    bound: dict[Phase, list[BoundGuardrail]] = {phase: [] for phase in phases}
    for entry in _entries(guardrails):
        for wrapped in entry.bind(phases):
            bound[wrapped.phase].append(wrapped)
    return bound


def declared_names(guardrails: GuardrailsList) -> list[str]:
    """Name every guardrail a `guardrails=` declaration lists, in the order it lists them.

    What such a declaration may hold is this module's own knowledge, so naming them for display
    lives here beside `flatten_guardrails` rather than in whichever caller (`runa ui`'s Agents
    page, today) needs it next.
    """
    return [entry.name for entry in _entries(guardrails)]


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
