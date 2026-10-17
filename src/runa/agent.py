"""Class-based Agent, built on Runa's own runtime (`runa.run_internal`).

`run`/`run_sync`/`run_streamed` are the only way to run an Agent, and `Run` the only thing they
return. The turn loop under them is this class's implementation, not a second entry point, so
every run gets an Agent's own wiring: its guardrail flattening, its memory and knowledge
resolution, its `RunConfig`, its refusal of two concurrent session-less runs.
"""

import asyncio
import copy
import difflib
import inspect
import re
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from runa import _loop, content
from runa._items import ConversationItem, user_message
from runa._log import logger
from runa._models import DEFAULT_MODEL, Model, check_model_name
from runa._types import MessageContent, ModelSettings, RunContext, Usage
from runa.compact import Compactor, default_compactor
from runa.exceptions import UserError
from runa.guardrail import BoundGuardrail, GuardrailsList, Phase, flatten_guardrails
from runa.handoff import Delegate, Mode
from runa.knowledge import Knowledge, KnowledgeLike
from runa.lifecycle import Hooks
from runa.memory import Memory, MemoryLike
from runa.run import Run, RunStream
from runa.run_internal.run_config import DEFAULT_MAX_TURNS, RunConfig
from runa.run_internal.run_loop import _run_async
from runa.run_state import RunState
from runa.session import Session
from runa.stream_events import StreamEvent
from runa.tool import FunctionTool
from runa.tracing.config import flush_exports

if TYPE_CHECKING:
    from graphviz import Digraph

    from runa.eval.case import Case
    from runa.eval.report import Report

_UNSET: Any = object()
"""`Agent.__init__`'s "this setting wasn't passed", which is not the same as being passed `None`.

Four settings take `None` as a value rather than as silence: `instructions=None` runs an agent on
none instead of loading `app/prompts/<name>.md`, and `memory=None`/`knowledge=None`/
`output_type=None` turn off what the subclass declared. A default of `None` would make
`SupportAgent(memory=None)` indistinguishable from not overriding `memory` at all, so the default
is this sentinel instead -- typed `Any` so every parameter keeps its own annotation while sharing
the one default.
"""


def _adapt_instructions(instructions: Any) -> Any:
    """Let `instructions` be a `(context) -> str` callable instead of the runner's 2-arg shape.

    `run_internal.run_loop` calls `instructions(run_context, agent)` when it's callable, mirroring
    the two-parameter shape a handful of tests rely on. A single-parameter callable is wrapped so it
    receives just `run_context.context`, the object passed to `Agent.run`/`run_sync`; anything
    else (a string, `None`, or an already two-parameter callable) passes through unchanged.
    """
    if not callable(instructions):
        return instructions
    params = list(inspect.signature(instructions).parameters.values())
    if len(params) != 1:
        return instructions

    def _resolved(run_context: RunContext[Any], _agent: Agent) -> Any:
        return instructions(run_context.context)

    return _resolved


def _resolve_session(session: Session | str | None) -> Session | None:
    """Turn a `session_id` into this deployment's session for it, passing an object through.

    A conversation id is all an app actually knows: which backend holds it is
    `RUNA_DATABASE_URL`'s answer, asked here through `runa.db` so that `session="user-42"` keeps
    following the variable the way traces, memory and the cache already do. Naming a backend at
    the call site (`SQLiteSession(...)`) stays available for pointing at a specific file, and a
    custom `Session` is passed straight through.

    Imported inside the function because `runa.db` resolves lazily by design; see its docstring.
    """
    if isinstance(session, str):
        from runa import db

        return db.session(session)
    return session


def _turn_input(
    message: MessageContent | RunState,
    history: list[ConversationItem],
    session: Session | None,
    context: Any = None,
) -> str | list[ConversationItem] | RunState:
    """Build the `input` for the turn loop from this turn's `message`.

    A paused `RunState` passes straight through, to be resumed, carrying the `context` of the
    run it paused: a `context=` passed alongside one would be dropped, so it is refused here
    instead. A list `message` goes through
    `runa.content.parts` first, classifying each item as text or an image (and rejecting a list
    of past messages, which belongs in `history`/`session`); a plain string is left untouched.
    With no `session`, the result joins `history` as a new user message. With a `session`, only
    the new turn is ever sent (prior turns come back from the session itself): a plain string
    passes straight through, a multimodal one is wrapped in a single-item message list instead,
    since the loop's session path only wraps a bare string into a user message itself.
    """
    if isinstance(message, RunState):
        if context is not None:
            raise UserError(
                "a RunState already carries the context of the run it paused, so `context=` "
                "would be ignored when resuming it: pass `context=` to the first run instead."
            )
        return message
    resolved = message if isinstance(message, str) else content.parts(message)
    if session is not None:
        return resolved if isinstance(resolved, str) else [user_message(resolved)]
    return [*history, user_message(resolved)]


_CAMEL_CASE_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def snake_case(name: str) -> str:
    """`SupportAgent` -> `support_agent`: the casing every name in a generated app follows.

    An agent's declared `name` is this applied to its class name, and the file names derived
    from it -- its module, its prompt, its eval dataset -- follow from that, so `runa generate`
    casts tool and guardrail names with the same rule rather than one of its own.
    """
    return _CAMEL_CASE_BOUNDARY.sub("_", name).lower()


def declared_name(agent_cls: type[Agent]) -> str:
    """The `name` `agent_cls` declares, falling back to the class's own name if it declares none.

    For the surfaces that read an `Agent` subclass without building one -- `runa ui`'s Agents
    page, `runa serve`'s route table, `runa chat`'s lookup by name -- where `__init__`'s guarantee
    that every agent has a `name` hasn't been made yet. One function rather than one `getattr`
    default per caller, because an agent answering to two different names depending on which
    surface is asking is the confusion this convention exists to prevent.

    An agent that never declares one is only ever *displayed* this way: constructing it still
    raises `TypeError`. That it doesn't vanish from a listing (or stop the page rendering) is the
    point -- a half-written agent is exactly what a developer is looking at the page to find.
    """
    return getattr(agent_cls, "name", agent_cls.__name__)


def prompt_path(app_dir: Path, name: str) -> Path:
    """Where the agent called `name` keeps its instructions: `<app_dir>/prompts/<name>.md`.

    The prompt-file convention itself, in the module that owns an agent's identity, so the two
    sides that must agree on it read it from here instead of each deriving it: `_load_prompt`
    below, which reads the file, and `runa generate prompt`, which writes it.
    """
    return app_dir / "prompts" / f"{snake_case(name)}.md"


_NO_SOURCE_FILE = (TypeError, OSError)


def _load_prompt(cls: type, name: str) -> str | None:
    """Read `<name>.md` from the `prompts/` directory next to `cls`'s `app/agents/` module.

    The path is `prompt_path`'s, the same call `runa generate prompt` writes through, resolved
    against the parent of the `agents/` directory the subclass is defined in. Reading only:
    constructing an Agent never writes to the project tree, so an immutable image or a read-only
    `app/` is an ordinary way to ship one. Writing that stub is `runa generate
    agent`/`runa generate prompt`'s job.

    Returns `None` (leaving `instructions` empty) when `cls` has no source file (e.g. defined at
    a REPL), its module doesn't live in an `agents/` directory, or no prompt file is there --
    logging the path `runa generate prompt` would write in that last case, so an agent running
    on empty instructions says so instead of being a silent surprise.
    """
    try:
        module_file = Path(inspect.getfile(cls)).resolve()
    except _NO_SOURCE_FILE:
        return None
    if module_file.parent.name != "agents":
        return None
    prompt_file = prompt_path(module_file.parent.parent, name)
    if not prompt_file.is_file():
        logger.warning(
            "%s has no instructions: write %s, or run `runa generate prompt %s`",
            name,
            prompt_file,
            name,
        )
        return None
    return prompt_file.read_text().strip()


SubagentsList = list["type[Agent] | Subagent"]


def _subagents(subagents: SubagentsList) -> SubagentsList:
    """The `subagents` declaration as a list: each entry bare (auto) or `.handoff`/`.delegate`.

    A list is the one shape, so a dict -- grouping entries by mode -- is a `UserError` naming
    the list it should be rather than its keys being built as agents.
    """
    if not isinstance(subagents, list | tuple):
        raise UserError(
            f"subagents must be a list, e.g. [Billing.handoff, Researcher.delegate, Helper], "
            f"got {type(subagents).__name__}"
        )
    return list(subagents)


_under_construction: ContextVar[dict[type[Agent], Agent] | None] = ContextVar(
    "_under_construction", default=None
)


@contextmanager
def _agent_graph(agent: Agent) -> Iterator[Callable[[type[Agent]], Agent]]:
    """Yield the one way `agent`'s `__init__` builds a subagent: each class once per graph.

    Building an agent builds its subagents, and theirs, so without this a class listed twice
    below one root is built twice (per request, under `runa serve`), and two agents listing each
    other -- a specialist handing back to triage -- recurse until `RecursionError`. Instead,
    every agent built while constructing one root is recorded by class, and a class already in
    the graph is the instance already there: a cycle closes onto the agent still being built,
    the same identity `runa.handoff.reachable` already walks a cycle by.

    Construction stays eager, so a broken subagent declaration still fails when the root is
    built, not on whichever turn first reaches it.
    """
    built = _under_construction.get()
    token = _under_construction.set(built := {}) if built is None else None
    built[type(agent)] = agent
    try:
        yield lambda cls: built[cls] if cls in built else cls()
    finally:
        if token is not None:
            _under_construction.reset(token)


def _resolve_compactor(compact: Any) -> Compactor | None:
    """Resolve a `compact=` setting to the `Compactor` the turn loop should run, or `None` for off.

    Resolved here, alongside `memory=`/`knowledge=`, for the reason they are: a declared attribute
    becomes a usable object once, at construction, so a value that is neither a bool nor a
    callable is a `UserError` naming the attribute rather than a `TypeError` raised mid-run, after a
    model call has already been paid for.
    """
    if compact is None or compact is False:
        return None
    if compact is True:
        return default_compactor
    if callable(compact):
        return cast(Compactor, compact)
    raise UserError(
        f"compact must be True, False, a (items, usage_tokens) -> items|None callable, "
        f"or None, got {compact!r}"
    )


def _checked_model(model: Any) -> Any:
    """Check `model=` names a model Runa can route, or is one, without building it yet.

    The exception to the rule the others follow: a declared attribute becomes a usable object at
    construction (see `_resolve_compactor`), but a model name cannot, because resolving it builds
    the provider's client and that is a `UserError` when the matching API key is unset -- which an
    agent that `runa generate` just wrote, or `graph` renders, or no one ever runs has no business
    needing. Resolution therefore waits for the run.

    Checking the name needs no client, though, so both ways of getting it wrong are a `UserError`
    naming the attribute here rather than a failure from inside the turn loop with an `llm` span
    already open: a value that is neither a name nor a `Model`, and a name whose prefix picks no
    provider. The second is the same mistake `_reject_unknown_settings` rejects one field over --
    `model = "claud-sonnet-5"` is a worse `modell = "claude-sonnet-5"`, and gets the same answer
    rather than the opposite one. `Model` is checked structurally, the way `run_internal` reads
    every backend, and is the shape an endpoint Runa doesn't name is reached through.
    """
    if isinstance(model, str):
        check_model_name(model)
    elif model is not None and not hasattr(model, "get_response"):
        raise UserError(
            f"model must be a model name, a Model instance, or None for Runa's default, "
            f"got {model!r}"
        )
    return model


def _is_code(value: Any) -> bool:
    """Is this class-body entry code (a method, a property, a nested class) rather than config?

    Config is data -- a string, a list, a number, a `ModelSettings`. Everything a subclass
    legitimately *writes* is callable or a descriptor, which is what keeps
    `_reject_unknown_settings` from standing between a subclass and its own helper methods.
    """
    return callable(value) or isinstance(value, staticmethod | classmethod | property)


def _reject_unknown_settings(cls: type[Agent]) -> None:
    """Refuse a class attribute of `cls` that isn't one of `_AGENT_FIELDS`.

    Class attributes are the whole configuration surface, so the one mistake that style invites
    is a misspelled name -- and a silently ignored `modell = "claude-sonnet-5"` doesn't fail, it
    runs the default model at a different price with different behaviour and surfaces days later
    as "the agent answers oddly".

    The constructor override makes the same promise `MCPServer`'s named parameters do, and makes
    it the same way: every setting is a named parameter of `__init__`, so a misspelled `modell=`
    is a `TypeError` from the signature itself (and a type-checker error before the code is ever
    run). This function is for the declaration, where Python offers no such check.

    Only data is checked: a name starting with `_`, a method, a property or a nested class is the
    subclass's own business. Mixins in the MRO are skipped for the same reason -- their attributes
    aren't claiming to be Runa settings.
    """
    for klass in cls.__mro__:
        if klass is Agent:
            break
        if not issubclass(klass, Agent):
            continue
        for name, value in vars(klass).items():
            if name.startswith("_") or name in _AGENT_FIELDS or _is_code(value):
                continue
            raise UserError(
                f"{klass.__name__} declares {name!r}, which is not an Agent setting, so it "
                f"would configure nothing.{_did_you_mean(name)} Prefix it with '_' if it is "
                f"the agent's own state rather than Runa config."
            )


def _did_you_mean(name: str) -> str:
    """Point a rejected name at the setting it most likely meant to be, or list them all."""
    close = difflib.get_close_matches(name, _AGENT_FIELDS, n=1)
    if close:
        return f" Did you mean {close[0]!r}?"
    return f" Agent settings are: {', '.join(sorted(_AGENT_FIELDS))}."


_SIGNALLED_WHEN_ABSENT = ("name", "instructions")
"""The two settings whose caller tells "never given" apart from "given as `None`"."""


def _settings(cls: type[Agent], passed: dict[str, Any]) -> dict[str, Any]:
    """Every setting resolved: what the constructor was passed, else what `cls` declares.

    `passed` is `__init__`'s own `locals()`, which holds exactly one entry per parameter and so
    *is* the signature -- the names resolved here are read off it (`_AGENT_FIELDS`) rather than
    listed a second time, so a setting added to the signature is resolved with no edit here. The
    override wins, which is the whole rule, `_UNSET` rather than `None` standing for "not passed"
    so that an override may pass `None` and mean it.

    A setting neither supplies is `None`, except for the two where never being given is itself
    the signal and `__init__` acts on `_UNSET`: a missing `name` is an error, and missing
    `instructions` means "load `app/prompts/<name>.md`" where `instructions=None` means it has
    none.
    """
    resolved: dict[str, Any] = {}
    for name in _AGENT_FIELDS:
        absent = _UNSET if name in _SIGNALLED_WHEN_ABSENT else None
        value = passed[name]
        resolved[name] = value if value is not _UNSET else getattr(cls, name, absent)
    return resolved


def _resolve_retrieval_setting(
    setting: Any, cls: type[Memory] | type[Knowledge], tools: list[FunctionTool]
) -> Any:
    """Resolve a `memory=`/`knowledge=` setting to the instance (or `None`) `Agent` should store.

    Shared by both, since they follow the identical four-shape contract documented on
    `Agent.__init__`: `None` passes through, `"auto"` builds a default instance, `"llm"` appends
    a search tool and leaves the attribute `None`. Anything else -- a `Memory`/`Knowledge`, or any
    other object shaped like `MemoryLike`/`KnowledgeLike` -- passes through untouched too, since
    `run_internal.run_loop` only ever calls its methods, never checks its type: this is the escape
    hatch for a wholesale custom `memory=`/`knowledge=` object. Only an unrecognized *string* is
    rejected, so a typo fails clearly instead of being silently treated as a custom object.
    """
    name = cls.__name__.lower()
    if setting is None or not isinstance(setting, str):
        return setting
    if setting == "auto":
        return cls()
    if setting == "llm":
        tools.append(cls()._as_tool())
        return None
    raise UserError(
        f"{name} must be 'auto', 'llm', a {cls.__name__}(...)-shaped object, or None, "
        f"got {setting!r}"
    )


class _Mode:
    """Descriptor behind `Agent.handoff`/`.delegate`; `.h`/`.d` alias the same instances.

    Accessed on a subclass (`SomeAgent.handoff`), builds a `Subagent` bound to that mode.
    """

    def __init__(self, mode: Mode) -> None:
        self.mode: Mode = mode

    def __get__(self, instance: object, owner: type[Agent]) -> Subagent:
        return Subagent(owner, self.mode)


class Agent:
    """An Agent whose config comes from class attributes instead of __init__ args.

    `usage` accumulates token usage across every `run`/`run_sync`/`run_streamed` call made on
    this instance; `last_usage` holds just the most recent call's usage. Both are read from
    `RunContext.usage`, which the turn loop populates regardless of `hooks`.

    One run is bounded three ways, each `None`/unset meaning "no ceiling of that kind":
    `max_turns` (model calls, default 10), `max_tokens` (total tokens the run may spend), and
    `timeout` (wall-clock seconds). Hitting any of them ends the run as `Run(status="error")`,
    the same as every other `RunaError`. `max_tokens` is the run's whole budget and is not the
    same knob as `ModelSettings(max_tokens=...)`, which caps one response's length.

    An instance carries per-run state (`history`, `usage`), so it is not safe to share across
    concurrent runs; see `run` for the rule and the two ways to satisfy it.
    """

    handoff = _Mode("handoff")
    h = handoff
    delegate = _Mode("delegate")
    d = delegate
    model = DEFAULT_MODEL
    max_turns = DEFAULT_MAX_TURNS
    max_tokens: int | None = None
    timeout: float | None = None

    def __init__(
        self,
        *,
        name: str = _UNSET,
        instructions: str | Callable[..., Any] | None = _UNSET,
        model: str | Model = _UNSET,
        model_settings: ModelSettings | None = _UNSET,
        tools: list[FunctionTool] | None = _UNSET,
        subagents: SubagentsList | None = _UNSET,
        guardrails: GuardrailsList | None = _UNSET,
        mcp: list[Any] | None = _UNSET,
        output_type: type | None = _UNSET,
        hooks: Hooks[Any] | None = _UNSET,
        memory: MemoryLike | Literal["auto", "llm"] | None = _UNSET,
        knowledge: KnowledgeLike | Literal["auto", "llm"] | None = _UNSET,
        compact: bool | Compactor | None = _UNSET,
        max_turns: int = _UNSET,
        max_tokens: int | None = _UNSET,
        timeout: float | None = _UNSET,
    ) -> None:
        """Build config from class attributes and wire up any subagents and guardrails.

        These parameters *are* the closed set of settings (`_AGENT_FIELDS` is read off this
        signature): each one can be declared on the subclass or passed here, the argument
        winning, and an unset one falls back to the class attribute of the same name -- in one
        pass over that tuple (`_settings`), so adding one here resolves it. A name
        that is neither is a `TypeError` from this signature, and a `UserError` naming the
        setting it probably meant when it is declared in a class body instead (see
        `_reject_unknown_settings`).

        `mcp` lists the agent's MCP servers; `self.mcp_servers` is what it resolved to.

        `memory` opts this agent into long-term memory, one of:
          - `"auto"` (or a `Memory(...)` instance, or any object shaped like `runa.memory`'s
            `MemoryLike`): the turn loop retrieves relevant memories before each run and
            persists new ones after -- no manual `memory.search`/`.remember` calls.
          - `"llm"`: the model gets a `search_memory` tool and decides itself when to call it;
            no automatic retrieval/persistence.
          - `None` (the default): the agent behaves exactly as if `runa.memory` didn't exist.
        `"llm"` always uses a default `Memory()`; pass your own instance for `"auto"` mode if you
        need a non-default `db_path`/`model`/`store` -- or a wholesale custom `MemoryLike` object
        to replace embeddings-based retrieval entirely, not just its storage backend.

        `knowledge` opts this agent into retrieval from application/domain documents, one of
        `"auto"`, `"llm"`, a `Knowledge(...)` (or `KnowledgeLike`-shaped) instance, or `None` (the
        default) -- the same four shapes as `memory`, with the same meaning: `"auto"`/an instance
        searches automatically before every turn (no `tools=[...]` wiring needed); `"llm"` gives
        the model a `search_knowledge` tool it calls itself; `None` leaves the agent unaffected.
        `"llm"` always uses a default `Knowledge()`; pass your own instance for `"auto"` mode if
        you need a non-default `directory`/`db_path`/`model`/`store`, or a custom `KnowledgeLike`
        object for a retrieval pipeline of your own.

        `compact` keeps `history`/`session` from growing without bound, one of:
          - `True`: `runa.compact.default_compactor` -- past `DEFAULT_COMPACTION_TOKENS`, keep
            only the most recent exchange. A rolling window, not a summary.
          - a `runa.compact.Compactor` (any `(items, usage_tokens) -> items | None` callable):
            your own strategy -- a different threshold, an LLM summary, whatever you return.
          - `False` (the default): off.
        Resolved to a `Compactor` (or `None`) here, as `self.compactor`, the way `memory=`/
        `knowledge=` are; anything else is a `UserError` at construction.

        `max_turns` caps how many model calls one run may make before `MaxTurnsExceeded`;
        `max_tokens` caps the tokens they may spend before `MaxTokensExceeded`; `timeout` caps
        the run's wall-clock seconds before `RunTimeout`. All three are optional ceilings, and
        all three end the run as `Run(status="error")`.
        """
        cls = type(self)
        if cls is Agent:
            raise UserError("Agent must be subclassed, e.g. `class MyAgent(Agent): name = ...`")
        _reject_unknown_settings(cls)

        setting = _settings(cls, locals())

        name = setting["name"]
        if name is _UNSET:
            raise UserError(f"{cls.__name__}(...) is missing the required 'name'")
        instructions = setting["instructions"]
        if instructions is _UNSET:
            instructions = _load_prompt(cls, name)

        # Before the subagents: one that lists this agent back wraps it as a tool named after it.
        self.name: str = name

        handoffs: list[Any] = []
        delegates: list[Delegate] = []
        tools = list(setting["tools"] or [])
        with _agent_graph(self) as build:
            for sub in _subagents(setting["subagents"] or []):
                if isinstance(sub, Subagent):
                    agent = build(sub.agent)
                    if sub.mode == "handoff":
                        handoffs.append(agent)
                    else:
                        delegates.append(
                            Delegate.from_agent(agent, sub.tool_name, sub.tool_description)
                        )
                else:
                    agent = build(sub)
                    handoffs.append(agent)
                    delegates.append(Delegate.from_agent(agent))

        self.memory = _resolve_retrieval_setting(setting["memory"], Memory, tools)
        self.knowledge = _resolve_retrieval_setting(setting["knowledge"], Knowledge, tools)

        self.instructions = _adapt_instructions(instructions)
        self.model: str | Any = _checked_model(setting["model"])
        self.model_settings: ModelSettings = setting["model_settings"] or ModelSettings()
        self.tools: list[FunctionTool] = tools
        self.handoffs: list[Any] = handoffs
        self.delegates: list[Delegate] = delegates
        self.mcp_servers: list[Any] = list(setting["mcp"] or [])
        # Not `self.guardrails`: that name belongs to the list a subclass declares, the way
        # `mcp` is declared and `mcp_servers` is what the declaration resolved to.
        self.bound_guardrails: dict[Phase, list[BoundGuardrail]] = flatten_guardrails(
            setting["guardrails"] or []
        )
        self.output_type: type | None = setting["output_type"]
        self.hooks: Hooks[Any] | None = setting["hooks"]
        # Not `self.compact`: that name belongs to the bool/callable a subclass declares, the way
        # `mcp` is declared and `mcp_servers` is what the declaration resolved to.
        self.compactor: Compactor | None = _resolve_compactor(setting["compact"])
        self.max_turns: int = setting["max_turns"]
        self.max_tokens: int | None = setting["max_tokens"]
        self.timeout: float | None = setting["timeout"]

        self.history: list[ConversationItem] = []
        self.usage = Usage()
        self.last_usage = Usage()
        self._in_flight = 0

    @property
    def graph(self) -> Digraph:
        """Render this agent, and its tools/subagents, as a Graphviz diagram.

        Displays inline in Jupyter; call `.render(filename)` to save it, or `.source` for the
        raw DOT text. Actually rendering an image needs the system Graphviz `dot` binary.
        """
        from runa._graph import draw_graph

        return draw_graph(self)

    def _record(self, run: Run, session: Session | None) -> Run:
        """Record `run`'s usage (and, if it completed a session-less turn, its history).

        The one thing an `Agent` does to a finished run, whichever of the three ways it finished:
        the loop builds the `Run` itself, so all that is left here is the bookkeeping only an
        instance can do -- which conversation this run belongs to, and what it has spent across
        every run so far. Usage is recorded for a failed run too, since the tokens were spent
        either way.

        History is written back only by a turn that completed. A paused run's is not over, and
        `to_state()` already carries what resuming it needs; a failed one's never will be, and an
        unfinished turn is not something the next one should start from.
        """
        self.last_usage = run.usage
        self.usage.add(self.last_usage)
        if run.status == "completed" and session is None:
            self.history = run._transcript.history()
        return run

    def _fresh(self) -> Agent:
        """A copy of this agent with empty per-run state, sharing its config.

        For a run that must not inherit this instance's conversation, and may be one of several
        in flight on it: an eval's agent is declared once per dataset, and its cases run
        concurrently, so running the shared instance directly would both accumulate history
        across unrelated cases and corrupt it when two overlap. (A delegate needs no copy: its
        nested run never goes through `run`, so it never touches an instance's history.)
        """
        clone = copy.copy(self)
        clone.history = []
        clone.usage = Usage()
        clone.last_usage = Usage()
        clone._in_flight = 0
        return clone

    @contextmanager
    def _exclusive(self, session: Session | None) -> Iterator[None]:
        """Hold this instance for one run, refusing a second concurrent run over `self.history`.

        Concurrent runs on one instance are safe exactly when their history lives somewhere
        else: each run reads `self.history` at the start and writes it back at the end, so two
        overlapping session-less runs silently lose one conversation into the other. That is a
        data leak between users in a server, not a crash, so it is refused rather than allowed to
        happen quietly. Pass a `session` per run (each one's history is its own), or build an
        `Agent` per run; both are cheap.
        """
        exclusive = session is None
        if exclusive and self._in_flight:
            raise UserError(
                f"{type(self).__name__} is already running: one Agent instance cannot run "
                "concurrently without a session, because both runs would share (and overwrite) "
                "`self.history`. Pass a `session=` to each run, or use one Agent per run."
            )
        self._in_flight += 1
        try:
            yield
        finally:
            self._in_flight -= 1

    async def run(
        self,
        message: MessageContent | RunState,
        context: Any = None,
        hooks: Hooks[Any] | None = None,
        session: Session | str | None = None,
    ) -> Run:
        """Run a turn asynchronously, appending it to the conversation history.

        `message` is plain text, or a list for a multimodal message: a bare string is text, or
        an image when it is an image URL or a `data:image/...` URI; a local image is a
        `Path("cat.jpg")`, never a bare string, since the list may carry a user's own words (see
        `content.parts`). Build a part explicitly with `content.text(...)`/`content.image(...)`
        when a string doesn't have a recognizable image extension. It can also be the `RunState`
        of a paused `Run`, once its interruptions are approved or rejected, to resume that run.

        It is always one user turn, never a transcript: a list of `{"role": ...}` messages raises
        `TypeError`. Start from an earlier conversation by setting `self.history` directly, or by
        seeding the `session` with `add_items` before the first run.

        `context` is available to a single-argument `instructions` callable (and to tools,
        guardrails, etc.) as-is; it is never sent to the model. `hooks` receives lifecycle
        callbacks (`on_agent_start`, `on_tool_end`, etc.); it defaults to `LoggingHooks`.

        Pass a `session` to persist conversation history there instead of on `self.history`; the
        session supplies prior turns automatically, so only the new `message` is sent as input,
        and `self.history` is left untouched. Resume a paused run with the same `session` it
        started with.

        A `session` is normally the conversation's id: `session="user-42"` persists to whichever
        backend `RUNA_DATABASE_URL` names, so an app moves to Postgres without naming one here.
        Pass a `Session` instead when the run needs more than the id -- `db.session(id,
        user_id=...)` to scope the run's memory, or your own store.

        Returns a `Run` exposing `.output`, `.status`, `.interruptions`, `.trace`, `.usage` and
        `.error`. A tool call needing approval pauses the run (`status="paused"`). Nothing a run
        fails on propagates: a guardrail tripwire, `MaxTurnsExceeded`, a model error, and equally
        an exception out of application code the run called -- a `Hooks` method, a `Session`, a
        `Compactor` -- all come back as `status="error"` with the message on `.error`
        (`UnexpectedError` wraps the last kind, and its traceback is logged). The two that do
        raise are `asyncio.CancelledError`, because a caller cancelling a run wants it stopped,
        and `OperatorError`, which is not a run failure at all (see `runa.exceptions`).
        `self.history` only changes once a turn completes.

        Token usage for this call is recorded to `self.last_usage` and accumulated into
        `self.usage`, regardless of `session`, `hooks`, or whether the run errored.

        An `Agent` instance holds the state of the conversation it is running, so **one instance
        runs one conversation at a time**. Two overlapping session-less runs on the same instance
        raise `UserError` rather than quietly interleaving their histories; give each run its own
        `session`, or its own `Agent`. Under a web server, build the agent inside the request
        handler (`runa serve` does exactly this).
        """
        session = _resolve_session(session)
        turn_input = _turn_input(message, self.history, session, context)
        with self._exclusive(session):
            # No `except` for a run's own failure: the loop's boundary returns the `"error"` `Run`
            # itself (`run_internal/run_loop._errored`), so there is nothing here to translate.
            # What still comes through is what a run is not allowed to swallow -- an
            # `OperatorError`, a `CancelledError` -- and those are the caller's to see.
            run = await _run_async(
                self,
                turn_input,
                context=context,
                hooks=hooks,
                run_config=RunConfig.of(self, session),
                session=session,
            )
            return self._record(run, session)

    def run_sync(
        self,
        message: MessageContent | RunState,
        context: Any = None,
        hooks: Hooks[Any] | None = None,
        session: Session | str | None = None,
    ) -> Run:
        """Synchronous `run`, for callers not already inside an event loop.

        The event loop is this call's own, and what the turn opened on it that must be shut down
        is, before it goes (see `runa._loop.run`): a `runa chat` turn closes every `.stdio` MCP
        server it reached, a subagent's included, instead of leaving a subprocess per turn behind.
        `await agent.run(...)` on an app's own loop keeps them open for the agent's lifetime.

        Its trace is written by the time it returns, so a script or a `runa chat` turn can read it
        back straight away; `await agent.run(...)` queues its export and returns without waiting.
        """
        try:
            return _loop.run(self.run(message, context, hooks, session))
        finally:
            flush_exports()

    def run_streamed(
        self,
        message: MessageContent | RunState,
        context: Any = None,
        hooks: Hooks[Any] | None = None,
        session: Session | str | None = None,
    ) -> RunStream:
        """Run a turn as a stream of events: the same run as `run`, with the same arguments.

        Iterate the returned `RunStream` for `StreamEvent`s (`raw_response_event`,
        `run_item_stream_event`, `agent_updated_stream_event`) as they arrive; once it ends,
        its `.run` holds the `Run` that `run` would have returned, paused, completed or errored.
        History and usage are recorded only once the stream is fully consumed, so a caller that
        stops iterating early leaves them unchanged.

        One instance still runs one conversation at a time: a stream shares `run`'s latch, so
        starting to consume a second session-less stream on the same instance while one is in
        flight raises `UserError` rather than interleaving the two histories.
        """
        session = _resolve_session(session)
        turn_input = _turn_input(message, self.history, session, context)
        run_config = RunConfig.of(self, session)

        async def events() -> AsyncIterator[StreamEvent]:
            # The latch is held here rather than in `run_streamed` because that is where the
            # turn runs: nothing is executed and `self.history` is untouched until a consumer
            # iterates, so a stream built and dropped holds nothing, and the generator's own
            # teardown releases one abandoned mid-flight. It sits outside the `except` below so
            # the refusal reaches the caller the way `run`'s does, as a raise rather than an
            # error `Run`.
            with self._exclusive(session):
                # The loop is a task writing into a queue, rather than an async generator,
                # because it has to keep running between a consumer's `__anext__` calls: a tool
                # call and the model request after it are the loop's work, not the caller's, and
                # a caller that stops iterating must not leave a turn half-executed.
                # `task.result()` is the finished `Run`, errored included -- see `run`, which has
                # no `except` for a run's own failure either.
                queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
                task = asyncio.ensure_future(
                    _run_async(
                        self,
                        turn_input,
                        context=context,
                        hooks=hooks,
                        run_config=run_config,
                        session=session,
                        emit=queue.put_nowait,
                    )
                )
                task.add_done_callback(lambda _: queue.put_nowait(None))
                try:
                    while (event := await queue.get()) is not None:
                        yield event
                    stream.run = self._record(task.result(), session)
                finally:
                    task.cancel()

        stream = RunStream(events())
        return stream

    async def evaluate(
        self,
        dataset: Iterable[Case],
        *,
        judge: str | None = None,
        threshold: float | None = None,
        thresholds: dict[str, float] | None = None,
        concurrency: int = 8,
    ) -> Report:
        """Run every case in `dataset` through this agent and grade it: see `runa.eval`.

        Deterministic checks and judge-graded semantic metrics (task completion, answer
        correctness/relevance, faithfulness, tool correctness) are chosen automatically per case
        based on what evidence it supplies; no metric configuration is required. `judge` overrides
        the model semantic metrics grade with, defaulting to this agent's own `model`. Up to
        `concurrency` cases run at once.
        """
        from runa.eval.evaluate import evaluate_agent

        return await evaluate_agent(
            self,
            dataset,
            judge=judge,
            threshold=threshold,
            thresholds=thresholds,
            concurrency=concurrency,
        )


_AGENT_FIELDS = tuple(inspect.signature(Agent.__init__).parameters)[1:]
"""Every name that configures an Agent, read off the one place they are written down.

`Agent.__init__`'s keyword-only parameters are the closed set, so the two ways of saying the same
thing can't drift: `_settings` walks this tuple to fall each parameter back to the class attribute
of the same name, and `_reject_unknown_settings` refuses any other name in a class body -- both
read the signature rather than repeat it. The constructor needs no such
check, having a signature: an unknown keyword is Python's own `TypeError`, suggesting the name it
meant, and a type checker has already flagged the call (and a `max_turns="ten"` the validator
would have let through).

A new setting also has to reach a run, by exactly one of two paths chosen by what the setting
is: a ceiling on the run (`max_turns`, `max_tokens`, `timeout`) belongs to `RunConfig`, which
configures the call rather than the agent; everything the agent *is* is read off the agent itself,
through a property of `runa.agent_like.AgentLike`, so a loop reading a setting that protocol
doesn't name is a type error rather than a default.
"""


@dataclass(frozen=True)
class Subagent:
    """A wired-up subagent, attached as a handoff or a delegate tool."""

    agent: type[Agent]
    mode: Mode
    tool_name: str | None = None
    tool_description: str | None = None

    def __call__(
        self, *, tool_name: str | None = None, tool_description: str | None = None
    ) -> Subagent:
        """Return a copy with the tool name/description overridden."""
        return replace(self, tool_name=tool_name, tool_description=tool_description)


__all__ = ["Agent", "Run", "RunState", "Subagent"]
