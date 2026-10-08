"""Class-based Agent, built on Runa's own runtime (`runa.run_internal`).

`run`/`run_sync`/`run_streamed` are the only way to run an Agent, and `Run` the only thing they
return. The turn loop under them (`run_internal/run_loop._run_async`) is this class's
implementation, not a second entry point: a `Runner` that forwarded to it used to sit here, which
is how `eval/judge.py` and `eval/tracing/adapter.py` came to run agents without an Agent's own
wiring -- its guardrail flattening, its memory and knowledge resolution, its `RunConfig`, its
refusal of two concurrent session-less runs. One door, so there is nothing to go around.
"""

import asyncio
import copy
import difflib
import inspect
import re
from collections.abc import AsyncIterator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import MISSING, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from runa import content
from runa._items import ConversationItem
from runa._models import DEFAULT_MODEL, ModelProvider
from runa._types import MessageContent, ModelSettings, RunContextWrapper, Usage
from runa.compact import Compactor, default_compactor
from runa.exceptions import RunaError, UserError
from runa.guardrail import BoundGuardrail, Phase, flatten_guardrails
from runa.handoff import agent_as_tool
from runa.knowledge import Knowledge
from runa.lifecycle import RunHooks, logger
from runa.memory import Memory
from runa.run import Run, RunStream
from runa.run_internal.run_config import DEFAULT_MAX_TURNS, RunConfig
from runa.run_internal.run_loop import _run_async
from runa.run_state import RunState
from runa.session import Session
from runa.stream_events import StreamEvent
from runa.tool import FunctionTool
from runa.tracing.manual import current_trace

if TYPE_CHECKING:
    from graphviz import Digraph

    from runa.eval.case import Case
    from runa.eval.report import Report

_AGENT_FIELDS = (
    "name",
    "instructions",
    "model",
    "model_settings",
    "tools",
    "subagents",
    "guardrails",
    "mcp",
    "mcp_servers",
    "output_type",
    "hooks",
    "memory",
    "knowledge",
    "compact",
    "max_turns",
    "max_tokens",
    "timeout",
)
"""Every name that configures an Agent, whether declared on the subclass or passed to it.

One tuple for both call sites, so the two ways of saying the same thing can't drift: the merge
loop in `__init__` reads each of these off `type(self)` as the default for the matching kwarg,
and `_reject_unknown_settings` refuses any other name in either place.
"""

_MODEL_PROVIDER = ModelProvider()


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

    def _resolved(run_context: RunContextWrapper[Any], _agent: Agent) -> Any:
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
    since the loop's session path only wraps a bare string into `{"role": "user", ...}`
    itself.
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
        return resolved if isinstance(resolved, str) else [{"role": "user", "content": resolved}]
    return [*history, {"role": "user", "content": resolved}]


_CAMEL_CASE_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _snake_case(name: str) -> str:
    return _CAMEL_CASE_BOUNDARY.sub("_", name).lower()


_NO_SOURCE_FILE = (TypeError, OSError)


def _load_prompt(cls: type, name: str) -> str | None:
    """Read `<name>.md` from the `prompts/` directory next to `cls`'s `app/agents/` module.

    Mirrors `runa generate prompt`'s naming: `app/prompts/<snake_case(name)>.md`, a sibling of
    the `agents/` directory the subclass is defined in. Reading only: constructing an Agent never
    writes to the project tree, so an immutable image or a read-only `app/` is an ordinary way to
    ship one. Writing that stub is `runa generate agent`/`runa generate prompt`'s job.

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
    prompt_file = module_file.parent.parent / "prompts" / f"{_snake_case(name)}.md"
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
SubagentsDict = dict[Literal["handoff", "delegate", "auto"], SubagentsList]


def _flatten_subagents(subagents: SubagentsList | SubagentsDict) -> SubagentsList:
    """Normalize the `subagents` class attribute to the flat list the wiring loop expects.

    Accepts either a plain list (each entry bare, or `.handoff`/`.delegate`-wrapped) or a
    `{"handoff": [...], "delegate": [...], "auto": [...]}` dict, where the key supplies the
    mode for any bare entry.
    """
    if not isinstance(subagents, dict):
        return list(subagents)
    flat: SubagentsList = []
    for mode, subs in subagents.items():
        for sub in subs:
            flat.append(sub if mode == "auto" or isinstance(sub, Subagent) else Subagent(sub, mode))
    return flat


def _resolve_compactor(compact: Any) -> Compactor | None:
    """Resolve a `compact=` setting to the `Compactor` the turn loop should run, or `None` for off.

    Resolved here, alongside `memory=`/`knowledge=`, for the reason they are: a declared attribute
    becomes a usable object once, at construction, so a value that is neither a bool nor a
    callable is a `UserError` naming the attribute rather than a `TypeError` raised from inside
    `run_internal.run_loop._maybe_compact` after a model call has already been paid for.
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
    """Check `model=` names a model or is one, without building it yet.

    The exception to the rule the others follow: a declared attribute becomes a usable object at
    construction (see `_resolve_compactor`), but a model name cannot, because resolving it builds
    the provider's client and that is a `UserError` when the matching API key is unset -- which an
    agent that `runa generate` just wrote, or `graph` renders, or no one ever runs has no business
    needing. Resolution therefore waits for the run (`AgentShape.resolve_model`). Rejecting a
    value that is neither a name nor a `Model` needs no client, though, so a mistyped `model=` is
    still a `UserError` naming the attribute here rather than an `AttributeError` from inside the
    turn loop, with an `llm` span already open. `Model` is checked structurally, the way
    `run_internal` reads every backend.
    """
    if model is None or isinstance(model, str) or hasattr(model, "get_response"):
        return model
    raise UserError(
        f"model must be a model name, a Model instance, or None for Runa's default, got {model!r}"
    )


def _is_code(value: Any) -> bool:
    """Is this class-body entry code (a method, a property, a nested class) rather than config?

    Config is data -- a string, a list, a number, a `ModelSettings`. Everything a subclass
    legitimately *writes* is callable or a descriptor, which is what keeps
    `_reject_unknown_settings` from standing between a subclass and its own helper methods.
    """
    return callable(value) or isinstance(value, staticmethod | classmethod | property)


def _reject_unknown_settings(cls: type[Agent], kwargs: dict[str, Any]) -> None:
    """Refuse a name that isn't one of `_AGENT_FIELDS`, declared on `cls` or passed to it.

    Class attributes are the whole configuration surface, so the one mistake that style invites
    is a misspelled name -- and a silently ignored `modell = "claude-sonnet-5"` doesn't fail, it
    runs the default model at a different price with different behaviour and surfaces days later
    as "the agent answers oddly". The same promise `MCPServer`'s named parameters make (a
    misspelled option is an error at the call site, not a server that ignores it) is made here,
    for both the declaration and the constructor override, against one set of names.

    Only data is checked: a name starting with `_`, a method, a property or a nested class is the
    subclass's own business. Mixins in the MRO are skipped for the same reason -- their attributes
    aren't claiming to be Runa settings.
    """
    for name in kwargs:
        if name not in _AGENT_FIELDS:
            raise UserError(
                f"{cls.__name__}(...) got an unexpected keyword argument {name!r}."
                f"{_did_you_mean(name)}"
            )
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

    def __init__(self, mode: Literal["handoff", "delegate"]) -> None:
        self.mode: Literal["handoff", "delegate"] = mode

    def __get__(self, instance: object, owner: type[Agent]) -> Subagent:
        return Subagent(owner, self.mode)


class Agent:
    """An Agent whose config comes from class attributes instead of __init__ args.

    `usage` accumulates token usage across every `run`/`run_sync`/`run_streamed` call made on
    this instance; `last_usage` holds just the most recent call's usage. Both are read from
    `RunContextWrapper.usage`, which the turn loop populates regardless of `hooks`.

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

    def __init__(self, **kwargs: Any) -> None:
        """Build config from class attributes and wire up any subagents and guardrails.

        Every `_AGENT_FIELDS` name can be declared on the subclass or passed here, the kwarg
        winning; any other name is a `UserError` in either place, naming the setting it was
        probably meant to be (see `_reject_unknown_settings`).

        `mcp=[...]` (as a constructor kwarg, or a `mcp` class attribute) is sugar for
        `mcp_servers=[...]`; both are merged into `mcp_servers` if given together.

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
        if type(self) is Agent:
            raise TypeError("Agent must be subclassed, e.g. `class MyAgent(Agent): name = ...`")

        _reject_unknown_settings(type(self), kwargs)
        for field_name in _AGENT_FIELDS:
            value = getattr(type(self), field_name, MISSING)
            if value is not MISSING:
                kwargs.setdefault(field_name, value)
        if "name" not in kwargs:
            raise TypeError(f"{type(self).__name__}(...) is missing the required 'name'")
        if "instructions" not in kwargs:
            kwargs["instructions"] = _load_prompt(type(self), kwargs["name"])

        handoffs: list[Any] = []
        tools = list(kwargs.get("tools") or [])
        mcp_servers = [*(kwargs.get("mcp_servers") or []), *(kwargs.get("mcp") or [])]
        for sub in _flatten_subagents(kwargs.get("subagents") or []):
            if isinstance(sub, Subagent):
                agent = sub.agent()
                if sub.mode == "handoff":
                    handoffs.append(agent)
                else:
                    tools.append(agent.as_tool(sub.tool_name, sub.tool_description))
            else:
                agent = sub()
                handoffs.append(agent)
                tools.append(agent.as_tool(None, None))

        bound_guardrails = flatten_guardrails(kwargs.get("guardrails") or [])

        self.memory = _resolve_retrieval_setting(kwargs.get("memory"), Memory, tools)
        self.knowledge = _resolve_retrieval_setting(kwargs.get("knowledge"), Knowledge, tools)

        self.name: str = kwargs["name"]
        self.instructions = _adapt_instructions(kwargs.get("instructions"))
        self.model: str | Any = _checked_model(kwargs["model"])
        self.model_settings: ModelSettings = kwargs.get("model_settings") or ModelSettings()
        self.tools: list[FunctionTool] = tools
        self.handoffs: list[Any] = handoffs
        self.mcp_servers: list[Any] = mcp_servers
        # Not `self.guardrails`: that name belongs to the list/dict a subclass declares, the way
        # `mcp` is declared and `mcp_servers` is what the declaration resolved to.
        self.bound_guardrails: dict[Phase, list[BoundGuardrail]] = bound_guardrails
        self.output_type: type | None = kwargs.get("output_type")
        self.hooks = kwargs.get("hooks")
        # Not `self.compact`: that name belongs to the bool/callable a subclass declares, the way
        # `mcp` is declared and `mcp_servers` is what the declaration resolved to.
        self.compactor: Compactor | None = _resolve_compactor(kwargs.get("compact"))
        self.max_turns: int = kwargs["max_turns"]
        self.max_tokens: int | None = kwargs.get("max_tokens")
        self.timeout: float | None = kwargs.get("timeout")

        self.history: list[ConversationItem] = []
        self.usage = Usage()
        self.last_usage = Usage()
        self._in_flight = 0

    def as_tool(self, tool_name: str | None, tool_description: str | None) -> FunctionTool:
        """Wrap this agent as a tool another agent can call; see `runa.handoff.agent_as_tool`."""
        return agent_as_tool(self, tool_name, tool_description)

    @property
    def graph(self) -> Digraph:
        """Render this agent, and its tools/subagents, as a Graphviz diagram.

        Displays inline in Jupyter; call `.render(filename)` to save it, or `.source` for the
        raw DOT text. Actually rendering an image needs the system Graphviz `dot` binary.
        """
        from runa._graph import draw_graph

        return draw_graph(self)

    def _run_config(self, session: Session | None) -> RunConfig:
        """Group this run under an enclosing `tracing.trace` block, else under its session."""
        outer = current_trace()
        return RunConfig(
            model_provider=_MODEL_PROVIDER,
            workflow_name=type(self).__name__,
            group_id=outer.id if outer else (session.session_id if session is not None else None),
            max_turns=self.max_turns,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

    def _completed(self, run: Run, session: Session | None) -> Run:
        """Record `run`'s usage (and, unless paused or session-backed, its history), and return it.

        The loop builds the `Run` itself, so all that is left here is the bookkeeping only an
        Agent instance can do: which conversation this run belongs to, and what it has spent
        across every run so far. A paused run's history is not written back: the turn is not over,
        and `to_state()` already carries what resuming it needs.
        """
        self.last_usage = run.usage
        self.usage.add(self.last_usage)
        if run.status != "paused" and session is None:
            self.history = run._history()
        return run

    def _failed(self, exc: RunaError) -> Run:
        """Record what a run that `exc` stopped had used, and ran, as an `"error"` `Run`."""
        context_wrapper = exc.run_data.context_wrapper if exc.run_data else RunContextWrapper()
        self.last_usage = context_wrapper.usage
        self.usage.add(self.last_usage)
        return Run(
            output=None,
            trace=exc.run_data.trace if exc.run_data else None,
            usage=self.last_usage,
            status="error",
            error=str(exc),
            guardrail_results=context_wrapper.guardrail_results.snapshot(),
        )

    def _fresh(self) -> Agent:
        """A copy of this agent with empty per-run state, sharing its config.

        For a run that must not inherit this instance's conversation, and may be one of several
        in flight on it. An agent declared once is reused for many such runs -- a `Subagent` is
        built once per caller, an eval's agent once per dataset -- so running the shared instance
        directly would both accumulate history across unrelated runs and corrupt it when two
        overlap, whether the model delegated twice in one message or two eval cases are running
        concurrently.
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
        hooks: RunHooks[Any] | None = None,
        session: Session | str | None = None,
        *,
        _context_wrapper: RunContextWrapper[Any] | None = None,
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
        callbacks (`on_agent_start`, `on_tool_end`, etc.); it defaults to `LoggingRunHooks`.

        Pass a `session` to persist conversation history there instead of on `self.history`; the
        session supplies prior turns automatically, so only the new `message` is sent as input,
        and `self.history` is left untouched. Resume a paused run with the same `session` it
        started with.

        A `session` is normally the conversation's id: `session="user-42"` persists to whichever
        backend `RUNA_DATABASE_URL` names, so an app moves to Postgres without naming one here.
        Pass a `Session` instead when the run needs more than the id -- `db.session(id,
        user_id=...)` to scope automatic memory, or your own store.

        Returns a `Run` exposing `.output`, `.status`, `.interruptions`, `.trace`, `.usage` and
        `.error`. A tool call needing approval pauses the run (`status="paused"`). A guardrail
        tripwire, `MaxTurnsExceeded`, or another `RunaError` is caught and reported as
        `status="error"` instead of propagating. `self.history` only changes once a turn
        completes.

        Token usage for this call is recorded to `self.last_usage` and accumulated into
        `self.usage`, regardless of `session`, `hooks`, or whether the run errored.

        An `Agent` instance holds the state of the conversation it is running, so **one instance
        runs one conversation at a time**. Two overlapping session-less runs on the same instance
        raise `UserError` rather than quietly interleaving their histories; give each run its own
        `session`, or its own `Agent`. Under a web server, build the agent inside the request
        handler (`runa serve` does exactly this).

        `_context_wrapper` is internal: a nested run takes the fork of an enclosing run's
        `RunContextWrapper` instead of building a fresh one, which is what shares the approval
        ledger, the usage accounting and the guardrail audit trail between the two. `context` is
        ignored when it's given, since a fork already carries the enclosing run's. Don't pass it
        directly; delegation, the one thing that needs it, is `runa.handoff`'s to wire up.
        """
        session = _resolve_session(session)
        turn_input = _turn_input(message, self.history, session, context)
        with self._exclusive(session):
            try:
                run = await _run_async(
                    self,
                    turn_input,
                    context=context,
                    hooks=hooks,
                    run_config=self._run_config(session),
                    session=session,
                    _context_wrapper=_context_wrapper,
                )
            except RunaError as exc:
                return self._failed(exc)
            return self._completed(run, session)

    def run_sync(
        self,
        message: MessageContent | RunState,
        context: Any = None,
        hooks: RunHooks[Any] | None = None,
        session: Session | str | None = None,
    ) -> Run:
        """Synchronous `run`, for callers not already inside an event loop."""
        return asyncio.run(self.run(message, context, hooks, session))

    def run_streamed(
        self,
        message: MessageContent | RunState,
        context: Any = None,
        hooks: RunHooks[Any] | None = None,
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
        run_config = self._run_config(session)

        async def events() -> AsyncIterator[StreamEvent]:
            # The latch is held here rather than in `run_streamed` because that is where the
            # turn runs: nothing is executed and `self.history` is untouched until a consumer
            # iterates, so a stream built and dropped holds nothing, and the generator's own
            # teardown releases one abandoned mid-flight. It sits outside the `except RunaError`
            # below so the refusal reaches the caller the way `run`'s does, as a raise rather
            # than an error `Run`.
            with self._exclusive(session):
                # The loop is a task writing into a queue, rather than an async generator,
                # because it has to keep running between a consumer's `__anext__` calls: a tool
                # call and the model request after it are the loop's work, not the caller's, and
                # a caller that stops iterating must not leave a turn half-executed.
                # `task.result()` re-raises whatever the run raised, on this side of the seam,
                # where `_failed` turns it into the same `Run` the non-streamed path returns.
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
                    stream.run = self._completed(task.result(), session)
                except RunaError as exc:
                    stream.run = self._failed(exc)
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


@dataclass(frozen=True)
class Subagent:
    """A wired-up subagent, attached as a handoff or a delegate tool."""

    agent: type[Agent]
    mode: Literal["handoff", "delegate"]
    tool_name: str | None = None
    tool_description: str | None = None

    def __call__(
        self, *, tool_name: str | None = None, tool_description: str | None = None
    ) -> Subagent:
        """Return a copy with the tool name/description overridden."""
        return replace(self, tool_name=tool_name, tool_description=tool_description)


__all__ = ["Agent", "Run", "RunState", "Subagent"]
