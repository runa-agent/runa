"""Class-based Agent, built on Runa's own runtime (`runa.run_internal`).

`run`/`run_sync`/`run_streamed` are the only way to run an Agent, and `Run` the only thing they
return. The turn loop under them is this class's implementation, not a second entry point, so
every run gets an Agent's own wiring: its guardrail flattening, its memory and knowledge
resolution, its trace's grouping, its refusal of two concurrent session-less runs.
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

`SupportAgent(memory=None)` turns off a declared `memory`, so `None` can't also mean "not passed".
Typed `Any` so every parameter keeps its own annotation.
"""


def _adapt_instructions(instructions: Any) -> Any:
    """Let `instructions` be a `(context) -> str` callable instead of the runner's 2-arg shape.

    The turn loop calls a callable `instructions(run_context, agent)`. A single-parameter callable
    is wrapped so it receives just `run_context.context`; anything else passes through unchanged.
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

    A string is resolved through `runa.db.session`, so the backend follows `RUNA_DATABASE_URL`;
    a `Session` object is passed straight through.
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

    A paused `RunState` passes through to be resumed; it carries its own `context`, so a
    `context=` alongside one is refused rather than dropped. A list `message` goes through
    `runa.content.parts`. With no `session`, the message joins `history`; with one, only the new
    turn is sent, a plain string as is and a multimodal one as a single user message.
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

    An agent's `name`, and the module, prompt and eval dataset named after it, follow this rule,
    as do `runa generate`'s tool and guardrail names.
    """
    return _CAMEL_CASE_BOUNDARY.sub("_", name).lower()


def declared_name(agent_cls: type[Agent]) -> str:
    """The `name` `agent_cls` declares, falling back to the class's own name if it declares none.

    For every surface that reads an `Agent` subclass without building one (`runa ui`, `runa
    serve`'s routes, `runa chat`), so an agent answers to one name everywhere. An agent without a
    `name` is still listed this way, though constructing it raises.
    """
    return getattr(agent_cls, "name", agent_cls.__name__)


def prompt_path(app_dir: Path, name: str) -> Path:
    """Where the agent called `name` keeps its instructions: `<app_dir>/prompts/<name>.md`.

    Read by `_load_prompt` and written by `runa generate prompt`.
    """
    return app_dir / "prompts" / f"{snake_case(name)}.md"


_NO_SOURCE_FILE = (TypeError, OSError)


def _load_prompt(cls: type, name: str) -> str | None:
    """Read `<name>.md` from the `prompts/` directory next to `cls`'s `app/agents/` module.

    Never writes, so a read-only `app/` works. Returns `None` when `cls` has no source file (a
    REPL), its module isn't in an `agents/` directory, or the prompt file is missing; the last
    case logs the path `runa generate prompt` would write.
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

    Anything but a list (a dict grouping entries by mode, say) is a `UserError`.
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

    A class listed twice below one root is built once, and a cycle (a specialist handing back to
    triage) closes onto the agent still being built. Construction stays eager, so a broken
    subagent declaration fails when the root is built.
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

    Resolved at construction, so a bad value is a `UserError` before any model call is paid for.
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

    Resolving a name builds the provider's client, which needs its API key, so that waits for the
    run; checking the name doesn't, so a bad one is a `UserError` at construction.
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

    Only data is checked as a setting, so a subclass's own helpers are never rejected.
    """
    return callable(value) or isinstance(value, staticmethod | classmethod | property)


def _reject_unknown_settings(cls: type[Agent]) -> None:
    """Refuse a class attribute of `cls` that isn't one of `_AGENT_FIELDS`.

    A misspelled `modell = ...` would otherwise run on the default model. The constructor needs no
    such check: a misspelled keyword is a `TypeError` from its signature. Only data is checked: a
    `_`-prefixed name, a method, a property, a nested class and a mixin's attributes are skipped.
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

    `passed` is `__init__`'s `locals()`, one entry per parameter, so a setting added to the
    signature is resolved with no edit here. A setting neither supplies is `None`, except `name`
    and `instructions`, which stay `_UNSET` for `__init__` to act on.
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

    `"auto"` builds a default instance, `"llm"` appends a search tool and resolves to `None`, and
    any non-string (`None`, or a `MemoryLike`/`KnowledgeLike` object) passes through. Any other
    string is a `UserError`.
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
    max_turns = 10
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

        These parameters are the closed set of settings: each can be declared on the subclass or
        passed here, the argument winning. An unknown name is a `TypeError` here, and a
        `UserError` naming the likely setting when declared in a class body.

        `mcp` lists the agent's MCP servers; `self.mcp_servers` is what it resolved to.

        `memory` opts this agent into long-term memory, one of:
          - `"auto"` (or a `Memory(...)` instance, or any object shaped like `runa.memory`'s
            `MemoryLike`): the turn loop retrieves relevant memories before each run and
            persists new ones after -- no manual `memory.search`/`.remember` calls.
          - `"llm"`: the model gets a `search_memory` tool and decides itself when to call it;
            no automatic retrieval/persistence.
          - `None` (the default): the agent behaves exactly as if `runa.memory` didn't exist.
        `"llm"` always uses a default `Memory()`; pass your own instance for a non-default
        `model`/`store`, or a custom `MemoryLike` object to replace retrieval entirely.

        `knowledge` opts this agent into retrieval from application/domain documents, one of
        `"auto"`, `"llm"`, a `Knowledge(...)` (or `KnowledgeLike`-shaped) instance, or `None` (the
        default) -- the same four shapes as `memory`, with the same meaning: `"auto"`/an instance
        searches automatically before every turn (no `tools=[...]` wiring needed); `"llm"` gives
        the model a `search_knowledge` tool it calls itself; `None` leaves the agent unaffected.
        `"llm"` always uses a default `Knowledge()`; pass your own instance for a non-default
        `directory`/`model`/`store`, or a custom `KnowledgeLike` object.

        `compact` keeps `history`/`session` from growing without bound, one of:
          - `True`: `runa.compact.default_compactor` -- past `DEFAULT_COMPACTION_TOKENS`, keep
            only the most recent exchange. A rolling window, not a summary.
          - a `runa.compact.Compactor` (any `(items, usage_tokens) -> items | None` callable):
            your own strategy -- a different threshold, an LLM summary, whatever you return.
          - `False` (the default): off.
        Resolved here, as `self.compactor`; anything else is a `UserError`.

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
        # Not `self.guardrails`, the name a subclass declares the list under.
        self.bound_guardrails: dict[Phase, list[BoundGuardrail]] = flatten_guardrails(
            setting["guardrails"] or []
        )
        self.output_type: type | None = setting["output_type"]
        self.hooks: Hooks[Any] | None = setting["hooks"]
        # Not `self.compact`, the name a subclass declares the setting under.
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

        Usage is recorded whatever the status; history only for a completed turn, since a paused
        one resumes from its `RunState` and a failed one is not a turn to build on.
        """
        self.last_usage = run.usage
        self.usage.add(self.last_usage)
        if run.status == "completed" and session is None:
            self.history = run._transcript.history()
        return run

    def _fresh(self) -> Agent:
        """A copy of this agent with empty per-run state, sharing its config.

        For runs that must not share a conversation, such as an eval's concurrent cases.
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

        Two overlapping session-less runs would overwrite one conversation with the other, so the
        second raises `UserError`. A run with a `session` keeps its history there and is never
        refused.
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

        Returns a `Run`. A tool call needing approval pauses it (`status="paused"`). Nothing a run
        fails on propagates: a guardrail tripwire, `MaxTurnsExceeded`, a model error and an
        exception out of a `Hooks`/`Session`/`Compactor` (wrapped in `UnexpectedError`, its
        traceback logged) all come back as `status="error"`. Only `asyncio.CancelledError` and
        `OperatorError` raise. `self.history` only changes once a turn completes; usage is
        recorded to `self.last_usage` and `self.usage` either way.

        **One instance runs one conversation at a time**: two overlapping session-less runs raise
        `UserError`. Give each run its own `session`, or its own `Agent`.
        """
        session = _resolve_session(session)
        turn_input = _turn_input(message, self.history, session, context)
        with self._exclusive(session):
            # A run's own failure comes back as an `"error"` `Run`; only an `OperatorError` or a
            # `CancelledError` gets through, and those are the caller's to see.
            run = await _run_async(
                self,
                turn_input,
                context=context,
                hooks=hooks,
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

        Runs on an event loop of its own, closing every `.stdio` MCP server the turn reached before
        the loop goes; `await agent.run(...)` keeps them open for the agent's lifetime. The trace
        is written by the time this returns, where `run` only queues its export.
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

        async def events() -> AsyncIterator[StreamEvent]:
            # Held only while a consumer iterates, so a stream built and dropped holds nothing.
            with self._exclusive(session):
                # A task feeding a queue keeps the turn running between `__anext__` calls.
                queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
                task = asyncio.ensure_future(
                    _run_async(
                        self,
                        turn_input,
                        context=context,
                        hooks=hooks,
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
"""Every name that configures an Agent: `Agent.__init__`'s keyword-only parameters.

A run reads each one off the agent through `runa.agent_like.AgentLike`.
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
