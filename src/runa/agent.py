"""Class-based Agent, built on Runa's own runtime (`runa.run_internal`).

`run`/`run_sync`/`run_streamed` are the only way to run an Agent, and `Run` the only thing they
return. The turn loop under them is this class's implementation, not a second entry point, so
every run gets an Agent's own wiring: its settings resolved once into the shape a run reads, its
trace's grouping, its refusal of two concurrent session-less runs.
"""

import asyncio
import copy
import difflib
import inspect
import re
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from runa import _app, _loop, content
from runa._items import ConversationItem, user_message
from runa._log import logger
from runa._models import DEFAULT_MODEL, Model
from runa._types import MessageContent, ModelSettings, Usage
from runa.compact import Compactor
from runa.exceptions import UserError
from runa.guardrail import GuardrailsList
from runa.handoff import Mode, Subagent
from runa.knowledge import KnowledgeLike
from runa.lifecycle import Hooks
from runa.memory import MemoryLike
from runa.run import Run, RunStream
from runa.run_internal.agent_shape import AgentShape
from runa.run_internal.run_loop import _run_async
from runa.run_state import RunState
from runa.session import Session
from runa.stream_events import StreamEvent
from runa.tool import FunctionTool
from runa.tracing.export import flush_exports

if TYPE_CHECKING:
    from graphviz import Digraph

    from runa.eval.case import Case
    from runa.eval.report import Report

_UNSET: Any = object()
"""`Agent.__init__`'s "this setting wasn't passed", which is not the same as being passed `None`.

`SupportAgent(memory=None)` turns off a declared `memory`, so `None` can't also mean "not passed".
Typed `Any` so every parameter keeps its own annotation.
"""


def _resolve_session(
    message: MessageContent | RunState, session: Session | str | None, agent: str
) -> Session | None:
    """The session this turn runs in: `session`, or the one a paused `message` started in.

    A string is resolved through `runa.db.session` as `agent`'s, so the backend follows
    `RUNA_DATABASE_URL`; a `Session` object is passed straight through. A `RunState` carries its
    own session, so one passed beside it is refused unless it is that session's `Session` object.
    """
    if isinstance(message, RunState):
        return message._session_for(session)
    if isinstance(session, str):
        from runa import db

        return db.session(session, agent)
    return session


def _turn_input(
    message: MessageContent | RunState,
    history: list[ConversationItem],
    session: Session | None,
    context: Any = None,
    user_id: str | None = None,
) -> str | list[ConversationItem] | RunState:
    """Build the `input` for the turn loop from this turn's `message`.

    A paused `RunState` passes through to be resumed; it carries its own `context` and
    `user_id`, so either one alongside it is refused rather than dropped (its session is
    `_resolve_session`'s). A list `message` goes
    through `runa.content.parts`. With no `session`, the message joins `history`; with one, only
    the new turn is sent, a plain string as is and a multimodal one as a single user message.
    """
    if isinstance(message, RunState):
        for name, value in (("context", context), ("user_id", user_id)):
            if value is not None:
                raise UserError(
                    f"a RunState already carries the {name} of the run it paused, so `{name}=` "
                    f"would be ignored when resuming it: pass `{name}=` to the first run instead."
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

_PROMPTS = "agent prompts"
"""The `_app` fact holding each class's prompt as read, so a reset forgets them with the app."""


def _prompt(cls: type, name: str) -> str | None:
    """`cls`'s instructions from `<name>.md`, read the first time an instance asks in this process.

    Like the code beside it, an edited prompt is picked up by a new process.
    """
    prompts = _app.held(_PROMPTS, dict[tuple[type, str], str | None])
    if (cls, name) not in prompts:
        prompts.setdefault((cls, name), _load_prompt(cls, name))
    return prompts[(cls, name)]


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


_LIST_SETTINGS = frozenset({"tools", "subagents", "guardrails", "mcp"})


def _stored(name: str, value: Any) -> Any:
    """`value` as the agent holds its `name` setting: a list setting's list as a tuple.

    `None` for a list setting is the empty tuple. Anything but a list stays as given, for the
    shape to refuse by name.
    """
    if name not in _LIST_SETTINGS:
        return value
    if value is None:
        return ()
    return tuple(value) if isinstance(value, list) else value


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
    `max_turns` (model calls, default 10), `max_tokens` (total tokens the run may spend, its
    memory's and knowledge's embedding and extraction calls included), and
    `timeout` (wall-clock seconds). Hitting any of them ends the run as `Run(status="error")`,
    the same as every other `RunaError`. `max_tokens` is the run's whole budget and is not the
    same knob as `ModelSettings(max_tokens=...)`, which caps one response's length.

    An instance carries per-run state (`history`, `usage`), so it is not safe to share across
    concurrent runs; see `run` for the rule and the two ways to satisfy it.

    A run reads the settings as they stood when the agent was built, or when one was last
    reassigned (`agent.model = ...` takes effect from the next run). The list settings are
    stored as tuples, so they change by reassignment only.
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
        `UserError` naming the likely setting when declared in a class body. Each is held under its
        own name as declared (`agent.memory` is `"auto"`, not the `Memory` it resolves to), and
        translated into what a run uses only by the shape the agent builds from them.

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
        Checked here; anything else is a `UserError`.

        `max_turns` caps how many model calls one run may make before `MaxTurnsExceeded`;
        `max_tokens` caps the tokens the run may spend before `MaxTokensExceeded`; `timeout` caps
        the run's wall-clock seconds before `RunTimeout`. All three are optional ceilings, and
        all three end the run as `Run(status="error")`.
        """
        cls = type(self)
        if cls is Agent:
            raise UserError("Agent must be subclassed, e.g. `class MyAgent(Agent): name = ...`")
        _reject_unknown_settings(cls)

        passed = locals()
        declared = all(passed[name] is _UNSET for name in _AGENT_FIELDS)
        setting = _settings(cls, passed)

        name = setting["name"]
        if name is _UNSET:
            raise UserError(f"{cls.__name__}(...) is missing the required 'name'")
        instructions = setting["instructions"]
        if instructions is _UNSET:
            instructions = _prompt(cls, name)

        self.name: str = name
        self.instructions: str | Callable[..., Any] | None = instructions
        self.model: str | Any = setting["model"]
        self.model_settings: ModelSettings | None = setting["model_settings"]
        self.tools: Sequence[FunctionTool] = setting["tools"]
        self.subagents: Sequence[type[Agent] | Subagent] = setting["subagents"]
        self.guardrails: GuardrailsList = setting["guardrails"]
        self.mcp: Sequence[Any] = setting["mcp"]
        # Unannotated: a subclass declares these as the setting, `"auto"` say, not as the source.
        self.memory = setting["memory"]
        self.knowledge = setting["knowledge"]
        self.output_type: type | None = setting["output_type"]
        self.hooks: Hooks[Any] | None = setting["hooks"]
        self.compact: bool | Compactor | None = setting["compact"]
        self.max_turns: int = setting["max_turns"]
        self.max_tokens: int | None = setting["max_tokens"]
        self.timeout: float | None = setting["timeout"]

        self.history: list[ConversationItem] = []
        self.usage = Usage()
        self.last_usage = Usage()
        self._in_flight = 0
        self._shape = AgentShape.declared(self) if declared else AgentShape.of(self)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute, rebuilding the shape a run reads when it is one of the settings.

        Only reassigning `subagents` builds the agents below this one again.
        """
        if name not in _AGENT_FIELDS:
            super().__setattr__(name, value)
            return
        super().__setattr__(name, _stored(name, value))
        if "_shape" in vars(self):
            self._shape = AgentShape.of(self, None if name == "subagents" else self._shape)

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
        """A copy of this agent with empty per-run state, sharing its config and subagents.

        For runs that must not share a conversation, such as an eval's concurrent cases.
        """
        clone = copy.copy(self)
        clone.history = []
        clone.usage = Usage()
        clone.last_usage = Usage()
        clone._in_flight = 0
        clone._shape = replace(self._shape, agent=clone)
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
        user_id: str | None = None,
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

        `user_id` is who the run acts for, `RunContext.user_id` to its tools and delegates. It is
        the scope of every `memory` read and write the run makes; `None` is a scope of its own,
        shared by every run that passes none. Like `context`, a paused `RunState` carries it.

        Pass a `session` to persist conversation history there instead of on `self.history`; the
        session supplies prior turns automatically, so only the new `message` is sent as input,
        and `self.history` is left untouched. A paused run's `RunState` carries its session, so
        it resumes there with no `session=`.

        A `session` is normally the conversation's id: `session="ticket-42"` persists to
        whichever backend `RUNA_DATABASE_URL` names, so an app moves to Postgres without naming
        one here. Pass a `Session` instead to point at a store of your own.

        Returns a `Run`. A tool call needing approval pauses it (`status="paused"`). Nothing a run
        fails on propagates: a guardrail tripwire, `MaxTurnsExceeded`, a model error and an
        exception out of a `Hooks`/`Session`/`Compactor` (wrapped in `UnexpectedError`, its
        traceback logged) all come back as `status="error"`. Only `asyncio.CancelledError` and
        `OperatorError` raise. `self.history` only changes once a turn completes; usage is
        recorded to `self.last_usage` and `self.usage` either way.

        **One instance runs one conversation at a time**: two overlapping session-less runs raise
        `UserError`. Give each run its own `session`, or its own `Agent`.
        """
        session = _resolve_session(message, session, self.name)
        turn_input = _turn_input(message, self.history, session, context, user_id)
        with self._exclusive(session):
            # A run's own failure comes back as an `"error"` `Run`; only an `OperatorError` or a
            # `CancelledError` gets through, and those are the caller's to see.
            run = await _run_async(
                self,
                turn_input,
                context=context,
                user_id=user_id,
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
        user_id: str | None = None,
    ) -> Run:
        """Synchronous `run`, for callers not already inside an event loop.

        Runs on Runa's one background event loop, shared by every synchronous call in the process,
        so the connections a turn opens (a database pool, a provider's HTTP client, an MCP server
        and its `.stdio` subprocess) are still open for the next one, as they are under
        `await agent.run(...)`. Ctrl-C cancels the run and waits for it to unwind before raising.
        Works from a notebook's running loop too, though it blocks that loop until it returns; from
        inside a coroutine already running on Runa's loop it raises `UserError`, since it would
        wait on itself. The trace is written by the time this returns, where `run` only queues its
        export.
        """
        try:
            return _loop.run_sync(self.run(message, context, hooks, session, user_id))
        finally:
            flush_exports()

    def run_streamed(
        self,
        message: MessageContent | RunState,
        context: Any = None,
        hooks: Hooks[Any] | None = None,
        session: Session | str | None = None,
        user_id: str | None = None,
    ) -> RunStream:
        """Run a turn as a stream of events: the same run as `run`, with the same arguments.

        Iterate the returned `RunStream` for `StreamEvent`s (`TextDelta`, `ToolCalled`,
        `ToolOutput`, `AgentChanged`) as they arrive; once it ends,
        its `.run` holds the `Run` that `run` would have returned, paused, completed or errored.
        History and usage are recorded only once the stream is fully consumed, so a caller that
        stops iterating early leaves them unchanged.

        One instance still runs one conversation at a time: a stream shares `run`'s latch, so
        starting to consume a second session-less stream on the same instance while one is in
        flight raises `UserError` rather than interleaving the two histories.
        """
        session = _resolve_session(message, session, self.name)
        turn_input = _turn_input(message, self.history, session, context, user_id)

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
                        user_id=user_id,
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
"""Every name that configures an Agent: `Agent.__init__`'s keyword-only parameters."""

__all__ = ["Agent", "Run", "RunState", "Subagent"]
