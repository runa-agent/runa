"""Class-based Agent, built on Runa's own runtime (`runa.run_internal`).

`run`/`run_sync`/`run_streamed` are the only way to run an Agent, and `Run` the only thing they
return. The turn loop under them is this class's implementation, not a second entry point, so
every run gets an Agent's own wiring: its settings resolved once into the shape a run reads, its
trace's grouping, its refusal of two concurrent session-less runs.
"""

import asyncio
import difflib
import inspect
import re
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
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
from runa.run_internal.agent_shape import AgentShape, Resolved, resolve
from runa.run_internal.record import resume_session
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


def _turn(
    agent: Agent,
    message: MessageContent | RunState,
    session: Session | str | None,
    context: Any,
    user_id: str | None,
) -> tuple[Session | None, str | list[ConversationItem] | RunState]:
    """The session this turn of `agent`'s runs in, and the `input` the turn loop is given.

    A paused `RunState` is resumed as is, in the session it started in; what may be passed beside
    one is `resume_session`'s to say. Otherwise a string `session` is resolved through
    `runa.db.session` as `agent`'s, so the backend follows `RUNA_DATABASE_URL`, and a `Session`
    object is passed straight through.

    A list `message` goes through `runa.content.parts`. With no session, the message joins
    `agent.history`; with one, only the new turn is sent, a plain string as is and a multimodal
    one as a single user message.
    """
    if isinstance(message, RunState):
        return resume_session(message, session, context=context, user_id=user_id), message
    if isinstance(session, str):
        from runa import db

        session = db.session(session, agent.name)
    resolved = message if isinstance(message, str) else content.parts(message)
    if session is not None:
        return session, resolved if isinstance(resolved, str) else [user_message(resolved)]
    return None, [*agent.history, user_message(resolved)]


_CAMEL_CASE_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def snake_case(name: str) -> str:
    """`SupportAgent` -> `support_agent`: the casing every name in a generated app follows.

    An agent's `name`, and the module, prompt and eval dataset named after it, follow this rule,
    as do `runa generate`'s tool and guardrail names.
    """
    return _CAMEL_CASE_BOUNDARY.sub("_", name).lower()


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


def _settings(cls: type[Agent], passed: Mapping[str, Any]) -> dict[str, Any]:
    """Every setting as an agent of `cls` holds it: what was passed, else what `cls` declares.

    `passed` holds one entry per setting, `_UNSET` where none was passed, so a setting added to
    `__init__`'s signature is resolved with no edit here. A setting neither supplies is `None`;
    `name` is required, and `instructions` defaults to `cls`'s prompt file. A class attribute
    that isn't a setting, and a missing `name`, are each a `UserError`.
    """
    _reject_unknown_settings(cls)
    setting: dict[str, Any] = {}
    for name in _AGENT_FIELDS:
        absent = _UNSET if name in _SIGNALLED_WHEN_ABSENT else None
        value = passed[name]
        setting[name] = _stored(name, value if value is not _UNSET else getattr(cls, name, absent))
    if setting["name"] is _UNSET:
        raise UserError(f"{cls.__name__}(...) is missing the required 'name'")
    if setting["instructions"] is _UNSET:
        setting["instructions"] = _prompt(cls, setting["name"])
    return setting


_DECLARATIONS = "agent declarations"
"""The `_app` fact holding each class's resolved declaration, so a reset forgets them."""


def declaration(agent_cls: type[Agent]) -> Resolved:
    """What `agent_cls` declares, resolved as every run of an agent built without overrides uses it.

    Resolved once per class per process and shared, the very `memory` and `knowledge` sources
    included, so a surface reading a class (`runa serve` warming its corpus) needs no instance.
    A declaration no agent could be built from is a `UserError`, as constructing one would be.
    """
    declarations = _app.held(_DECLARATIONS, dict[type, Resolved])
    if agent_cls not in declarations:
        declared = _settings(agent_cls, dict.fromkeys(_AGENT_FIELDS, _UNSET))
        declarations.setdefault(agent_cls, resolve(declared))
    return declarations[agent_cls]


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
    """An agent, always subclassed: its settings declared on the class, or passed to `__init__`.

    `usage` accumulates token usage across every `run`/`run_sync`/`run_streamed` call made on
    this instance; `last_usage` holds just the most recent call's usage. Both are read from
    `RunContext.usage`, which the turn loop populates regardless of `hooks`.

    An instance carries one conversation (`history`, `usage`) at a time; see `_exclusive`. The
    settings are resolved when the agent is built, and again when one is reassigned. RUNA.md §1
    is the contract for each setting.
    """

    handoff = _Mode("handoff")
    h = handoff
    delegate = _Mode("delegate")
    d = delegate

    # Every setting, typed as an instance holds it once built. One without a value here is
    # absent from the class, so `_settings` reads a subclass's declaration or falls back.
    name: str
    instructions: str | Callable[..., Any] | None
    model: str | Any = DEFAULT_MODEL
    model_settings: ModelSettings | None
    tools: Sequence[FunctionTool]
    subagents: Sequence[type[Agent] | Subagent]
    guardrails: GuardrailsList
    mcp: Sequence[Any]
    output_type: type | None
    hooks: Hooks[Any] | None
    memory: MemoryLike | Literal["auto", "llm"] | None
    knowledge: KnowledgeLike | Literal["auto", "llm"] | None
    compact: bool | Compactor | None
    max_turns: int = 10
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
        """Build the agent from its settings, wiring up any subagents and guardrails.

        These parameters are the closed set of settings: each can be declared on the subclass or
        passed here. Passing one beats declaring it, and passing `None` means `None`. An unknown
        name is a `TypeError` here, and a `UserError` naming the likely setting when declared in
        a class body. Each is held under its own name as given (`agent.memory` is `"auto"`, not
        the `Memory` it resolves to). What each setting means is RUNA.md's to say (§1, §7, §8).
        """
        cls = type(self)
        if cls is Agent:
            raise UserError("Agent must be subclassed, e.g. `class MyAgent(Agent): name = ...`")
        passed = locals()
        declared = all(passed[name] is _UNSET for name in _AGENT_FIELDS)
        setting = _settings(cls, passed)
        vars(self).update(setting)
        self._start_conversation()
        self._shape = AgentShape.of(self, declaration(cls) if declared else resolve(setting))

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute, building the shape a run reads again when it is one of the settings.

        The agents below this one are built again with it, as when the agent was constructed.
        """
        if name not in _AGENT_FIELDS:
            super().__setattr__(name, value)
            return
        super().__setattr__(name, _stored(name, value))
        if "_shape" in vars(self):
            settings = {setting: getattr(self, setting) for setting in _AGENT_FIELDS}
            self._shape = AgentShape.of(self, resolve(settings))

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

    def __copy__(self) -> Agent:
        """Another conversation with this agent's settings: `copy.copy(agent)`.

        The copy shares the settings' values (its tools and model) and starts with empty
        `history` and `usage`, so it can run beside the original. Its subagent graph is its own, so
        a subagent handing back reaches the copy, never the original.
        """
        clone = type(self).__new__(type(self))
        vars(clone).update(vars(self))
        clone._start_conversation()
        clone._shape = AgentShape.of(clone, self._shape)
        return clone

    def _start_conversation(self) -> None:
        """Give this instance the empty state one conversation owns."""
        self.history: list[ConversationItem] = []
        self.usage = Usage()
        self.last_usage = Usage()
        self._in_flight = 0

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
        """Run one user message asynchronously, appending the exchange to the conversation history.

        `message` is plain text, or a list for a multimodal message: a bare string is text, or
        an image when it is an image URL or a `data:image/...` URI; a local image is a
        `Path("cat.jpg")`, never a bare string, since the list may carry a user's own words (see
        `content.parts`). Build a part explicitly with `content.text(...)`/`content.image(...)`
        when a string doesn't have a recognizable image extension. It can also be the `RunState`
        of a paused `Run`, once its interruptions are approved or rejected, to resume that run.

        It is always one user message, never a transcript: a list of `{"role": ...}` messages raises
        `TypeError`. Start from an earlier conversation by setting `self.history` directly, or by
        seeding the `session` with `add_items` before the first run.

        `context` is available to a single-argument `instructions` callable (and to tools,
        guardrails, etc.) as-is; it is never sent to the model. `hooks` receives lifecycle
        callbacks (`on_agent_start`, `on_tool_end`, etc.); it defaults to `LoggingHooks`.

        `user_id` is who the run acts for, `RunContext.user_id` to its tools and delegates. It is
        the scope of every `memory` read and write the run makes; `None` is a scope of its own,
        shared by every run that passes none.

        Pass a `session` to persist conversation history there instead of on `self.history`; the
        session supplies prior turns automatically, so only the new `message` is sent as input,
        and `self.history` is left untouched. A `session` is normally the conversation's id,
        resolved through `runa.db.session`; pass a `Session` instead to point at a store of your
        own. A `RunState` carries its own `context`, `user_id` and session (see `RunState`).

        Returns a `Run`, whatever the run fails on; what still raises, and which calls are refused
        with `UserError` before a run starts, is RUNA.md's Agent section. `self.history` only
        changes once a turn completes; usage is recorded to `self.last_usage` and `self.usage`
        either way.
        """
        session, turn_input = _turn(self, message, session, context, user_id)
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
        """Run one user message as a stream of events: the same run as `run`, with its arguments.

        Iterate the returned `RunStream` for `StreamEvent`s (`TextDelta`, `ToolCalled`,
        `ToolOutput`, `AgentChanged`) as they arrive; once it ends,
        its `.run` holds the `Run` that `run` would have returned, paused, completed or errored.
        History and usage are recorded only once the stream is fully consumed, so a caller that
        stops iterating early leaves them unchanged. The instance is held (see `_exclusive`) only
        while the stream is consumed.
        """
        session, turn_input = _turn(self, message, session, context, user_id)

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
