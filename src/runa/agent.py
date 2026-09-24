"""Class-based Agent, built on Runa's own runtime (`runa.runner`/`runa.run_internal`)."""

import inspect
import re
from collections.abc import AsyncIterator, Iterable
from dataclasses import MISSING, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from runa import content
from runa._models import ModelProvider
from runa._types import MessageContent, ModelSettings, RunContextWrapper, TResponseInputItem, Usage
from runa.exceptions import RunaError, UserError
from runa.guardrail import flatten_agent_guardrails
from runa.handoff import agent_as_tool
from runa.knowledge import Knowledge
from runa.lifecycle import LoggingRunHooks, RunHooks
from runa.memory import Memory
from runa.run import Run
from runa.run_config import RunConfig
from runa.run_state import RunState
from runa.runner import Runner
from runa.session import SessionABC
from runa.stream_events import StreamEvent
from runa.tool import FunctionTool

if TYPE_CHECKING:
    from graphviz import Source

    from runa.eval.case import Case
    from runa.eval.report import Report

_AGENT_FIELDS = (
    "name",
    "instructions",
    "model",
    "model_settings",
    "tools",
    "mcp_servers",
    "output_type",
    "hooks",
    "memory",
    "knowledge",
    "compact",
)

_MODEL_PROVIDER = ModelProvider()


def _default_hooks() -> RunHooks[Any]:
    """Build the default `hooks` for a run: `LoggingRunHooks`."""
    return LoggingRunHooks()


def _usage_from_exception(exc: RunaError) -> Usage:
    """Recover whatever token usage a run accrued before a `RunaError` stopped it."""
    return exc.run_data.context_wrapper.usage if exc.run_data else Usage()


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


def _turn_input(
    message: MessageContent, history: list[TResponseInputItem], session: SessionABC | None
) -> str | list[TResponseInputItem]:
    """Build the `input` for `Runner.run`/`run_sync` from this turn's `message`.

    A list `message` goes through `runa.content.parts` first, auto-detecting each bare string
    as text or an image; a plain string is left untouched. With no `session`, the result joins
    `history` as a new user message. With a `session`, only the new turn is ever sent (prior
    turns come back from the session itself): a plain string passes straight through, a
    multimodal one is wrapped in a single-item message list instead, since `Runner.run`'s
    session path only wraps a bare string into `{"role": "user", ...}` itself.
    """
    resolved = message if isinstance(message, str) else content.parts(message)
    if session is not None:
        return resolved if isinstance(resolved, str) else [{"role": "user", "content": resolved}]
    return [*history, {"role": "user", "content": resolved}]


_CAMEL_CASE_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _snake_case(name: str) -> str:
    return _CAMEL_CASE_BOUNDARY.sub("_", name).lower()


_NO_SOURCE_FILE = (TypeError, OSError)

_PROMPT_TEMPLATE = """TODO: write the prompt {name} uses.
"""


def _load_prompt(cls: type, name: str) -> str | None:
    """Read `<name>.md` from the `prompts/` directory next to `cls`'s `app/agents/` module.

    Mirrors `runa generate prompt`'s naming: `app/prompts/<snake_case(name)>.md`, a sibling of
    the `agents/` directory the subclass is defined in. Missing, it's created from
    `_PROMPT_TEMPLATE`, the same stub `runa generate prompt` (`cli/generate.py`, which imports
    this constant rather than duplicating it) would write, so a fresh agent always has a prompt
    file ready to edit instead of silently running with empty instructions.

    Returns `None` (leaving `instructions` empty) when `cls` has no source file (e.g. defined at
    a REPL) or its module doesn't live in an `agents/` directory, nothing is ever created outside
    the one location `runa new`'s convention establishes for prompts.
    """
    try:
        module_file = Path(inspect.getfile(cls)).resolve()
    except _NO_SOURCE_FILE:
        return None
    if module_file.parent.name != "agents":
        return None
    stem = _snake_case(name)
    prompts_dir = module_file.parent.parent / "prompts"
    prompt_file = prompts_dir / f"{stem}.md"
    if not prompt_file.is_file():
        prompts_dir.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(_PROMPT_TEMPLATE.format(name=stem))
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
    `RunContextWrapper.usage`, which `Runner` populates regardless of `hooks`.
    """

    handoff = _Mode("handoff")
    h = handoff
    delegate = _Mode("delegate")
    d = delegate
    model = "gpt-5.4-nano"

    def __init__(self, **kwargs: Any) -> None:
        """Build config from class attributes and wire up any subagents and guardrails.

        `mcp=[...]` (as a constructor kwarg, or a `mcp` class attribute) is sugar for
        `mcp_servers=[...]`; both are merged into `mcp_servers` if given together.

        `memory` opts this agent into long-term memory, one of:
          - `"auto"` (or a `Memory(...)` instance, or any object shaped like `runa.memory`'s
            `MemoryLike`): `Runner` retrieves relevant memories before each run and persists new
            ones after -- no manual `memory.search`/`.remember` calls.
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
        """
        if type(self) is Agent:
            raise TypeError("Agent must be subclassed, e.g. `class MyAgent(Agent): name = ...`")

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
        mcp_servers = [
            *(kwargs.get("mcp_servers") or []),
            *(kwargs.pop("mcp", None) or getattr(type(self), "mcp", [])),
        ]
        for sub in _flatten_subagents(getattr(type(self), "subagents", [])):
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

        new_input_guardrails, new_output_guardrails = flatten_agent_guardrails(
            getattr(type(self), "guardrails", [])
        )

        self.memory = _resolve_retrieval_setting(kwargs.get("memory"), Memory, tools)
        self.knowledge = _resolve_retrieval_setting(kwargs.get("knowledge"), Knowledge, tools)

        self.name: str = kwargs["name"]
        self.instructions = _adapt_instructions(kwargs.get("instructions"))
        self.model: str | Any = kwargs.get("model", "gpt-5.4-nano")
        self.model_settings: ModelSettings = kwargs.get("model_settings") or ModelSettings()
        self.tools: list[FunctionTool] = tools
        self.handoffs: list[Any] = handoffs
        self.mcp_servers: list[Any] = mcp_servers
        self.input_guardrails = new_input_guardrails
        self.output_guardrails = new_output_guardrails
        self.output_type: type | None = kwargs.get("output_type")
        self.hooks = kwargs.get("hooks")
        self.compact: bool = kwargs.get("compact", False)

        self.history: list[TResponseInputItem] = []
        self.usage = Usage()
        self.last_usage = Usage()

    def as_tool(self, tool_name: str | None, tool_description: str | None) -> FunctionTool:
        """Wrap this agent as a tool another agent can call; see `runa.handoff.agent_as_tool`."""
        return agent_as_tool(self, tool_name, tool_description)

    @property
    def graph(self) -> Source:
        """Render this agent, and its tools/handoffs, as a Graphviz diagram.

        Displays inline in Jupyter; call `.render(filename)` to save it, or `.source` for the
        raw DOT text. Actually rendering an image needs the system Graphviz `dot` binary.
        """
        from runa._graph import draw_graph

        return draw_graph(self)

    def _run_config(self, session: SessionABC | None) -> RunConfig:
        return RunConfig(
            model_provider=_MODEL_PROVIDER,
            workflow_name=type(self).__name__,
            group_id=getattr(session, "session_id", None),
        )

    async def run(
        self,
        message: MessageContent,
        context: Any = None,
        hooks: RunHooks[Any] | None = None,
        session: SessionABC | None = None,
        *,
        _context_wrapper: RunContextWrapper[Any] | None = None,
    ) -> Run:
        """Run a turn asynchronously, appending it to the conversation history.

        `message` is plain text, or a list for a multimodal message: bare strings are
        auto-detected as text or an image by extension (`"cat.jpg"`, a URL, a `data:image/...`
        URI), or build a part explicitly with `content.text(...)`/`content.image(...)` when a
        string doesn't have a recognizable image extension.

        `context` is available to a single-argument `instructions` callable (and to tools,
        guardrails, etc.) as-is; it is never sent to the model. `hooks` receives lifecycle
        callbacks (`on_agent_start`, `on_tool_end`, etc.) from `Runner`; it defaults to
        `LoggingRunHooks`.

        Pass a `session` (e.g. `SQLiteSession`) to persist conversation history there instead
        of on `self.history`; the session supplies prior turns automatically, so only the new
        `message` is sent as input, and `self.history` is left untouched.

        Returns a `Run` exposing `.output`, `.trace`, `.usage`, `.status`, and `.error`. A
        guardrail tripwire, `MaxTurnsExceeded`, or another `RunaError` is caught and reported as
        `status="error"` instead of propagating; `self.history` is left unchanged when that
        happens, since the turn never completed.

        Token usage for this call is recorded to `self.last_usage` and accumulated into
        `self.usage`, regardless of `session`, `hooks`, or whether the run errored.

        `_context_wrapper` is internal, used by `agent_as_tool`'s nested delegate calls to share
        a forked `RunContextWrapper` with the caller instead of building a fresh one; `context`
        is ignored when it's given. Don't pass it directly.
        """
        turn_input = _turn_input(message, self.history, session)
        run_hooks = hooks or _default_hooks()
        try:
            result = await Runner.run(
                self,
                turn_input,
                context=context,
                hooks=run_hooks,
                run_config=self._run_config(session),
                session=session,
                _context_wrapper=_context_wrapper,
            )
        except RunaError as exc:
            self.last_usage = _usage_from_exception(exc)
            self.usage.add(self.last_usage)
            return Run(
                output=None,
                trace=exc.run_data.trace if exc.run_data else None,
                usage=self.last_usage,
                status="error",
                error=str(exc),
            )
        self.last_usage = result.context_wrapper.usage
        self.usage.add(self.last_usage)
        if session is None and not result.interruptions:
            self.history = result.to_input_list()
        return Run(output=result.final_output, trace=result.trace, usage=self.last_usage)

    async def evaluate(
        self,
        dataset: Iterable["Case"],  # noqa: UP037 -- Case is TYPE_CHECKING-only, must stay quoted
        *,
        judge: str | None = None,
        threshold: float | None = None,
        thresholds: dict[str, float] | None = None,
        concurrency: int = 8,
    ) -> "Report":  # noqa: UP037 -- Report is TYPE_CHECKING-only, must stay quoted
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

    async def run_streamed(
        self,
        message: MessageContent,
        context: Any = None,
        hooks: RunHooks[Any] | None = None,
        session: SessionABC | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run a turn as a stream of events, appending it to the conversation history.

        `message` is plain text, or a list for a multimodal message: bare strings are
        auto-detected as text or an image by extension (`"cat.jpg"`, a URL, a `data:image/...`
        URI), or build a part explicitly with `content.text(...)`/`content.image(...)` when a
        string doesn't have a recognizable image extension.

        Yields `StreamEvent`s (`raw_response_event`, `run_item_stream_event`,
        `agent_updated_stream_event`) as they arrive. `context`, `hooks` and `session` behave as
        in `run`/`run_sync`. The conversation history is updated only once the stream is fully
        consumed, so a caller that stops iterating early leaves `self.history` unchanged.

        Token usage for this call is recorded to `self.last_usage` and accumulated into
        `self.usage` once the stream is fully consumed; a caller that stops iterating early
        leaves both unchanged, same as `self.history`.
        """
        result = Runner.run_streamed(
            self,
            _turn_input(message, self.history, session),
            context=context,
            hooks=hooks or _default_hooks(),
            run_config=self._run_config(session),
            session=session,
        )
        async for event in result:
            yield event
        self.last_usage = result.context_wrapper.usage
        self.usage.add(self.last_usage)
        if session is None:
            self.history = result.to_input_list()

    def run_sync(
        self,
        message: MessageContent,
        context: Any = None,
        hooks: RunHooks[Any] | None = None,
        session: SessionABC | None = None,
    ) -> Run:
        """Run a turn synchronously, appending it to the conversation history.

        `message` is plain text, or a list for a multimodal message: bare strings are
        auto-detected as text or an image by extension (`"cat.jpg"`, a URL, a `data:image/...`
        URI), or build a part explicitly with `content.text(...)`/`content.image(...)` when a
        string doesn't have a recognizable image extension.

        `context` is available to a single-argument `instructions` callable (and to tools,
        guardrails, etc.) as-is; it is never sent to the model. `hooks` receives lifecycle
        callbacks (`on_agent_start`, `on_tool_end`, etc.) from `Runner`; it defaults to
        `LoggingRunHooks`.

        Pass a `session` (e.g. `SQLiteSession`) to persist conversation history there instead
        of on `self.history`; the session supplies prior turns automatically, so only the new
        `message` is sent as input, and `self.history` is left untouched.

        Returns a `Run` exposing `.output`, `.trace`, `.usage`, `.status`, and `.error`. A
        guardrail tripwire, `MaxTurnsExceeded`, or another `RunaError` is caught and reported as
        `status="error"` instead of propagating; `self.history` is left unchanged when that
        happens, since the turn never completed.

        Token usage for this call is recorded to `self.last_usage` and accumulated into
        `self.usage`, regardless of `session`, `hooks`, or whether the run errored.
        """
        turn_input = _turn_input(message, self.history, session)
        run_hooks = hooks or _default_hooks()
        try:
            result = Runner.run_sync(
                self,
                turn_input,
                context=context,
                hooks=run_hooks,
                run_config=self._run_config(session),
                session=session,
            )
        except RunaError as exc:
            self.last_usage = _usage_from_exception(exc)
            self.usage.add(self.last_usage)
            return Run(
                output=None,
                trace=exc.run_data.trace if exc.run_data else None,
                usage=self.last_usage,
                status="error",
                error=str(exc),
            )
        self.last_usage = result.context_wrapper.usage
        self.usage.add(self.last_usage)
        if session is None and not result.interruptions:
            self.history = result.to_input_list()
        return Run(output=result.final_output, trace=result.trace, usage=self.last_usage)


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
