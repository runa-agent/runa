"""agent_shape.py: `AgentShape`, an agent as the runtime reads it.

An `Agent` holds its settings as declared (`subagents`, `mcp`, `memory = "auto"`), and its shape
is the one place they are translated into what a run uses: the subagent graph built, `memory` and
`knowledge` resolved to a source and the model's search tools. The agent builds its shape when it
is constructed, and again whenever one of its settings is reassigned, so the two always agree and
a bad value is a `UserError` at the line that set it. What a class declares is translated once per
process (`runa.agent.declaration`) and shared by every instance built without overrides, so
building one costs only its subagent graph, whose agents are still its own. The turn loop reads an
agent's settings from its shape and nowhere else: `shape.agent` is the identity handed to user
code (a hook, an `instructions` callable, a pause), never read for a setting.

`shape_of` is how anything outside the agent reads one, and `subagent_edges`/`reachable` walk the
graph the shape builds, so a third subagent mode is a change in this module alone.

Two things wait for a run: the tools an agent's MCP servers list (`with_mcp_tools`, which may
start a subprocess) and the model client (`resolve_model`, which needs the provider's API key).
"""

import inspect
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any

from runa._models import Model, ModelProvider, ToolSchema, check_model_name
from runa._types import ModelSettings, RunContext
from runa.compact import Compactor, resolve_compactor
from runa.exceptions import UserError
from runa.guardrail import BoundGuardrail, Phase, flatten_guardrails
from runa.handoff import Delegate, Handoff, Mode, Subagent
from runa.knowledge import Knowledge, KnowledgeLike
from runa.lifecycle import Hooks
from runa.memory import Memory, MemoryLike
from runa.tool import FunctionTool

if TYPE_CHECKING:
    from runa.agent import Agent

_MODEL_PROVIDER = ModelProvider()

_under_construction: ContextVar[dict[type[Agent], Agent] | None] = ContextVar(
    "_under_construction", default=None
)


@contextmanager
def _agent_graph(agent: Agent) -> Iterator[Callable[[type[Agent]], Agent]]:
    """Yield the one way `agent`'s shape builds a subagent: each class once per graph.

    A class listed twice below one root is built once, and a cycle (a specialist handing back to
    triage) closes onto `agent` itself. Construction is eager, so a broken subagent declaration
    fails where `subagents` is set. An agent is found again by its `name` when a paused run
    resumes, so two classes in one graph sharing a name is a `UserError`.
    """
    built = _under_construction.get()
    token = _under_construction.set(built := {}) if built is None else None
    try:
        for other in built.values():
            if other is not agent and other.name == agent.name:
                raise UserError(
                    f"{type(other).__name__} and {type(agent).__name__} are both named "
                    f"{agent.name!r} in one agent graph, so a paused run could resume the "
                    f"wrong one. Give each agent a name of its own."
                )
        built[type(agent)] = agent
        yield lambda cls: built[cls] if cls in built else cls()
    finally:
        if token is not None:
            _under_construction.reset(token)


def _subagent_graph(agent: Agent) -> tuple[list[Handoff], list[Delegate]]:
    """`agent.subagents` built: each entry bare (auto, so both modes) or `.handoff`/`.delegate`."""
    handoffs: list[Handoff] = []
    delegates: list[Delegate] = []
    with _agent_graph(agent) as build:
        for sub in agent.subagents:
            if isinstance(sub, Subagent):
                target = build(sub.agent)
                if sub.mode == "handoff":
                    handoffs.append(Handoff.from_agent(target, sub.tool_name, sub.tool_description))
                else:
                    delegates.append(
                        Delegate.from_agent(
                            target,
                            sub.tool_name,
                            sub.tool_description,
                            needs_approval=sub.needs_approval,
                            guardrails=sub.guardrails,
                        )
                    )
            else:
                target = build(sub)
                handoffs.append(Handoff.from_agent(target))
                delegates.append(Delegate.from_agent(target))
    return handoffs, delegates


def _retrieval(setting: Any, cls: type[Memory] | type[Knowledge]) -> tuple[Any, list[FunctionTool]]:
    """Resolve a `memory`/`knowledge` setting to the source a run consults and the model's tools.

    `"auto"` builds a default instance, `"llm"` is no source but a search tool for the model, and
    any non-string (`None`, or a `MemoryLike`/`KnowledgeLike` object) is the source itself. Any
    other string is a `UserError`.
    """
    if not isinstance(setting, str):
        return setting, []
    if setting == "auto":
        return cls(), []
    if setting == "llm":
        return None, [cls()._as_tool()]
    raise UserError(
        f"{cls.__name__.lower()} must be 'auto', 'llm', a {cls.__name__}(...)-shaped object, "
        f"or None, got {setting!r}"
    )


def _adapt_instructions(instructions: Any) -> Any:
    """Let `instructions` be a `(context) -> str` callable instead of the loop's 2-arg shape.

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


def _checked_model(model: Any) -> Any:
    """Check `model` names a model Runa can route, or is one, without building it yet."""
    if isinstance(model, str):
        check_model_name(model)
    elif model is not None and not hasattr(model, "get_response"):
        raise UserError(
            f"model must be a model name, a Model instance, or None for Runa's default, "
            f"got {model!r}"
        )
    return model


def _checked_tools(tools: tuple[Any, ...]) -> tuple[FunctionTool, ...]:
    """Check every `tools` entry is a `@tool`, which is the one shape a tool takes."""
    for entry in tools:
        if not isinstance(entry, FunctionTool):
            raise UserError(f"tools entries must be @tool functions, got {entry!r}")
    return tools


def _checked_mcp(servers: tuple[Any, ...]) -> tuple[Any, ...]:
    """Check every `mcp` entry is a server, as `MCPServer(...).http`/`.stdio` returns one."""
    for entry in servers:
        if not hasattr(entry, "list_tools"):
            raise UserError(
                f"mcp entries must be servers built as MCPServer(...).http(url) or "
                f"MCPServer(...).stdio(command), got {entry!r}"
            )
    return servers


def _checked_unique(agent_name: str, names: Iterable[str]) -> None:
    """Refuse two callables offered under one name: a call could only ever reach one of them.

    The name is what the model calls, what a call is dispatched by and what a sticky approval is
    keyed by, so a duplicate would leave all but one of them unreachable, silently.
    """
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise UserError(
            f"{agent_name} offers the model more than one tool named {duplicates[0]!r}, so only "
            f"one of them could ever be called. Rename the @tool, pass tool_name= to the "
            f"subagent's .handoff/.delegate, or give each MCP server its own name=."
        )


@dataclass(frozen=True)
class RunPolicy:
    """The settings a run holds to for its whole length, so a handoff never changes them.

    A run reads its starting agent's: see RUNA.md, Human Approval.
    """

    compactor: Compactor | None
    """What trims the conversation, or `None` when nothing does."""
    memory: MemoryLike | None
    """The source a run consults before each turn: `None` for `"llm"`, whose search is a tool."""
    knowledge: KnowledgeLike | None
    """The same as `memory`, for the `knowledge` setting."""
    max_turns: int
    max_tokens: int | None
    timeout: float | None


@dataclass(frozen=True)
class Resolved:
    """Every setting of an agent but `subagents`, each in the form the turn loop uses it.

    Holds no agent, so every instance of a class that declares the same settings shares one.
    """

    name: str
    instructions: str | Callable[[RunContext[Any], Agent], Any] | None
    """A string, or a callable the loop calls with `(run_context, agent)`."""
    model: str | Model
    model_settings: ModelSettings
    output_type: type | None
    hooks: Hooks[Any] | None
    """Its agent-scoped hooks, which a run fires beside the run's own."""
    tools: tuple[FunctionTool, ...]
    """Its `@tool`s and memory's or knowledge's search; `with_mcp_tools` adds its servers'."""
    mcp_servers: tuple[Any, ...]
    """Its `mcp` setting."""
    guardrails: dict[Phase, list[BoundGuardrail]]
    """By the phase each one runs in, each phase's in list order."""
    policy: RunPolicy
    """What a run it starts holds to; a run handed to it keeps its starting agent's."""


def resolve(settings: Mapping[str, Any]) -> Resolved:
    """`settings`, one value per `Agent` setting as an agent holds it, translated for a run.

    A value no run could use is a `UserError`. Every field is named here, so a setting added to
    `Agent` and to `Resolved` is a `TypeError` until it is translated here too.
    """
    memory, memory_tools = _retrieval(settings["memory"], Memory)
    knowledge, knowledge_tools = _retrieval(settings["knowledge"], Knowledge)
    return Resolved(
        name=settings["name"],
        instructions=_adapt_instructions(settings["instructions"]),
        model=_checked_model(settings["model"]),
        model_settings=settings["model_settings"] or ModelSettings(),
        output_type=settings["output_type"],
        hooks=settings["hooks"],
        tools=(*_checked_tools(settings["tools"]), *memory_tools, *knowledge_tools),
        mcp_servers=_checked_mcp(settings["mcp"]),
        guardrails=flatten_guardrails(settings["guardrails"]),
        policy=RunPolicy(
            compactor=resolve_compactor(settings["compact"]),
            memory=memory,
            knowledge=knowledge,
            max_turns=settings["max_turns"],
            max_tokens=settings["max_tokens"],
            timeout=settings["timeout"],
        ),
    )


@dataclass(frozen=True)
class AgentShape(Resolved):
    """One agent's settings as the turn loop uses them, wired into its agent graph.

    Every tool, delegate and handoff it offers has a name no other one has: a shape where two
    would share one is a `UserError` instead.
    """

    agent: Agent
    handoffs: dict[str, Handoff]
    """Its `subagents` wired up as handoffs, keyed by the tool name that triggers each."""
    delegates: dict[str, Delegate]
    """Its `subagents` wired up as delegates, keyed by the tool name that triggers each."""

    @classmethod
    def of(cls, agent: Agent, resolved: Resolved) -> AgentShape:
        """`agent`'s shape: its settings as `resolved`, wired into a subagent graph built for it."""
        handoffs, delegates = _subagent_graph(agent)
        _checked_unique(
            agent.name,
            (
                *(tool.name for tool in resolved.tools),
                *(delegate.tool.name for delegate in delegates),
                *(handoff.tool_name for handoff in handoffs),
            ),
        )
        return cls(
            **{field.name: getattr(resolved, field.name) for field in fields(Resolved)},
            agent=agent,
            handoffs={handoff.tool_name: handoff for handoff in handoffs},
            delegates={delegate.tool.name: delegate for delegate in delegates},
        )

    async def with_mcp_tools(self) -> AgentShape:
        """This shape with the tools its MCP servers list added to `tools`, for one run."""
        if not self.mcp_servers:
            return self
        listed = [tool for server in self.mcp_servers for tool in await server.list_tools()]
        shape = replace(self, tools=(*self.tools, *listed))
        _checked_unique(self.name, (schema.name for schema in shape.offered))
        return shape

    @cached_property
    def tools_by_name(self) -> dict[str, FunctionTool]:
        """Every tool, its MCP servers' included once listed, by the name the model calls it."""
        return {tool.name: tool for tool in self.tools}

    @cached_property
    def offered(self) -> list[ToolSchema]:
        """Every tool, delegate and handoff, in the order the model is offered them.

        A handoff takes no structured input from the model: calling it is itself the signal to
        switch agents, so its schema's `parameters` is always an empty object.
        """
        no_input = {"type": "object", "properties": {}}
        return [
            *(
                ToolSchema(tool.name, tool.description or "", tool.params_json_schema or no_input)
                for tool in (*self.tools, *(delegate.tool for delegate in self.delegates.values()))
            ),
            *(
                ToolSchema(handoff.tool_name, handoff.tool_description or "", no_input)
                for handoff in self.handoffs.values()
            ),
        ]

    def resolve_model(self) -> Model:
        """This agent's `model`: a string names a provider's model, anything else is one.

        A name resolves its provider's backend, which reads the API key, so it waits for a run:
        an `Agent` is constructible without one.
        """
        model = self.model
        return _MODEL_PROVIDER.get_model(model) if isinstance(model, str) else model


def shape_of(agent: Agent) -> AgentShape:
    """`agent`'s shape as resolved when it was built or last had a setting reassigned."""
    return agent._shape


def subagent_edges(agent: Agent) -> Iterator[tuple[Mode, Agent]]:
    """`agent`'s outgoing edges in the agent graph: one `(mode, target)` per subagent of it.

    Handoffs first, then delegates, each in declaration order. An `auto`-mode subagent is wired
    up as both, so it is yielded twice, once per mode -- which is what it is: two tools the model
    picks between.
    """
    shape = shape_of(agent)
    for handoff in shape.handoffs.values():
        yield "handoff", handoff.agent
    for delegate in shape.delegates.values():
        yield "delegate", delegate.agent


def reachable(agent: Agent) -> Iterator[Agent]:
    """`agent`, then every agent reachable from it through handoffs and delegates, breadth-first.

    Each agent once, by identity, so a cycle (two agents listing each other, an agent listing
    itself) terminates rather than recursing forever.
    """
    seen: set[int] = set()
    queue: list[Agent] = [agent]
    while queue:
        candidate = queue.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate
        queue.extend(target for _, target in subagent_edges(candidate))


__all__ = [
    "AgentShape",
    "Resolved",
    "RunPolicy",
    "reachable",
    "resolve",
    "shape_of",
    "subagent_edges",
]
