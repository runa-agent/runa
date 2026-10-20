"""agent_shape.py: `AgentShape`, an agent as the runtime reads it.

An `Agent` holds its settings as declared (`subagents`, `mcp`, `memory = "auto"`), and its shape
is the one place they are translated into what a run uses: the subagent graph built, `memory` and
`knowledge` resolved to a source and the model's search tools. The agent builds its shape when it
is constructed, and again whenever one of its settings is reassigned, so the two always agree and
a bad value is a `UserError` at the line that set it. The turn loop reads an agent's settings
from its shape and nowhere else: `shape.agent` is the identity handed to user code (a hook, an
`instructions` callable, a pause), never read for a setting.

Two things wait for a run: the tools an agent's MCP servers list (`with_mcp_tools`, which may
start a subprocess) and the model client (`resolve_model`, which needs the provider's API key).
"""

import inspect
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any

from runa._models import Model, ModelProvider, ToolSchema, check_model_name
from runa._types import ModelSettings, RunContext
from runa.compact import Compactor, resolve_compactor
from runa.exceptions import UserError
from runa.guardrail import BoundGuardrail, Phase, flatten_guardrails
from runa.handoff import Delegate, Handoff, Subagent
from runa.knowledge import Knowledge, KnowledgeLike
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
    """`agent.subagents` built: each entry bare (auto, so both modes) or `.handoff`/`.delegate`.

    Anything but a list (a dict grouping entries by mode, say) is a `UserError`.
    """
    declared = agent.subagents
    if not isinstance(declared, list | tuple):
        raise UserError(
            f"subagents must be a list, e.g. [Billing.handoff, Researcher.delegate, Helper], "
            f"got {type(declared).__name__}"
        )
    handoffs: list[Handoff] = []
    delegates: list[Delegate] = []
    with _agent_graph(agent) as build:
        for sub in declared:
            if isinstance(sub, Subagent):
                target = build(sub.agent)
                if sub.mode == "handoff":
                    handoffs.append(Handoff.from_agent(target, sub.tool_name, sub.tool_description))
                else:
                    delegates.append(
                        Delegate.from_agent(target, sub.tool_name, sub.tool_description)
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
class AgentShape:
    """One agent's settings, each in the form the turn loop uses it.

    Every tool, delegate and handoff it offers has a name no other one has: a shape where two
    would share one is a `UserError` instead.
    """

    agent: Agent
    name: str
    instructions: str | Callable[[RunContext[Any], Agent], Any] | None
    """A string, or a callable the loop calls with `(run_context, agent)`."""
    model: str | Model
    model_settings: ModelSettings
    output_type: type | None
    tools: tuple[FunctionTool, ...]
    """Its `@tool`s and memory's or knowledge's search; `with_mcp_tools` adds its servers'."""
    mcp_servers: tuple[Any, ...]
    """Its `mcp` setting."""
    handoffs: dict[str, Handoff]
    """Its `subagents` wired up as handoffs, keyed by the tool name that triggers each."""
    delegates: dict[str, Delegate]
    """Its `subagents` wired up as delegates, keyed by the tool name that triggers each."""
    guardrails: dict[Phase, list[BoundGuardrail]]
    """By the phase each one runs in, each phase's in list order."""
    compactor: Compactor | None
    """What trims the conversation, or `None` when nothing does."""
    memory: MemoryLike | None
    """The source a run consults before each turn: `None` for `"llm"`, whose search is a tool."""
    knowledge: KnowledgeLike | None
    """The same as `memory`, for the `knowledge` setting."""
    max_turns: int
    max_tokens: int | None
    timeout: float | None

    @classmethod
    def of(cls, agent: Agent, previous: AgentShape | None = None) -> AgentShape:
        """`agent`'s shape, translated from its settings; a value no run could use is a `UserError`.

        `previous` is a shape of the same agent graph whose subagents are kept rather than built
        again, given when `subagents` isn't what changed, so the agents below this one keep their
        identity.
        """
        handoffs, delegates = (
            _subagent_graph(agent)
            if previous is None
            else ([*previous.handoffs.values()], [*previous.delegates.values()])
        )
        memory, memory_tools = _retrieval(agent.memory, Memory)
        knowledge, knowledge_tools = _retrieval(agent.knowledge, Knowledge)
        tools = (*agent.tools, *memory_tools, *knowledge_tools)
        _checked_unique(
            agent.name,
            (
                *(tool.name for tool in tools),
                *(delegate.tool_name for delegate in delegates),
                *(handoff.tool_name for handoff in handoffs),
            ),
        )
        return cls(
            agent=agent,
            name=agent.name,
            instructions=_adapt_instructions(agent.instructions),
            model=_checked_model(agent.model),
            model_settings=agent.model_settings or ModelSettings(),
            output_type=agent.output_type,
            tools=tools,
            mcp_servers=tuple(agent.mcp),
            handoffs={handoff.tool_name: handoff for handoff in handoffs},
            delegates={delegate.tool_name: delegate for delegate in delegates},
            guardrails=flatten_guardrails(agent.guardrails),
            compactor=resolve_compactor(agent.compact),
            memory=memory,
            knowledge=knowledge,
            max_turns=agent.max_turns,
            max_tokens=agent.max_tokens,
            timeout=agent.timeout,
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
    def offered(self) -> list[ToolSchema]:
        """Every tool, delegate and handoff, in the order the model is offered them.

        A handoff takes no structured input from the model: calling it is itself the signal to
        switch agents, so its schema's `parameters` is always an empty object.
        """
        no_input = {"type": "object", "properties": {}}
        return [
            *(
                ToolSchema(tool.name, tool.description or "", tool.params_json_schema or no_input)
                for tool in self.tools
            ),
            *(
                ToolSchema(
                    delegate.tool_name, delegate.tool_description, delegate.params_json_schema
                )
                for delegate in self.delegates.values()
            ),
            *(
                ToolSchema(handoff.tool_name, handoff.tool_description or "", no_input)
                for handoff in self.handoffs.values()
            ),
        ]

    def find_tool(self, name: str) -> FunctionTool | None:
        """The `FunctionTool` this agent calls `name`, or `None` if it declares no such tool."""
        for candidate in self.tools:
            if isinstance(candidate, FunctionTool) and candidate.name == name:
                return candidate
        return None

    def resolve_model(self) -> Model:
        """This agent's `model`: a string names a provider's model, anything else is one.

        A name builds its provider's client, which is a `UserError` when the API key is unset, so
        it waits for a run: an `Agent` is constructible without one.
        """
        model = self.model
        return _MODEL_PROVIDER.get_model(model) if isinstance(model, str) else model


__all__ = ["AgentShape"]
