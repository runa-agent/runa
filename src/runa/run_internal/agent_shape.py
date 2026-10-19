"""agent_shape.py: `AgentShape`, an agent as the runtime reads it.

An `Agent` builds its shape when it is constructed, and again whenever one of its settings is
reassigned, so the two always agree and a bad value is a `UserError` at the line that set it. The
turn loop reads an agent's settings from its shape and nowhere else: `shape.agent` is the identity
handed to user code (a hook, an `instructions` callable, a pause), never read for a setting.

Two things wait for a run: the tools an agent's MCP servers list (`with_mcp_tools`, which may
start a subprocess) and the model client (`resolve_model`, which needs the provider's API key).
"""

import inspect
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any

from runa._models import Model, ModelProvider, ToolSchema, check_model_name
from runa._types import ModelSettings, RunContext
from runa.compact import Compactor, resolve_compactor
from runa.exceptions import UserError
from runa.guardrail import BoundGuardrail, Phase, flatten_guardrails
from runa.handoff import Delegate, Handoff
from runa.tool import FunctionTool

if TYPE_CHECKING:
    from runa.agent import Agent
    from runa.knowledge import KnowledgeLike
    from runa.memory import MemoryLike

_MODEL_PROVIDER = ModelProvider()


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
    handoffs: dict[str, Handoff]
    """Keyed by the tool name that triggers each."""
    delegates: dict[str, Delegate]
    """Keyed by the tool name that triggers each."""
    guardrails: dict[Phase, list[BoundGuardrail]]
    """By the phase each one runs in, each phase's in list order."""
    compactor: Compactor | None
    """What trims the conversation, or `None` when nothing does."""
    memory: MemoryLike | None
    knowledge: KnowledgeLike | None
    max_turns: int
    max_tokens: int | None
    timeout: float | None

    @classmethod
    def of(cls, agent: Agent) -> AgentShape:
        """`agent`'s shape, read off its attributes; a value no run could use is a `UserError`."""
        _checked_unique(
            agent.name,
            (
                *(tool.name for tool in agent.tools),
                *(delegate.tool_name for delegate in agent.delegates),
                *(handoff.tool_name for handoff in agent.handoffs),
            ),
        )
        return cls(
            agent=agent,
            name=agent.name,
            instructions=_adapt_instructions(agent.instructions),
            model=_checked_model(agent.model),
            model_settings=agent.model_settings,
            output_type=agent.output_type,
            tools=tuple(agent.tools),
            mcp_servers=tuple(agent.mcp_servers),
            handoffs={handoff.tool_name: handoff for handoff in agent.handoffs},
            delegates={delegate.tool_name: delegate for delegate in agent.delegates},
            guardrails=flatten_guardrails(agent.guardrails),
            compactor=resolve_compactor(agent.compact),
            memory=agent.memory,
            knowledge=agent.knowledge,
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
