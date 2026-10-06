"""agent_shape.py: what the turn loop reads off an Agent, and the one way to read each of it.

Everything here answers a question about one agent's declared shape (its tools, including the
ones its MCP servers currently list; its handoffs; its sampling settings; its model), asked by
more than one module: `run_loop`, `tool_execution` and `run_state` all have to agree on the
answers, so none of them owns these. Work that belongs to a single caller lives with that
caller instead -- approval gating in `tool_execution`, output parsing and instruction
resolution in `run_loop`, agent lookup by name in `run_state`.
"""

from typing import Any, Protocol

from runa._models import Model, ModelProvider
from runa._types import ModelSettings
from runa.guardrail import InputGuardrail, OutputGuardrail
from runa.handoff import Handoff
from runa.tool import FunctionTool


class AgentLike(Protocol):
    """The whole of what the turn loop reads off an agent, named in one place.

    `Agent` satisfies this structurally, and `Agent.run` passing `self` into `_run_async` is
    what proves it: nothing here imports `Agent`, which is what keeps `run_internal` from
    depending on the class it implements. Declaring the surface is also what lets the loop read
    `agent.mcp_servers` instead of `getattr(agent, "mcp_servers", [])` -- a name the agent
    doesn't carry is now a typo the typechecker catches, not a silently empty default.

    The loose annotations are the ones `Agent` itself leaves open: `model` is a string or a
    `Model`, `instructions` a string or a callable, `memory`/`knowledge` a mode string or a
    store, `compact` a bool or a `Compactor`.
    """

    name: str
    instructions: Any
    model: Any
    model_settings: ModelSettings
    tools: list[FunctionTool]
    handoffs: list[Any]
    mcp_servers: list[Any]
    input_guardrails: list[InputGuardrail[Any]]
    output_guardrails: list[OutputGuardrail[Any]]
    output_type: type | None
    memory: Any
    knowledge: Any
    compact: Any


def _normalized_handoffs(handoffs: list[Any]) -> dict[str, Handoff]:
    """Map each handoff's tool name to its `Handoff`, wrapping a bare `Agent` if given one."""
    result: dict[str, Handoff] = {}
    for entry in handoffs:
        handoff = entry if isinstance(entry, Handoff) else Handoff.from_agent(entry)
        result[handoff.tool_name] = handoff
    return result


async def _agent_tools(agent: AgentLike) -> list[FunctionTool]:
    """This agent's own `tools`, plus whatever its `mcp_servers` currently list."""
    tools = list(agent.tools)
    for server in agent.mcp_servers:
        tools.extend(await server.list_tools())
    return tools


def _find_tool(tools: list[Any], name: str) -> FunctionTool | None:
    for candidate in tools:
        if isinstance(candidate, FunctionTool) and candidate.name == name:
            return candidate
    return None


def _resolve_model(agent: AgentLike, model_provider: ModelProvider) -> Model:
    """This agent's `model`: a string names one of `model_provider`'s, anything else is one."""
    return (
        agent.model if not isinstance(agent.model, str) else model_provider.get_model(agent.model)
    )


__all__ = [
    "AgentLike",
    "_agent_tools",
    "_find_tool",
    "_normalized_handoffs",
    "_resolve_model",
]
