"""agent_shape.py: `AgentShape`, what the turn loop works out about an agent once it starts running.

An agent's settings are read off the agent itself, through `runa.agent_like`'s `AgentLike`. This
module holds only what a run has to *resolve* from them, asked by more than one module: its tools,
including the ones its MCP servers list; its handoffs and delegates, keyed by the tool name that
triggers each; and its model. `run_loop`, `tool_execution` and `guardrails` all have to agree on
those answers, so none of them owns these, and each is worked out once when the agent starts
running rather than re-derived by whichever module needs it next. A model name resolves through
the one process-wide `ModelProvider`, which caches each provider's client.

Only `run_internal` imports this module.

Work that belongs to a single caller lives with that caller instead -- approval gating in
`tool_execution`, output parsing and instruction resolution in `run_loop`, agent lookup by name in
`run_state`.
"""

from dataclasses import dataclass

from runa._models import Model, ModelProvider
from runa.agent_like import AgentLike
from runa.handoff import Delegate, Handoff
from runa.tool import FunctionTool

_MODEL_PROVIDER = ModelProvider()


@dataclass
class AgentShape:
    """One agent as a run has resolved it: `agent` as declared, and the facts that needed a run.

    Every other setting is `agent`'s own, read as `shape.agent.<setting>`.
    """

    agent: AgentLike
    tools: list[FunctionTool]
    handoffs: dict[str, Handoff]
    delegates: dict[str, Delegate]

    @classmethod
    async def of(cls, agent: AgentLike) -> AgentShape:
        """`agent`'s shape: its MCP servers' tools fetched, its subagents keyed by tool name."""
        tools = list(agent.tools)
        for server in agent.mcp_servers:
            tools.extend(await server.list_tools())
        return cls(
            agent=agent,
            tools=tools,
            handoffs={handoff.tool_name: handoff for handoff in agent.handoffs},
            delegates={delegate.tool_name: delegate for delegate in agent.delegates},
        )

    def find_tool(self, name: str) -> FunctionTool | None:
        """The `FunctionTool` this agent calls `name`, or `None` if it declares no such tool."""
        for candidate in self.tools:
            if isinstance(candidate, FunctionTool) and candidate.name == name:
                return candidate
        return None

    def resolve_model(self) -> Model:
        """This agent's `model`: a string names a provider's model, anything else is one.

        Resolved here rather than at construction because resolving a name builds the provider's
        client, which is a `UserError` when its API key is unset, and an `Agent` must be
        constructible without one.
        """
        model = self.agent.model
        return _MODEL_PROVIDER.get_model(model) if isinstance(model, str) else model


__all__ = ["AgentShape"]
