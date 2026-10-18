"""agent_shape.py: `AgentShape`, what a run resolves about an agent once it starts running.

Its tools, including the ones its MCP servers list; its handoffs and delegates, keyed by the tool
name that triggers each; the schema of every callable the model is offered; and its model, through
the one process-wide `ModelProvider`. Each is worked out once per agent per run and shared by the
whole turn loop. The agent's declared settings themselves are read through `AgentLike`.

Only `run_internal` imports this module.
"""

from dataclasses import dataclass

from runa._models import Model, ModelProvider, ToolSchema
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
    offered: list[ToolSchema]
    """Every tool, delegate and handoff, in the order the model is offered them."""

    @classmethod
    async def of(cls, agent: AgentLike) -> AgentShape:
        """`agent`'s shape: its MCP servers' tools fetched, its subagents keyed by tool name.

        A handoff takes no structured input from the model: calling it is itself the signal to
        switch agents, so its schema's `parameters` is always an empty object.
        """
        tools = list(agent.tools)
        for server in agent.mcp_servers:
            tools.extend(await server.list_tools())
        no_input = {"type": "object", "properties": {}}
        offered = [
            *(
                ToolSchema(tool.name, tool.description or "", tool.params_json_schema or no_input)
                for tool in tools
            ),
            *(
                ToolSchema(
                    delegate.tool_name, delegate.tool_description, delegate.params_json_schema
                )
                for delegate in agent.delegates
            ),
            *(
                ToolSchema(handoff.tool_name, handoff.tool_description or "", no_input)
                for handoff in agent.handoffs
            ),
        ]
        return cls(
            agent=agent,
            tools=tools,
            handoffs={handoff.tool_name: handoff for handoff in agent.handoffs},
            delegates={delegate.tool_name: delegate for delegate in agent.delegates},
            offered=offered,
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
