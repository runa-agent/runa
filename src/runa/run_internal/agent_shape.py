"""agent_shape.py: `AgentShape`, what the turn loop reads off an Agent, resolved once per agent.

Everything here answers a question about one agent's declared shape -- its tools, including the
ones its MCP servers list; its handoffs, keyed by the tool name that triggers them; its sampling
settings; its model; its guardrails -- asked by more than one module. `run_loop`,
`tool_execution` and `run_state` all have to agree on the answers, so none of them owns these,
and each answer is worked out once when the agent starts running rather than re-derived by
whichever module needs it next.

A concrete dataclass rather than a `Protocol`, because a `Protocol` describes a surface and
builds nothing: the only implementation of the old one was a thirteen-field `SimpleNamespace`
hand-written in six test files, where a fourteenth field broke all six silently, at read time.
Every field here has a default instead, so a test names the two or three its behaviour turns on.
`agent` is the object user code is handed -- a hook's callback, a guardrail function,
`RunState.agent` -- and for a shape built directly it is the shape itself, which is what lets one
stand in for an agent the loop would otherwise need built in full.

Work that belongs to a single caller lives with that caller instead -- approval gating in
`tool_execution`, output parsing and instruction resolution in `run_loop`, agent lookup by name in
`run_state`.
"""

from dataclasses import dataclass, field
from typing import Any

from runa._models import Model, ModelProvider
from runa._types import ModelSettings
from runa.guardrail import InputGuardrail, OutputGuardrail
from runa.handoff import Handoff
from runa.lifecycle import AgentHooks
from runa.tool import FunctionTool


def _normalized_handoffs(handoffs: Any) -> dict[str, Handoff]:
    """Map each handoff's tool name to its `Handoff`, wrapping a bare `Agent` if given one.

    A `dict` is already normalized -- an `AgentShape`'s `handoffs` -- and passes straight through,
    so code walking a graph of agents (`run_state._find_agent_by_name`) doesn't have to know
    whether it is looking at an agent or at a shape standing in for one.
    """
    if isinstance(handoffs, dict):
        return handoffs
    result: dict[str, Handoff] = {}
    for entry in handoffs:
        handoff = entry if isinstance(entry, Handoff) else Handoff.from_agent(entry)
        result[handoff.tool_name] = handoff
    return result


@dataclass
class AgentShape:
    """One agent's surface as the turn loop needs it, resolved and named in one place.

    The loose annotations are the ones `Agent` itself leaves open: `model` is a string or a
    `Model`, `instructions` a string or a callable, `memory`/`knowledge` a mode string or a
    store, `compact` a bool or a `Compactor`.
    """

    name: str = "agent"
    instructions: Any = None
    model: Any = None
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    tools: list[FunctionTool] = field(default_factory=list)
    handoffs: dict[str, Handoff] = field(default_factory=dict)
    input_guardrails: list[InputGuardrail[Any]] = field(default_factory=list)
    output_guardrails: list[OutputGuardrail[Any]] = field(default_factory=list)
    output_type: type | None = None
    memory: Any = None
    knowledge: Any = None
    compact: Any = False
    hooks: AgentHooks[Any] | None = None
    agent: Any = None

    def __post_init__(self) -> None:
        """Normalize `handoffs`, and default `agent` to the shape itself if none was given."""
        self.handoffs = _normalized_handoffs(self.handoffs)
        if self.agent is None:
            self.agent = self

    @classmethod
    async def of(cls, agent: Any) -> AgentShape:
        """`agent`'s shape: its MCP servers' tools fetched, its handoffs normalized.

        Hand it one it already made and it comes straight back. `Agent` satisfies this
        structurally and nothing here imports it, which is what keeps `run_internal` from
        depending on the class it implements.
        """
        if isinstance(agent, cls):
            return agent
        tools = list(agent.tools)
        for server in agent.mcp_servers:
            tools.extend(await server.list_tools())
        return cls(
            name=agent.name,
            instructions=agent.instructions,
            model=agent.model,
            model_settings=agent.model_settings,
            tools=tools,
            handoffs=agent.handoffs,
            input_guardrails=agent.input_guardrails,
            output_guardrails=agent.output_guardrails,
            output_type=agent.output_type,
            memory=agent.memory,
            knowledge=agent.knowledge,
            compact=agent.compact,
            hooks=agent.hooks,
            agent=agent,
        )

    def find_tool(self, name: str) -> FunctionTool | None:
        """The `FunctionTool` this agent calls `name`, or `None` if it declares no such tool."""
        for candidate in self.tools:
            if isinstance(candidate, FunctionTool) and candidate.name == name:
                return candidate
        return None

    def resolve_model(self, provider: ModelProvider) -> Model:
        """This agent's `model`: a string names one of `provider`'s, anything else is one."""
        return provider.get_model(self.model) if isinstance(self.model, str) else self.model


__all__ = ["AgentShape", "_normalized_handoffs"]
