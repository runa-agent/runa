"""handoff.py: `Handoff` (agent-as-a-switch) and `Delegate` (agent-as-a-tool).

Both wrap a sub-`Agent` as something the model can call, differing in what happens next: calling a
`Handoff`'s tool switches `run_internal`'s current agent for the rest of the run (`Agent.subagents`'
`.handoff` mode); calling a `Delegate`'s tool runs the sub-agent to completion, as a nested run of
the caller's, and hands its output back to the *calling* agent, which keeps going (`.delegate`
mode). See `runa.agent.Subagent` for how a class wires either mode up from its `subagents` list.

The two modes are the only edges between agents, so this module also names them (`Mode`) and walks
them: `subagent_edges` is one agent's outgoing edges and `reachable` the whole graph below it. A
module that needs "everything reachable from this agent" asks here rather than re-reading
`handoffs` and `delegates` itself, which is what keeps a third mode from being a change in five
places.

An agent is read here, never built, so the attributes that walk is made of are `runa.agent_like`'s
`AgentLike` (which is also why no read below needs a `getattr` default: an agent that has no
`handoffs` is a type error at `Agent.run`, not an agent with no subagents). The one exception is
`Delegate.agent`, which the turn loop *runs* rather than reads, with its own ceilings, and so is
annotated as the `Agent` it always is -- a type-only import, so the runtime dependency still
points one way.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from runa.agent_like import AgentLike

if TYPE_CHECKING:
    from runa.agent import Agent

Mode = Literal["handoff", "delegate"]
"""Which of the two mechanisms above wires a subagent up: `Agent.subagents`' `.handoff`/`.delegate`.

Named here, beside both of them, rather than in `runa.agent`: a declaration spells the mode and
`Subagent` carries it, but what each one *means* is this module's.
"""


def _slugify(name: str) -> str:
    """Turn an agent's `name` into a valid, lowercase, underscore-separated tool name."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


@dataclass
class Handoff:
    """A sub-agent registered as a switch: calling its tool hands the run over to it.

    `run_internal` sees a `Handoff` in an agent's `handoffs` list and recognizes a call to
    `tool_name` as a request to switch `current_agent` to `agent`, rather than a normal tool call.
    """

    agent: AgentLike
    tool_name: str
    tool_description: str

    @classmethod
    def from_agent(
        cls, agent: AgentLike, tool_name: str | None = None, tool_description: str | None = None
    ) -> Handoff:
        """Build a `Handoff` for `agent`, its name/description derived from it unless given."""
        return cls(
            agent=agent,
            tool_name=tool_name or f"transfer_to_{_slugify(agent.name)}",
            tool_description=tool_description or f"Transfer the conversation to {agent.name}.",
        )


@dataclass(frozen=True)
class Delegate:
    """A sub-agent registered as a tool: calling it runs `agent` on the `input` the model wrote.

    `run_internal` sees a `Delegate` in an agent's `delegates` list and runs a call to `tool_name`
    as a nested run of the caller's, whose output goes back to the caller as the call's result.
    """

    agent: Agent
    tool_name: str
    tool_description: str
    params_json_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        }
    )
    """What the model calls it with: the one `input` string the delegate is handed as its turn."""

    @classmethod
    def from_agent(
        cls, agent: Agent, tool_name: str | None = None, tool_description: str | None = None
    ) -> Delegate:
        """Build a `Delegate` for `agent`, its name/description derived from it unless given."""
        return cls(
            agent=agent,
            tool_name=tool_name or _slugify(agent.name),
            tool_description=tool_description or f"Delegate a task to {agent.name}.",
        )


def subagent_edges(agent: AgentLike) -> Iterator[tuple[Mode, AgentLike]]:
    """`agent`'s outgoing edges in the agent graph: one `(mode, target)` per subagent of it.

    Handoffs first, then delegates, each in declaration order. An `auto`-mode subagent is wired
    up as both, so it is yielded twice, once per mode -- which is what it is: two tools the model
    picks between.

    Reading both lists lives here and not in each module that needs the graph: `run_state`
    resolves a paused run's agent by name, `Agent.graph` draws the edges.
    """
    for handoff in agent.handoffs:
        yield "handoff", handoff.agent
    for delegate in agent.delegates:
        yield "delegate", delegate.agent


def reachable(agent: AgentLike) -> Iterator[AgentLike]:
    """`agent`, then every agent reachable from it through handoffs and delegates, breadth-first.

    Each agent once, by identity, so a cycle (two agents listing each other, an agent listing
    itself) terminates rather than recursing forever.
    """
    seen: set[int] = set()
    queue: list[AgentLike] = [agent]
    while queue:
        candidate = queue.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate
        queue.extend(target for _, target in subagent_edges(candidate))


__all__ = ["Delegate", "Handoff", "Mode", "reachable", "subagent_edges"]
