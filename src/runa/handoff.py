"""handoff.py: `Handoff` (agent-as-a-switch) and `agent_as_tool` (agent-as-a-tool).

Both wrap a sub-`Agent` as something the model can call, differing in what happens next: calling a
`Handoff`'s tool switches `run_internal`'s current agent for the rest of the run (`Agent.subagents`'
`.handoff` mode); calling an `agent_as_tool()` tool runs the sub-agent to completion, as a nested
run of the caller's, and hands its output back to the *calling* agent, which keeps going
(`.delegate` mode). See `runa.agent.Subagent` for how a class wires either mode up from its
`subagents` list.

The two modes are the only edges between agents, so this module also names them (`Mode`) and walks
them: `subagent_edges` is one agent's outgoing edges and `reachable` the whole graph below it. A
module that needs "everything reachable from this agent" asks here rather than re-reading
`handoffs` and `tools` itself, which is what keeps a third mode from being a change in five places.

An agent is read here, never built, so the attributes that walk is made of are `runa.agent_like`'s
`AgentLike` (which is also why no read below needs a `getattr` default: an agent that has no
`handoffs` is a type error at `Agent.run`, not an agent with no subagents). The one exception is
`agent_as_tool`, whose delegate the turn loop *runs* rather than reads, and so is annotated as the
`Agent` it is only ever called with -- a type-only import, so the runtime dependency still points
one way.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from runa._types import RunContext
from runa.agent_like import AgentLike
from runa.exceptions import UserError
from runa.tool import FunctionTool

if TYPE_CHECKING:
    from runa.agent import Agent

Mode = Literal["handoff", "delegate"]
"""Which of the two mechanisms above wires a subagent up: `Agent.subagents`' `.handoff`/`.delegate`.

Named here, beside both of them, rather than in `runa.agent`: a declaration spells the mode and
`Subagent` carries it, but what each one *means* is this module's.
"""

_DELEGATE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {"input": {"type": "string"}},
    "required": ["input"],
}


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
    def from_agent(cls, agent: AgentLike) -> Handoff:
        """Build a `Handoff` for `agent` with a name/description derived from it."""
        return cls(
            agent=agent,
            tool_name=f"transfer_to_{_slugify(agent.name)}",
            tool_description=f"Transfer the conversation to {agent.name}.",
        )

    @classmethod
    def by_tool_name(cls, handoffs: Any) -> dict[str, Handoff]:
        """`handoffs` keyed by the tool name that triggers each, wrapping any bare `Agent` given.

        What a `handoffs` list may hold is this class's own knowledge, so turning one into the
        map the run loop reads lives here rather than in whichever caller needs it next -- both
        `run_internal.agent_shape` (once per agent, when it starts running) and `subagent_edges`
        below (walking a graph of agents, where a node may be an `Agent` or an `AgentShape`
        standing in for one) ask for it. An already-keyed `dict` --
        an `AgentShape`'s `handoffs` -- is what that map is, so it passes straight through and
        neither caller has to know which of the two it is looking at.
        """
        if isinstance(handoffs, dict):
            return handoffs
        keyed: dict[str, Handoff] = {}
        for entry in handoffs:
            handoff = entry if isinstance(entry, cls) else cls.from_agent(entry)
            keyed[handoff.tool_name] = handoff
        return keyed


def agent_as_tool(
    agent: Agent, tool_name: str | None, tool_description: str | None
) -> FunctionTool:
    """Wrap `agent` as a `FunctionTool` the model calls with a generated `input` string.

    The tool is a schema and the agent behind it (`FunctionTool.delegate`), nothing more: what a
    delegate's run shares with its caller's is the turn loop's to decide, because it is the loop
    that runs it, as a nested run of the caller's. Its `on_invoke_tool` is never reached from a
    run, and raises `UserError` if called directly.
    """
    resolved_name = tool_name or _slugify(agent.name)

    async def on_invoke_tool(ctx: RunContext, arguments_json: str, call_id: str) -> Any:
        raise UserError(f"{resolved_name} is a delegate: it runs only as part of its caller's run")

    return FunctionTool(
        name=resolved_name,
        description=tool_description or f"Delegate a task to {agent.name}.",
        params_json_schema=_DELEGATE_INPUT_SCHEMA,
        on_invoke_tool=on_invoke_tool,
        delegate=agent,
    )


def subagent_edges(agent: AgentLike) -> Iterator[tuple[Mode, AgentLike]]:
    """`agent`'s outgoing edges in the agent graph: one `(mode, target)` per subagent of it.

    Handoffs first, then delegates, each in declaration order. An `auto`-mode subagent is wired
    up as both, so it is yielded twice, once per mode -- which is what it is: two tools the model
    picks between.

    What a `handoffs` list may hold is this module's knowledge (see `Handoff.by_tool_name`), and so
    is a delegate being a tool carrying the agent behind it (`agent_as_tool` sets it), so reading
    both lives here and not in each module that needs the graph: `run_state` resolves a paused run's
    agent by name, `Agent.graph` draws the edges. An `AgentShape` standing in for an agent walks the
    same way, the two being one shape from here (`Handoff.by_tool_name` absorbs the one field where
    they differ).
    """
    for handoff in Handoff.by_tool_name(agent.handoffs).values():
        yield "handoff", handoff.agent
    for tool in agent.tools:
        if tool.delegate is not None:
            yield "delegate", tool.delegate


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


__all__ = ["Handoff", "Mode", "agent_as_tool", "reachable", "subagent_edges"]
