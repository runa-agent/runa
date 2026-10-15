"""_graph.py: `draw_graph`, a small DOT-graph generator behind `Agent.graph`.

Not a port of `agents.extensions.visualization` -- just enough to show an agent's own tools,
delegates and handoffs (recursively) as a Graphviz digraph, since that's all `Agent.graph` ever
promised. A delegate is drawn as the agent it wraps (dotted edge), not as a plain tool box; a
handoff gets a dashed edge.

Which agents those are comes from `runa.handoff.subagent_edges`, the same walk `run_state` resumes
a paused run through; only the recursion stays here, because drawing needs each agent's tools and
MCP servers too, which a walk over agents alone doesn't carry.
"""

from typing import TYPE_CHECKING

from runa.agent_like import AgentLike
from runa.handoff import Mode, subagent_edges

if TYPE_CHECKING:
    from graphviz import Digraph

_EDGE_STYLE: dict[Mode, str] = {"handoff": "dashed", "delegate": "dotted"}
"""How each subagent mode is drawn. A mode `runa.handoff` grows and this doesn't is a `KeyError`
here, not a plausible-looking edge in the wrong style."""


def _agent_node_id(agent: AgentLike) -> str:
    return f"agent_{id(agent)}"


def _add_agent(dot: Digraph, agent: AgentLike, seen: set[int]) -> None:
    if id(agent) in seen:
        return
    seen.add(id(agent))
    node_id = _agent_node_id(agent)
    dot.node(node_id, agent.name, shape="ellipse", style="filled", fillcolor="lightblue")

    for tool in agent.tools:
        if tool.delegate is not None:
            continue  # an agent, drawn as one by the subagent loop below, not a tool box
        tool_id = f"tool_{id(tool)}"
        dot.node(tool_id, tool.name, shape="box", style="filled", fillcolor="lightyellow")
        dot.edge(node_id, tool_id)

    for server in agent.mcp_servers:
        server_id = f"mcp_{id(server)}"
        dot.node(
            server_id,
            server.name,
            shape="box",
            style="filled,dashed",
            fillcolor="lightgrey",
        )
        dot.edge(node_id, server_id)

    for mode, target in subagent_edges(agent):
        dot.edge(node_id, _agent_node_id(target), style=_EDGE_STYLE[mode])
        _add_agent(dot, target, seen)


def draw_graph(agent: AgentLike) -> Digraph:
    """Render `agent`, and its tools/subagents/MCP servers (recursively), as a Graphviz `Digraph`.

    Building it runs no Graphviz executable: only `.render()` (or Jupyter's inline image) needs
    the system `dot` binary.
    """
    from graphviz import Digraph

    dot = Digraph()
    _add_agent(dot, agent, set())
    return dot


__all__ = ["draw_graph"]
