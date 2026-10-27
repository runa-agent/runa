"""_graph.py: `draw_graph`, a small DOT-graph generator behind `Agent.graph`.

Just enough to show an agent's own tools, delegates and handoffs (recursively) as a Graphviz
digraph, since that's all `Agent.graph` promises. A delegate is drawn as the agent it wraps
(dotted edge), not as a plain tool box; a handoff gets a dashed edge.

Which agents those are comes from `reachable`, the same walk a resumed run finds its agents again
through, each agent class drawn once with the settings a run gives it.
"""

from typing import TYPE_CHECKING

from runa.handoff import Mode
from runa.run_internal.agent_shape import reachable, subagent_edges

if TYPE_CHECKING:
    from graphviz import Digraph

    from runa.agent import Agent

_EDGE_STYLE: dict[Mode, str] = {"handoff": "dashed", "delegate": "dotted"}
"""How each subagent mode is drawn. A mode `runa.handoff` grows and this doesn't is a `KeyError`
here, not a plausible-looking edge in the wrong style."""


def _agent_node_id(agent_cls: type[Agent]) -> str:
    return f"agent_{id(agent_cls)}"


def draw_graph(agent: Agent) -> Digraph:
    """Render `agent`, and its tools/subagents/MCP servers (recursively), as a Graphviz `Digraph`.

    Building it runs no Graphviz executable: only `.render()` (or Jupyter's inline image) needs
    the system `dot` binary.
    """
    from graphviz import Digraph

    dot = Digraph()
    for agent_cls, resolved in reachable(agent):
        node_id = _agent_node_id(agent_cls)
        dot.node(node_id, resolved.name, shape="ellipse", style="filled", fillcolor="lightblue")

        for tool in resolved.tools:
            tool_id = f"tool_{id(tool)}"
            dot.node(tool_id, tool.name, shape="box", style="filled", fillcolor="lightyellow")
            dot.edge(node_id, tool_id)

        for server in resolved.mcp_servers:
            server_id = f"mcp_{id(server)}"
            dot.node(
                server_id,
                server.name,
                shape="box",
                style="filled,dashed",
                fillcolor="lightgrey",
            )
            dot.edge(node_id, server_id)

        for mode, target in subagent_edges(resolved):
            dot.edge(node_id, _agent_node_id(target), style=_EDGE_STYLE[mode])
    return dot


__all__ = ["draw_graph"]
