"""_graph.py: `draw_graph`, a small DOT-graph generator behind `Agent.graph`.

Not a port of `agents.extensions.visualization` -- just enough to show an agent's own tools,
delegates and handoffs (recursively) as a Graphviz digraph, since that's all `Agent.graph` ever
promised. A delegate is drawn as the agent it wraps (dotted edge), not as a plain tool box; a
handoff gets a dashed edge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graphviz import Digraph, Source


def _agent_node_id(agent: Any) -> str:
    return f"agent_{id(agent)}"


def _add_agent(dot: Digraph, agent: Any, seen: set[int]) -> None:
    if id(agent) in seen:
        return
    seen.add(id(agent))
    node_id = _agent_node_id(agent)
    dot.node(node_id, agent.name, shape="ellipse", style="filled", fillcolor="lightblue")

    for tool in getattr(agent, "tools", []) or []:
        if getattr(tool, "delegate", None) is not None:
            dot.edge(node_id, _agent_node_id(tool.delegate), style="dotted")
            _add_agent(dot, tool.delegate, seen)
            continue
        tool_id = f"tool_{id(tool)}"
        dot.node(
            tool_id,
            getattr(tool, "name", str(tool)),
            shape="box",
            style="filled",
            fillcolor="lightyellow",
        )
        dot.edge(node_id, tool_id)

    for server in getattr(agent, "mcp_servers", []) or []:
        server_id = f"mcp_{id(server)}"
        dot.node(
            server_id,
            getattr(server, "name", "mcp"),
            shape="box",
            style="filled,dashed",
            fillcolor="lightgrey",
        )
        dot.edge(node_id, server_id)

    for handoff in getattr(agent, "handoffs", []) or []:
        target = getattr(handoff, "agent", handoff)
        dot.edge(node_id, _agent_node_id(target), style="dashed")
        _add_agent(dot, target, seen)


def draw_graph(agent: Any) -> Source:
    """Render `agent`, and its tools/subagents/MCP servers (recursively), as a Graphviz `Source`."""
    from graphviz import Digraph

    dot = Digraph()
    _add_agent(dot, agent, set())
    return dot.unflatten(stagger=3)


__all__ = ["draw_graph"]
