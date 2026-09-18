"""`RunHooks`: an observer passed per-call, firing for every agent in the run, subagents included.

See RUNA.md #11 and docs/tracing.md ("Hooks").

Tracing is unconditional and separate from this -- hooks are for your own logic: metrics, a
single audit log for the whole run. Every method is a no-op unless overridden.

Run it:

    uv run python examples/11_hooks/run_hooks.py
"""

from runa import Agent, RunHooks, tool


@tool
def look_up_order(order_id: str) -> str:
    """Return the current status of an order.

    order_id: the order's identifier
    """
    return "shipped, arriving Thursday"


class ToolLogger(RunHooks):
    """Prints every tool call made during the run, as it happens."""

    async def on_tool_end(
        self, context: object, agent: object, tool: object, result: object
    ) -> None:
        """Print the tool's name and result."""
        print(f"  [hook] {tool.name} -> {result!r}")  # type: ignore[attr-defined]


class SupportAgent(Agent):
    """Looks up orders on request."""

    name = "support_agent"
    instructions = "Use look_up_order to answer order questions."
    tools = [look_up_order]


run = SupportAgent().run_sync("What's the status of order A100?", hooks=ToolLogger())
print(run.output)
