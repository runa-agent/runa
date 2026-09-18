"""`@tool`: a plain function, its schema derived from the signature and docstring.

See RUNA.md #2 and docs/tools.md.

Run it:

    uv run python examples/02_tool/basic_tool.py
"""

from runa import Agent, tool

_ORDERS = {
    "A100": "shipped, arriving Thursday",
    "A101": "delayed at customs",
}


@tool
def look_up_order(order_id: str) -> str:
    """Return the current status of an order.

    order_id: the order's identifier, e.g. "A100"
    """
    return _ORDERS.get(order_id, "no order with that id")


class SupportAgent(Agent):
    """Looks up orders on request. The model decides on its own when to call the tool."""

    name = "support_agent"
    instructions = "You are a store support agent. Use look_up_order to answer order questions."
    tools = [look_up_order]


run = SupportAgent().run_sync("What's the status of order A100?")
print(run.output)
