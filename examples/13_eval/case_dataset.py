"""An eval module: module-level `agent` and `dataset`, graded with `agent.evaluate(...)`.

See RUNA.md #13 and docs/evaluation.md.

A real app puts this under `evals/`, where `runa eval` discovers every such module
automatically; this file lives in `examples/` to stay runnable standalone.

Run it:

    uv run python examples/13_eval/case_dataset.py
"""

import asyncio

from runa import Agent, Case, tool

_ORDERS = {"A100": "shipped, arriving Thursday"}


@tool
def look_up_order(order_id: str) -> str:
    """Return the current status of an order.

    order_id: the order's identifier
    """
    return _ORDERS.get(order_id, "no order with that id")


class SupportAgent(Agent):
    """A support agent, graded against the dataset below."""

    name = "support_agent"
    instructions = "You are a helpful support assistant. Use look_up_order for order questions."
    tools = [look_up_order]


agent = SupportAgent()

dataset = [
    Case(
        input="What's the status of order A100?",
        expected="Tells the user the order shipped and is arriving Thursday",
        expected_tool="look_up_order",
    ),
]


async def main() -> None:
    """Grade `agent` against `dataset`, and print the aggregate report."""
    report = await agent.evaluate(dataset)
    print(report)


asyncio.run(main())
