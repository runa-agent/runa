"""An eval module whose `dataset` is loaded from a JSONL file next to it.

See RUNA.md #13 and docs/evaluation.md.

Run it:

    uv run python examples/13_eval/jsonl_dataset.py
"""

import asyncio
from pathlib import Path

from runa import Agent, Dataset, tool

_ORDERS = {"A100": "shipped, arriving Thursday"}


@tool
def look_up_order(order_id: str) -> str:
    """Return the current status of an order.

    order_id: the order's identifier
    """
    return _ORDERS.get(order_id, "no order with that id")


class SupportAgent(Agent):
    """A support agent, graded against `jsonl_dataset.jsonl`."""

    name = "support_agent"
    instructions = "You are a helpful support assistant. Use look_up_order for order questions."
    tools = [look_up_order]


agent = SupportAgent()

dataset = Dataset.from_jsonl(Path(__file__).with_suffix(".jsonl"))


async def main() -> None:
    """Grade `agent` against `dataset`, and print the aggregate report."""
    report = await agent.evaluate(dataset)
    print(report)


asyncio.run(main())
