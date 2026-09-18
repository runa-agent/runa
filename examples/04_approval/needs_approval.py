"""`needs_approval`: pausing a sensitive tool call until a human signs off.

See RUNA.md #4 and docs/approval.md.

Run it:

    uv run python examples/04_approval/needs_approval.py
"""

import asyncio

from runa import Agent, Runner, approval, tool


@approval
def large_refund(amount: float) -> bool:
    """Refunds of $50 or more need a human to sign off."""
    return amount >= 50


@tool(needs_approval=large_refund)
def issue_refund(order_id: str, amount: float) -> str:
    """Refund the customer for an order.

    order_id: the order's identifier
    amount: the refund amount in dollars
    """
    return f"refunded ${amount:.2f} for order {order_id}"


class SupportAgent(Agent):
    """Issues refunds, pausing for approval on anything $50 or over."""

    name = "support_agent"
    instructions = "You issue refunds with issue_refund when the user asks for one."
    tools = [issue_refund]


async def main() -> None:
    """Run one turn, approving every interruption Runner surfaces along the way."""
    agent = SupportAgent()
    result = await Runner.run(agent, "Refund $75 to order A101, they've waited long enough.")
    while result.interruptions:
        state = result.to_state()
        for interruption in result.interruptions:
            print(f"  needs approval: {interruption.name}({interruption.arguments})")
            state.approve(interruption)  # state.reject(interruption) would skip the tool instead
        result = await Runner.run(agent, state)
    print(result.final_output)


asyncio.run(main())
