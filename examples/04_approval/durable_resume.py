"""`RunState` surviving a process restart: serialize a pause, rebuild it, then resume.

See RUNA.md #4 and docs/approval.md ("Durability").

`state.to_json()` is a plain dict, safe to store anywhere (a database row, a queue message).
Rebuilding it needs a fresh instance of the agent the run started with, used to re-resolve the
current agent and each pending tool by name -- a live `Agent` instance can't round-trip through
JSON itself.

Run it:

    uv run python examples/04_approval/durable_resume.py
"""

import asyncio

from runa import Agent, Runner, RunState, tool


@tool(needs_approval=True)
def issue_refund(order_id: str, amount: float) -> str:
    """Refund the customer for an order.

    order_id: the order's identifier
    amount: the refund amount in dollars
    """
    return f"refunded ${amount:.2f} for order {order_id}"


class SupportAgent(Agent):
    """Issues refunds, always pausing for a human's approval first."""

    name = "support_agent"
    instructions = "You issue refunds with issue_refund when the user asks for one."
    tools = [issue_refund]


result = Runner.run_sync(SupportAgent(), "Refund $75 to order A101.")
state = result.to_state()
blob = state.to_json()  # store this anywhere; the run is now paused indefinitely
print(f"paused, serialized to {len(str(blob))} bytes of JSON-safe data")

# ... a process restart happens here; nothing but `blob` survives it ...

restored_agent = SupportAgent()  # a fresh instance of the same agent class
restored_state = asyncio.run(RunState.from_json(restored_agent, blob))
for interruption in restored_state.pending:
    restored_state.approve(interruption)

result = Runner.run_sync(restored_agent, restored_state)
print(result.final_output)
