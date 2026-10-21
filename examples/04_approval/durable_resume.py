"""A paused run surviving a process restart: load it by its `run_id`, decide it, resume it.

See RUNA.md #4 and docs/approval.md ("Durability").

A pause is kept in `runa.db` the moment it happens, so `run.run_id` is all another process needs.
Loading it takes a fresh instance of the agent the run started with, used to find the current
agent again by name. A pause resumes once, whichever process gets to it.

Run it:

    uv run python examples/04_approval/durable_resume.py
"""

from runa import Agent, RunState, tool


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


run = SupportAgent().run_sync("Refund $75 to order A101.")
run_id = run.run_id  # hand this to whoever approves; the pause itself stays in runa.db
print(f"paused as {run_id}")

# ... a process restart happens here; nothing but `run_id` survives it ...

restored_agent = SupportAgent()  # a fresh instance of the same agent class
restored_state = RunState.load_sync(restored_agent, run_id or "")
if restored_state is None:
    raise SystemExit(f"{run_id} was already resumed")
for interruption in restored_state.pending:
    restored_state.approve(interruption)

run = restored_agent.run_sync(restored_state)
print(run.output)
