"""`.handoff`: transferring the whole conversation to another agent.

See RUNA.md #5 and docs/subagents.md.

From the point of handoff, the subagent is the one talking to the user -- the main agent doesn't
see the reply or get a turn back.

Run it:

    uv run python examples/05_subagent/handoff.py
"""

from runa import Agent


class BillingAgent(Agent):
    """Takes over billing disputes once handed the conversation."""

    name = "billing_agent"
    instructions = "You resolve billing disputes directly. Be firm but fair."


class SupportAgent(Agent):
    """Front-line support, handing off billing disputes entirely."""

    name = "support_agent"
    instructions = "You help with general support. Hand off billing disputes to billing."
    subagents = [BillingAgent.handoff]


run = SupportAgent().run_sync("I want to dispute a charge on my card.")
print(run.output)
