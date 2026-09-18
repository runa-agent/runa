"""`.delegate`: calling another agent as a subroutine, like a tool call.

See RUNA.md #5 and docs/subagents.md.

The calling agent stays in control: it gets the delegate's output back and decides what to say,
unlike a handoff which transfers the conversation entirely.

Run it:

    uv run python examples/05_subagent/delegate.py
"""

from runa import Agent


class PolicyResearcher(Agent):
    """Answers questions about store policy; called as a subroutine, never talks to the user."""

    name = "policy_researcher"
    instructions = "Answer questions about refund and shipping policy in one or two sentences."


class SupportAgent(Agent):
    """Consults the policy researcher, then answers the user itself."""

    name = "support_agent"
    instructions = "Consult the policy researcher for policy questions, then answer the user."
    subagents = [PolicyResearcher.delegate]


run = SupportAgent().run_sync("What's our refund policy for delayed orders?")
print(run.output)
