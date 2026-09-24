"""A guided tour: tools, guardrails, approval, subagents, tracing, and eval, in one run.

One agent, wired the way a real app would, so you can see the primitives interact: a guardrail
blocking a bad input, a tool call showing up in the trace, a delegate answering a subquestion, a
handoff transferring the conversation, a sensitive tool pausing for approval, and the whole agent
graded against an eval dataset. Each section prints its own trace, so `run.trace`'s span tree is
visible right next to the primitive that produced it.

Run it:

    uv run python examples/applications/tour.py

Uses Runa's default model (`gpt-5.4-nano`), so it runs with just `OPENAI_API_KEY` set -- change
`model` on `SupportAgent` below to `"claude-sonnet-5"` (and set `ANTHROPIC_API_KEY`) to run it on
Claude instead.
"""

import asyncio

from runa import (
    Agent,
    Case,
    ConsoleExporter,
    RunHooks,
    SQLiteExporter,
    approval,
    guardrail,
    observe,
    tool,
)

# --------------------------------------------------------------------------------------------
# 1. Tools, with guardrails on both the agent and one tool's arguments.
# --------------------------------------------------------------------------------------------

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


@guardrail
def no_negative_amount(args: dict) -> bool:
    """Trip if a refund's amount is zero or negative."""
    return args.get("amount", 0) <= 0


@approval
def large_refund(amount: float) -> bool:
    """Refunds of $50 or more need a human to sign off."""
    return amount >= 50


@tool(guardrails=[no_negative_amount.input], needs_approval=large_refund)
def issue_refund(order_id: str, amount: float) -> str:
    """Refund the customer for an order.

    order_id: the order's identifier
    amount: the refund amount in dollars
    """
    return f"refunded ${amount:.2f} for order {order_id}"


@guardrail
def block_empty(input: str) -> bool:
    """Trip when the user sends an empty message."""
    return not input.strip()


# --------------------------------------------------------------------------------------------
# 2. Subagents: a delegate the main agent consults, and a handoff it can transfer control to.
# --------------------------------------------------------------------------------------------


class PolicyResearcher(Agent):
    """Answers questions about store policy; called as a subroutine, never talks to the user."""

    name = "policy_researcher"
    instructions = "Answer questions about refund and shipping policy in one or two sentences."


class BillingAgent(Agent):
    """Takes over billing disputes once handed the conversation."""

    name = "billing_agent"
    instructions = "You resolve billing disputes directly. Be firm but fair."


# --------------------------------------------------------------------------------------------
# 3. Hooks: an app-level observer, independent of the trace every run already records.
# --------------------------------------------------------------------------------------------


class ToolLogger(RunHooks):
    """Prints every tool call made during a run, as it happens."""

    async def on_tool_end(
        self, context: object, agent: object, tool: object, result: object
    ) -> None:
        """Print the tool's name and result."""
        print(f"  [hook] {tool.name} -> {result!r}")  # type: ignore[attr-defined]


class SupportAgent(Agent):
    """Front-line support: looks up orders, issues refunds, and escalates as needed."""

    name = "support_agent"
    instructions = (
        "You are a store support agent. Always call look_up_order before discussing an order. "
        "Issue refunds with issue_refund when asked. Consult the policy researcher for policy "
        "questions. Hand off billing disputes to billing."
    )
    tools = [look_up_order, issue_refund]
    guardrails = [block_empty.input]
    subagents = [PolicyResearcher.delegate, BillingAgent.handoff]


async def demo_guardrail() -> None:
    """Show a tripped input guardrail: the run stops before the model is ever called."""
    print("\n=== 1. Guardrail: an empty message never reaches the model ===")
    run = await SupportAgent().run("   ")
    print(f"status={run.status!r} error={run.error!r}")


async def demo_tool_and_trace() -> None:
    """Show a normal tool call, the hook that observed it, and the trace it produced."""
    print("\n=== 2. Tool call, observed by a hook and recorded on the trace ===")
    run = await SupportAgent().run("What's the status of order A100?", hooks=ToolLogger())
    print(run.output)
    print(run.trace)
    print(f"usage: {run.usage.total_tokens} tokens")


async def demo_delegate() -> None:
    """Show a delegate: the main agent consults the researcher, then answers itself."""
    print("\n=== 3. Delegate: the researcher answers a subquestion, support replies ===")
    run = await SupportAgent().run("What's our refund policy for delayed orders?")
    print(run.output)


async def demo_handoff() -> None:
    """Show a handoff: billing takes over the conversation entirely."""
    print("\n=== 4. Handoff: billing takes over the conversation ===")
    run = await SupportAgent().run("I want to dispute a charge on my card.")
    print(run.output)


async def demo_approval() -> None:
    """Show a tool pausing for human approval, then resuming once it's granted."""
    print("\n=== 5. Approval: a $75 refund pauses for a human, then resumes ===")
    agent = SupportAgent()
    run = await agent.run("Refund $75 to order A101, they've waited long enough.")
    while run.status == "paused":
        state = run.to_state()
        for interruption in run.interruptions:
            print(f"  needs approval: {interruption.name}({interruption.arguments})")
            state.approve(interruption)  # state.reject(interruption) would skip the tool instead
        run = await agent.run(state)
    print(run.output)


async def demo_eval() -> None:
    """Show `agent.evaluate()`: grade a fresh agent instance against a small dataset."""
    print("\n=== 6. Eval: grade the agent against a dataset ===")
    dataset = [
        Case(
            input="What's the status of order A100?",
            expected="Tells the user the order shipped and is arriving Thursday",
            expected_tool="look_up_order",
        ),
        Case(
            input="What's your refund policy for delayed orders?",
            expected="Gives a policy answer about refunds for delayed orders",
        ),
    ]
    report = await SupportAgent().evaluate(dataset)
    print(report)  # aggregate: mean score per metric (report.metrics), pass/fail counts, failures
    print()
    for case in report.cases:
        print(f"{case.id}: {case.case.input!r}")
        for result in case.results:
            score = "n/a" if result.score is None else f"{result.score:.2f}"
            print(f"  {result.metric:<20} {result.status:<7} score={score}")


async def main() -> None:
    """Run every demo in turn."""
    observe(exporter=[SQLiteExporter(), ConsoleExporter()])
    await demo_guardrail()
    await demo_tool_and_trace()
    await demo_delegate()
    await demo_handoff()
    await demo_approval()
    await demo_eval()


if __name__ == "__main__":
    asyncio.run(main())
