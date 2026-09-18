"""`Agent`: a plain subclass, its `Run` result, and history across calls.

See RUNA.md #1 and docs/agents.md.

Run it:

    uv run python examples/01_agent/basic_agent.py
"""

from runa import Agent


class SupportAgent(Agent):
    """A minimal support agent -- `name` is the only required attribute."""

    name = "support_agent"
    instructions = "You are a helpful, concise customer support assistant."


agent = SupportAgent()

run = agent.run_sync("My order hasn't arrived.")
print(f"output={run.output!r}")
print(f"status={run.status!r}")
print(f"usage: {run.usage.total_tokens} tokens")
print(run.trace)

# `run_sync`/`run`/`run_streamed` all append to `agent.history`, so the next call on the same
# instance continues the conversation without passing anything back in by hand.
run2 = agent.run_sync("It's order #4821, placed a week ago.")
print(f"output={run2.output!r}")

# Accumulated across every call on this instance, not just the last one.
print(f"total usage so far: {agent.usage.total_tokens} tokens")
