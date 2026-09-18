"""`AgentHooks`: scoped to one `Agent` subclass, via its `hooks` class attribute.

See RUNA.md #11 and docs/tracing.md ("Hooks").

Unlike `RunHooks`, this fires only for the agent it's attached to, not for the whole run
(subagents included).

Run it:

    uv run python examples/11_hooks/agent_hooks.py
"""

from runa import Agent, AgentHooks


class LoggingHooks(AgentHooks):
    """Prints when this agent starts and finishes a run."""

    async def on_start(self, context: object, agent: object) -> None:
        """Print that the agent started."""
        print(f"  [hook] {agent.name} started")  # type: ignore[attr-defined]

    async def on_end(self, context: object, agent: object, output: object) -> None:
        """Print the agent's final output."""
        print(f"  [hook] {agent.name} finished -> {output!r}")  # type: ignore[attr-defined]


class SupportAgent(Agent):
    """A support agent whose own start/end are logged, independent of the run as a whole."""

    name = "support_agent"
    instructions = "You are a helpful support assistant."
    hooks = LoggingHooks()


run = SupportAgent().run_sync("My order hasn't arrived.")
print(run.output)
