"""`Hooks` on an `Agent` subclass: scoped to that one agent, via its `hooks` class attribute.

See RUNA.md #11 and docs/tracing.md ("Hooks").

Same class as run_hooks.py passes to `run`; living here is what scopes it. It fires only for
the agent it's attached to, not for every agent in the run.

Run it:

    uv run python examples/11_hooks/agent_hooks.py
"""

from runa import Agent, Hooks


class StartEndLogger(Hooks):
    """Prints when this agent starts and finishes a run."""

    async def on_agent_start(self, context: object, agent: object) -> None:
        """Print that the agent started."""
        print(f"  [hook] {agent.name} started")  # type: ignore[attr-defined]

    async def on_agent_end(self, context: object, agent: object, output: object) -> None:
        """Print the agent's final output."""
        print(f"  [hook] {agent.name} finished -> {output!r}")  # type: ignore[attr-defined]


class SupportAgent(Agent):
    """A support agent whose own start/end are logged, independent of the run as a whole."""

    name = "support_agent"
    instructions = "You are a helpful support assistant."
    hooks = StartEndLogger()


run = SupportAgent().run_sync("My order hasn't arrived.")
print(run.output)
