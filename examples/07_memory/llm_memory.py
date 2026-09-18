"""`memory = "llm"`: the model gets a `search_memory` tool and decides when to use it.

See RUNA.md #7 and docs/memory.md.

Nothing is written automatically in this mode -- call `memory.remember(...)` from your own code
if you want something to persist. This example writes one fact directly via `Memory`, then lets
the model decide to search for it.

Run it:

    uv run python examples/07_memory/llm_memory.py
"""

import asyncio

from runa import Agent, Memory

memory = Memory()


class SupportAgent(Agent):
    """Searches memory itself, via the `search_memory` tool the framework adds in `"llm"` mode."""

    name = "support_agent"
    instructions = "You are a helpful support assistant. Search memory if it might help."
    memory = "llm"


async def main() -> None:
    """Seed one memory directly, then let the model decide to search for it."""
    await memory.remember("The support team's SLA for refunds is 3 business days.")
    run = await SupportAgent().run("How long do refunds normally take?")
    print(run.output)


asyncio.run(main())
