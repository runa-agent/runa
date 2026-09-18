"""An `async def` tool, and the reserved `ctx` parameter for reading run context.

See RUNA.md #2 and docs/tools.md ("Async Tools", "Reserved Parameters").

A parameter named `ctx` or `call_id` is filled in by the framework instead of the model, and
never appears in the tool's schema.

Run it:

    uv run python examples/02_tool/async_tool.py
"""

import asyncio
from dataclasses import dataclass

from runa import Agent, tool


@dataclass
class Context:
    """Per-run data made available to tools via the reserved `ctx` parameter."""

    user_id: str


@tool
async def fetch_price(symbol: str) -> float:
    """Look up a stock's current price.

    symbol: the ticker symbol, e.g. "MSFT"
    """
    await asyncio.sleep(0)  # a real lookup would await an HTTP call here
    return 417.32


@tool
def whoami(ctx) -> str:
    """Return the current user's id from run context."""
    return ctx.context.user_id


class InvestingAgent(Agent):
    """Answers questions about stock prices and who is asking."""

    name = "investing_agent"
    instructions = "You answer questions about stock prices, and who the current user is."
    tools = [fetch_price, whoami]


async def main() -> None:
    """Run the agent once, with a context carrying the current user's id."""
    question = "What's MSFT trading at, and who am I?"
    run = await InvestingAgent().run(question, context=Context(user_id="user-42"))
    print(run.output)


asyncio.run(main())
