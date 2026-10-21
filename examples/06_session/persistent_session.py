"""`db.session`: conversation history that survives past one `Agent` instance.

See RUNA.md #6 and docs/sessions.md.

With a `session`, only the new message is ever passed to `run`/`run_sync` -- prior turns come
back from `runa.db` automatically, and `agent.history` is left untouched.

Run it:

    uv run python examples/06_session/persistent_session.py
"""

import asyncio

from runa import Agent, db


class SupportAgent(Agent):
    """A support agent with no memory of its own -- history lives in the session instead."""

    name = "support_agent"
    instructions = "You are a helpful support assistant."


agent = SupportAgent()
session = db.session("user-42")  # db/runa.db, or wherever RUNA_DATABASE_URL points

agent.run_sync("My order hasn't arrived.", session=session)
run = agent.run_sync("It's order #4821.", session=session)  # remembers the first message too
print(run.output)


async def inspect() -> None:
    """The session's own async methods, beyond what `run`/`run_sync` call for you."""
    items = await session.get_items()
    print(f"{len(items)} items stored for session {session.session_id!r}")
    await session.clear_session()


asyncio.run(inspect())
