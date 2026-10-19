"""Writing your own session backend: `Session`'s three abstract methods.

See RUNA.md #6 and docs/sessions.md ("Writing Your Own Backend").

`SQLiteSession` is the only session Runa ships for a single local process; a real custom store
would back this with something durable, not a plain dict. This one stays in-process on purpose,
to keep the interface itself the whole point.

Run it:

    uv run python examples/06_session/custom_session_store.py
"""

import asyncio

from runa import Agent, ConversationItem
from runa.session import Session


class DictSession(Session):
    """The minimal `Session` implementation: one dict, keyed by `session_id`."""

    _store: dict[str, list[ConversationItem]] = {}

    def __init__(self, session_id: str, *, user_id: str | None = None) -> None:
        """Store `session_id`; `user_id` scopes this session's automatic memory, if any."""
        self.session_id = session_id
        self.user_id = user_id

    async def get_items(self) -> list[ConversationItem]:
        """Return this session's items, oldest first."""
        return list(self._store.get(self.session_id, []))

    async def add_items(self, items: list[ConversationItem]) -> None:
        """Append `items` to this session's history."""
        self._store.setdefault(self.session_id, []).extend(items)

    async def set_items(self, items: list[ConversationItem]) -> None:
        """Replace this session's entire history with `items`."""
        self._store[self.session_id] = list(items)


class SupportAgent(Agent):
    """A support agent whose history lives in the in-process `DictSession` above."""

    name = "support_agent"
    instructions = "You are a helpful support assistant."


async def main() -> None:
    """Run two turns against the same `DictSession`, then confirm history round-tripped."""
    session = DictSession("user-42")
    agent = SupportAgent()
    await agent.run("My order hasn't arrived.", session=session)
    run = await agent.run("It's order #4821.", session=session)
    print(run.output)
    print(f"{len(await session.get_items())} items stored")


asyncio.run(main())
