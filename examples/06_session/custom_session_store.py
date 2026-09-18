"""Writing your own session backend: `SessionABC`'s four abstract methods, nothing less.

See RUNA.md #6 and docs/sessions.md ("Writing Your Own Backend").

`SQLiteSession` is the only session Runa ships for a single local process; a real custom store
would back this with something durable, not a plain dict. This one stays in-process on purpose,
to keep the interface itself the whole point.

Run it:

    uv run python examples/06_session/custom_session_store.py
"""

import asyncio

from runa import Agent
from runa._types import TResponseInputItem
from runa.session import SessionABC


class DictSession(SessionABC):
    """The minimal `SessionABC` implementation: one dict, keyed by `session_id`."""

    _store: dict[str, list[TResponseInputItem]] = {}

    def __init__(self, session_id: str, *, user_id: str | None = None) -> None:
        """Store `session_id`; `user_id` scopes this session's automatic memory, if any."""
        self.session_id = session_id
        self.user_id = user_id

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        """Return this session's items, oldest first, capped at the latest `limit` if given."""
        items = self._store.get(self.session_id, [])
        return items if limit is None else items[-limit:]

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """Append `items` to this session's history."""
        self._store.setdefault(self.session_id, []).extend(items)

    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return this session's most recent item, or `None` if it has none."""
        items = self._store.get(self.session_id, [])
        return items.pop() if items else None

    async def clear_session(self) -> None:
        """Delete this session and all of its items."""
        self._store.pop(self.session_id, None)


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
