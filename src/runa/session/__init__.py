"""`runa.session`: `Session`, the conversation history a run reads and appends to.

Which database a session lives in is `runa.db`'s decision, not the call site's. There is one
implementation, `session/database.py`'s, over whichever database `RUNA_DATABASE_URL` names.
`runa serve`, `runa chat` and `Agent.run` all take whatever `db.session(...)` hands them, so
moving a deployment to Postgres is one environment variable rather than an edit, and
`db.at(url).session(...)` is a session in some other database.

`db.sessions()` is the read side of the same tables, for `runa sessions` and `runa ui`.
"""

from abc import ABC, abstractmethod

from runa._items import ConversationItem


class Session(ABC):
    """The conversation history a run replays before its turn and saves after it.

    A custom store implements the three abstract methods, the whole of what a run calls.
    `clear_session` has a default built on `set_items`; override it when deleting a session means
    more than emptying its history.
    """

    session_id: str

    @abstractmethod
    async def get_items(self) -> list[ConversationItem]:
        """Return this session's items, oldest first."""

    @abstractmethod
    async def add_items(self, items: list[ConversationItem]) -> None:
        """Append `items` to this session's history."""

    @abstractmethod
    async def set_items(self, items: list[ConversationItem]) -> None:
        """Replace this session's entire history with `items`."""

    async def clear_session(self) -> None:
        """Delete this session's history."""
        await self.set_items([])


__all__ = ["Session"]
