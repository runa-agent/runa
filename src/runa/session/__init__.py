"""`runa.session`: `Session`, the conversation history a run reads and appends to.

Which implementation a deployment gets is `runa.db`'s decision, not the call site's:
`session/sqlite.py` locally, `session/postgres.py` when `RUNA_DATABASE_URL` points at a shared
database. `runa serve`, `runa chat` and `Agent.run` all take whatever `db.session(...)` hands
them, so moving a deployment to Postgres is one environment variable rather than an edit.

`session/store.py` is the read side of the same tables, a `SessionStore` per backend, for
`runa sessions` and `runa ui`. `runa.db.sessions(...)` resolves that one the same way.
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


# Below `Session`, not above: `session/sqlite.py` subclasses it, so the name has to exist
# first. Re-exported because the local adapter is always importable, where `PostgresSession`
# needs the `postgres` extra and so stays an explicit `runa.session.postgres` import.
from runa.session.sqlite import SQLiteSession  # noqa: E402

__all__ = ["SQLiteSession", "Session"]
