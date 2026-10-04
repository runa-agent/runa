"""`runa.session`: `SessionABC`, the conversation history a run reads and appends to.

Which implementation a deployment gets is `runa.db`'s decision, not the call site's:
`session/sqlite.py` locally, `session/postgres.py` when `RUNA_DATABASE_URL` points at a shared
database. `runa serve`, `runa chat` and `Agent.run` all take whatever `db.session(...)` hands
them, so moving a deployment to Postgres is one environment variable rather than an edit.

`session/store.py` is the read side of the same tables, a `SessionStore` per backend, for
`runa sessions` and `runa ui`. `runa.db.sessions(...)` resolves that one the same way.
"""

from abc import ABC, abstractmethod

from runa._types import TResponseInputItem


class SessionABC(ABC):
    """What `run_internal` needs to persist and replay conversation history across turns.

    `user_id` is optional and unrelated to history: `run_internal` reads it (when set) to scope
    automatic `Agent.memory` retrieval/persistence to one user, so app code doesn't have to
    thread a `user_id` through `Memory.remember`/`.search` itself. `None` means memory that goes
    through this session lands in its own user-less scope, not everyone's.
    """

    session_id: str
    user_id: str | None

    @abstractmethod
    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        """Return this session's items, oldest first, capped at the latest `limit` if given."""

    @abstractmethod
    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """Append `items` to this session's history."""

    @abstractmethod
    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return this session's most recent item, or `None` if it has none."""

    @abstractmethod
    async def clear_session(self) -> None:
        """Delete this session and all of its items."""

    async def set_items(self, items: list[TResponseInputItem]) -> None:
        """Replace this session's entire history with `items`, `Agent(compact=...)`'s hook.

        A concrete default built from `clear_session`/`add_items`, not `@abstractmethod`: an
        existing custom `SessionABC` gets this for free, with no new method it's forced to
        implement. Override for a single-transaction replace if that matters for your store, the
        way `SQLiteSession` does.
        """
        await self.clear_session()
        await self.add_items(items)


# Below `SessionABC`, not above: `session/sqlite.py` subclasses it, so the name has to exist
# first. Re-exported because the local adapter is always importable, where `PostgresSession`
# needs the `postgres` extra and so stays an explicit `runa.session.postgres` import.
from runa.session.sqlite import SQLiteSession  # noqa: E402

__all__ = ["SQLiteSession", "SessionABC"]
