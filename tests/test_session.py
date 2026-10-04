"""What `SQLiteSession` promises beyond the session contract every backend answers.

The round-trip, `limit`, `pop_item`, `set_items`, `clear_session` and isolation behavior is
`tests/contracts/session.py`, driven over this adapter (among others) by
`tests/test_session_store.py`. What is left here is this adapter's own: that clearing a session
removes its `agent_sessions` row rather than just its messages, and that a hand-written
`SessionABC` gets `set_items` without implementing it.
"""

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from runa.session import SessionABC, SQLiteSession


def test_clear_session_drops_its_items_and_row(tmp_path: Path) -> None:
    """`clear_session` deletes the session's messages and its `agent_sessions` row."""
    db_path = tmp_path / "runa.db"
    session = SQLiteSession("s1", db_path=db_path)

    async def _run():
        await session.add_items([{"role": "user", "content": "hi"}])
        await session.clear_session()
        return await session.get_items()

    assert asyncio.run(_run()) == []
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute("SELECT 1 FROM agent_sessions WHERE session_id = ?", ("s1",)).fetchone()
    assert row is None


def test_sessionabc_default_set_items_works_without_an_override(tmp_path: Path) -> None:
    """A custom `SessionABC` gets `set_items` for free from `clear_session`/`add_items`.

    No new abstract method to implement -- an existing subclass that predates `set_items` still
    gets correct (if not transactional) behavior for it, unlike `SQLiteSession`'s own override.
    """

    class PlainSession(SessionABC):
        def __init__(self) -> None:
            self.session_id = "s1"
            self.user_id = None
            self._items: list[Any] = []

        async def get_items(self, limit: int | None = None) -> list[Any]:
            return list(self._items) if limit is None else self._items[-limit:]

        async def add_items(self, items: list[Any]) -> None:
            self._items.extend(items)

        async def pop_item(self) -> Any | None:
            return self._items.pop() if self._items else None

        async def clear_session(self) -> None:
            self._items = []

    session = PlainSession()

    async def _run() -> list[Any]:
        await session.add_items([{"role": "user", "content": "old"}])
        await session.set_items([{"role": "user", "content": "new"}])
        return await session.get_items()

    assert asyncio.run(_run()) == [{"role": "user", "content": "new"}]
