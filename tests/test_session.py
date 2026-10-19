"""What `SQLiteSession` promises beyond the session contract every backend answers.

The round-trip, `set_items`, `clear_session` and isolation behavior is
`tests/contracts/session.py`, driven over this adapter (among others) by
`tests/test_session_store.py`. What is left here is this adapter's own: that clearing a session
removes its `agent_sessions` row rather than just its messages, that its blocking `sqlite3` calls
happen off the event loop, and that a hand-written `Session` gets `clear_session` without
implementing it.
"""

import asyncio
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from runa.session import Session, SQLiteSession


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


def test_a_session_call_leaves_the_event_loop_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`sqlite3` is blocking, so the `await` has to be a real one.

    On `runa serve`'s default deployment this adapter runs on the loop that is also serving every
    other request, and a session-backed turn does three of these. A body that stalls the loop
    makes one turn's file I/O every concurrent turn's latency, which is what the sync `@tool` and
    `@guardrail` offloads already prevent for application code.

    Asserted two ways, because each alone can pass for the wrong reason: the body runs on another
    thread, and a task scheduled alongside it actually gets to run while it is in flight.
    """
    loop_thread = threading.get_ident()
    bodies: list[int] = []
    original = SQLiteSession._connect

    def slow_connect(self: SQLiteSession) -> sqlite3.Connection:
        bodies.append(threading.get_ident())
        time.sleep(0.05)
        return original(self)

    monkeypatch.setattr(SQLiteSession, "_connect", slow_connect)
    session = SQLiteSession("s1", db_path=tmp_path / "runa.db")

    async def _run() -> int:
        ticks = 0

        async def tick() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.005)
                ticks += 1

        ticking = asyncio.create_task(tick())
        await session.add_items([{"role": "user", "content": "hi"}])
        ticking.cancel()
        return ticks

    assert asyncio.run(_run()) > 0
    assert bodies and loop_thread not in bodies


def test_a_custom_session_gets_clear_session_from_set_items() -> None:
    """A `Session` implementing only the three abstract methods can still be cleared."""

    class PlainSession(Session):
        def __init__(self) -> None:
            self.session_id = "s1"
            self.user_id = None
            self._items: list[Any] = []

        async def get_items(self) -> list[Any]:
            return list(self._items)

        async def add_items(self, items: list[Any]) -> None:
            self._items.extend(items)

        async def set_items(self, items: list[Any]) -> None:
            self._items = list(items)

    session = PlainSession()

    async def _run() -> list[Any]:
        await session.add_items([{"role": "user", "content": "old"}])
        await session.clear_session()
        return await session.get_items()

    assert asyncio.run(_run()) == []
