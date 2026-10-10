"""_loop.py: one resource per event loop, for the clients and pools that cannot outlive theirs.

An `asyncpg.Pool`, a `redis.asyncio.Redis`, an `httpx.AsyncClient` and an MCP `ClientSession` all
open their connections on the event loop that was running when they first used them, and none of
them survives that loop being closed. A second `asyncio.run()` in the same process is enough to
hit it -- `Agent.run_sync` makes one per call -- and the symptom is a hang acquiring a connection
tied to a dead loop, or "Event loop is closed" on the next request.

The rule is the same for all four: hold the resource against the loop it belongs to, and build a
new one when the current loop is not that one. It was written three times, in three slightly
different ways, and the oldest of them keyed by `id(loop)`, which both kept an entry per
closed loop forever and could in principle hand a resource to an unrelated loop that happened to
reuse the address. This is that rule, once. A new long-lived client belongs here too: if it holds
a connection, it is loop-scoped, and the exception is the one that has to be argued for.

Nothing here knows what it is holding. `db/pool.py` keys pools by URL, `cache/redis.py` its client
by URL, `mcp.py` its sessions by the server's address, `ModelProvider` its HTTP clients by provider
prefix, and `embeddings.py` its one client by base URL; what they share is the lifetime, not the
resource.
"""

import asyncio
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


def current_loop() -> asyncio.AbstractEventLoop | None:
    """The running event loop, or `None` when the caller is synchronous.

    `asyncio.get_running_loop`'s `RuntimeError` is not an error here: `ModelProvider.get_model`
    does no I/O and is called at `Agent` construction time, outside any loop, which is ordinary.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


@dataclass
class _Entries[K, V]:
    """One loop's resources, with the lock that guards building them on that loop."""

    values: dict[K, V] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class LoopCache[K, V]:
    """One `V` per key per event loop, built on first use and dropped with its loop.

    Entries are held weakly against the loop, so a loop that has been closed and let go takes
    its resources with it rather than accumulating: a test suite that runs `asyncio.run()` a
    hundred times keeps one pool at a time, not a hundred. A long-lived deployment's single loop
    still gets exactly one resource per key, which is the case that matters.

    Several loops can be alive at once -- a synchronous CLI read path runs on `db/pool.py`'s
    background loop while an async app runs on its own -- so this is a resource per loop rather
    than one resource and a staleness check. Each loop also gets its own `asyncio.Lock`, since a
    lock is bound to the loop that first awaits it.
    """

    def __init__(self) -> None:
        """Start empty; nothing is built until a caller asks for a key."""
        self._loops: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _Entries[K, V]] = (
            weakref.WeakKeyDictionary()
        )
        self._detached: _Entries[K, V] = _Entries()

    def _entries(self) -> _Entries[K, V]:
        """The current loop's entries, or the loopless ones for a synchronous caller."""
        loop = current_loop()
        if loop is None:
            return self._detached
        return self._loops.setdefault(loop, _Entries())

    def get(self, key: K, factory: Callable[[], V]) -> V:
        """Return `key`'s resource on the current loop, calling `factory` if there is none.

        A resource built while no loop was running is adopted by the first loop that asks for it,
        rather than shared by every loop that ever does: the `httpx.AsyncClient` created when an
        `Agent` is constructed is the same one its first run uses, and a second loop gets its own.
        """
        entries = self._entries()
        if key in entries.values:
            return entries.values[key]
        if entries is not self._detached and key in self._detached.values:
            entries.values[key] = self._detached.values.pop(key)
            return entries.values[key]
        entries.values[key] = factory()
        return entries.values[key]

    async def aget(self, key: K, factory: Callable[[], Awaitable[V]]) -> V:
        """Return `key`'s resource on the current loop, awaiting `factory` if there is none.

        For the resources that take I/O to build: two coroutines on one loop racing for the same
        key open one pool between them, not two.
        """
        entries = self._entries()
        if key in entries.values:
            return entries.values[key]
        async with entries.lock:
            if key not in entries.values:
                entries.values[key] = await factory()
        return entries.values[key]

    def pop(self, key: K) -> V | None:
        """Remove and return `key`'s resource on the current loop, if it has one.

        The escape hatch for a caller that has to shut a resource down rather than let it fall
        away with its loop, like a test closing the pool it opened so a live database is not left
        holding those connections for the rest of the session.
        """
        return self._entries().values.pop(key, None)


__all__ = ["LoopCache", "current_loop"]
