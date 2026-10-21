"""_loop.py: the one sync edge, and one resource per event loop for what cannot outlive its loop.

An `asyncpg.Pool`, a `redis.asyncio.Redis`, an `httpx.AsyncClient` and an MCP `ClientSession` all
open their connections on the event loop running when they are first used, and none survives
that loop closing. A synchronous caller (`Agent.run_sync`, `runa chat`/`eval`/`test`, a trace
export, a `runa traces` read) therefore never gets a loop of its own: `run_sync` runs its
coroutine on one long-lived loop on a daemon thread, shared by every such caller, so what a sync
caller opens lives for the process, exactly as it does for an async app on its own loop.

Several loops can still be alive at once -- that background loop beside an app's own, or a test
suite's `asyncio.run()` per test -- so every long-lived client that holds a connection is held
per loop through `LoopCache`. Nothing here knows what it is holding; callers key their resource
(a URL, a server address, a provider prefix) and share only the lifetime. A resource with
something outside the process on the other end, like an MCP `.stdio` server's subprocess, is
cached with a `close`, and the background loop closes every such resource it holds before the
interpreter exits.
"""

import asyncio
import atexit
import contextvars
import threading
import weakref
from collections.abc import Awaitable, Callable, Coroutine
from concurrent.futures import Future, wait
from dataclasses import dataclass, field
from typing import Any

from runa._log import logger
from runa.exceptions import UserError


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


# The caches holding something to close, per loop. Held strongly, so a resource whose owner was
# dropped is still closed when the process exits.
_closable: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, list[LoopCache[Any, Any]]] = (
    weakref.WeakKeyDictionary()
)


class LoopCache[K, V]:
    """One `V` per key per event loop, built on first use and dropped with its loop.

    Entries are held weakly against the loop, so a loop that has been closed and let go takes
    its resources with it rather than accumulating: a test suite that runs `asyncio.run()` a
    hundred times keeps one pool at a time, not a hundred. A long-lived loop -- an app's own, or
    `run_sync`'s -- gets exactly one resource per key, which is the case that matters.

    Each loop also gets its own `asyncio.Lock`, since a lock is bound to the loop that first
    awaits it.
    """

    def __init__(self, close: Callable[[V], Awaitable[None]] | None = None) -> None:
        """Start empty; nothing is built until a caller asks for a key.

        `close` shuts one resource down. Given one, the resources this cache holds on
        `run_sync`'s loop are closed before the interpreter exits; without one, or on any other
        loop, they fall away with the loop.
        """
        self._close = close
        self._loops: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _Entries[K, V]] = (
            weakref.WeakKeyDictionary()
        )
        self._detached: _Entries[K, V] = _Entries()

    def _entries(self) -> _Entries[K, V]:
        """The current loop's entries, or the loopless ones for a synchronous caller."""
        loop = current_loop()
        if loop is None:
            return self._detached
        entries = self._loops.get(loop)
        if entries is None:
            entries = self._loops[loop] = _Entries()
            if self._close is not None:
                _closable.setdefault(loop, []).append(self)
        return entries

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

    async def _close_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Close every resource this cache holds on `loop`, logging any that refuses to go."""
        entries = self._loops.pop(loop, None)
        if entries is None or self._close is None:
            return
        for value in entries.values.values():
            try:
                await self._close(value)
            except Exception:  # noqa: BLE001 -- one refusal must not leak the rest
                logger.warning("closing %r failed", value, exc_info=True)


async def _close_resources(loop: asyncio.AbstractEventLoop) -> None:
    """Close every closable resource held on `loop`, whichever cache or caller opened it."""
    for cache in _closable.pop(loop, []):
        await cache._close_loop(loop)


_CLOSE_TIMEOUT = 5.0
"""How long the interpreter's exit waits for the background loop's resources to close."""

_background: tuple[asyncio.AbstractEventLoop, threading.Thread] | None = None
_background_lock = threading.Lock()


def _background_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """The daemon loop every `run_sync` runs on, and its thread, started on first use.

    Started again in a forked child, whose copy of the parent's thread is not running.
    """
    global _background
    with _background_lock:
        if _background is None or not _background[1].is_alive():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=loop.run_forever, name="runa-loop", daemon=True)
            thread.start()
            _background = (loop, thread)
            atexit.register(_shut_down, _background)
        return _background


def _shut_down(background: tuple[asyncio.AbstractEventLoop, threading.Thread]) -> None:
    """Close what the background loop holds and stop it, if it is still this process's loop."""
    loop, thread = background
    if background is not _background or not thread.is_alive():
        return
    try:
        asyncio.run_coroutine_threadsafe(_close_resources(loop), loop).result(_CLOSE_TIMEOUT)
    except Exception:  # noqa: BLE001 -- the interpreter is exiting either way
        logger.warning("closing the background loop's resources failed", exc_info=True)
    loop.call_soon_threadsafe(loop.stop)


def run_sync[T](main: Coroutine[Any, Any, T], *, timeout: float | None = None) -> T:
    """Run `main` on the background loop and block the calling thread until it finishes.

    `main` runs in a copy of the caller's context, so a `tracing.trace` block, an
    `observe.scoped(...)` override or any other `ContextVar` the caller set is what it sees.
    Whatever it opens stays open on that loop for later calls to reuse.

    A `KeyboardInterrupt` (or `timeout` running out) cancels `main` and waits for it to unwind
    before raising, so its `finally` blocks have run by the time the caller sees it.

    Works from a thread already running a loop of its own -- a notebook's, or `runa serve`'s --
    at the cost of stalling that loop for the duration. Called from the background loop's own
    thread, it would wait on itself forever, so it raises `UserError` instead: code already
    running there awaits rather than calling this.
    """
    loop, thread = _background_loop()
    if threading.current_thread() is thread:
        main.close()
        raise UserError(
            "run_sync was called from inside a coroutine Runa is already running; "
            "await it instead (`await agent.run(...)`, not `agent.run_sync(...)`)"
        )

    outcome: Future[T] = Future()
    context = contextvars.copy_context()
    started: list[asyncio.Task[T]] = []

    def settle(task: asyncio.Task[T]) -> None:
        if task.cancelled():
            outcome.set_exception(asyncio.CancelledError())
        elif (error := task.exception()) is not None:
            outcome.set_exception(error)
        else:
            outcome.set_result(task.result())

    def start() -> None:
        task = loop.create_task(main, context=context)
        task.add_done_callback(settle)
        started.append(task)

    loop.call_soon_threadsafe(start)
    try:
        return outcome.result(timeout)
    except BaseException:
        if not outcome.done():
            # Scheduled after `start`, so the task exists by the time this runs.
            loop.call_soon_threadsafe(lambda: started[0].cancel())
            wait([outcome])
        raise


__all__ = ["LoopCache", "current_loop", "run_sync"]
