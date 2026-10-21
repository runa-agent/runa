"""Tests for `runa._loop`: one resource per event loop, and `run_sync`'s one background loop."""

import asyncio
import logging
from contextvars import ContextVar

import pytest

from runa import _loop
from runa._loop import LoopCache, run_sync
from runa.exceptions import UserError


class _Resource:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False


async def _close(resource: _Resource) -> None:
    resource.closed = True


def test_run_sync_returns_the_result_and_raises_the_error() -> None:
    """`main`'s outcome is the caller's, either way."""

    async def answer() -> int:
        return 42

    async def fail() -> None:
        raise RuntimeError("boom")

    assert run_sync(answer()) == 42
    with pytest.raises(RuntimeError, match="boom"):
        run_sync(fail())


def test_every_run_sync_shares_one_loop_so_resources_outlive_the_call() -> None:
    """A second call reuses what the first opened, rather than rebuilding it on a new loop."""
    cache: LoopCache[str, _Resource] = LoopCache(close=_close)
    built: list[_Resource] = []

    async def use() -> _Resource:
        return cache.get("a", lambda: built.append(_Resource("a")) or built[-1])

    first, second = run_sync(use()), run_sync(use())

    assert first is second
    assert len(built) == 1
    assert not first.closed


def test_run_sync_sees_the_callers_context() -> None:
    """A `ContextVar` set by the caller is what the coroutine reads on the background thread."""
    var: ContextVar[str] = ContextVar("var", default="unset")
    var.set("caller")

    async def read() -> str:
        return var.get()

    assert run_sync(read()) == "caller"


def test_run_sync_inside_the_background_loop_raises_instead_of_waiting_on_itself() -> None:
    """Code already running on Runa's loop is told to await, not left deadlocked."""

    async def inner() -> None: ...

    async def outer() -> None:
        run_sync(inner())

    with pytest.raises(UserError, match="await it instead"):
        run_sync(outer())


def test_a_timeout_cancels_the_coroutine_and_waits_for_it_to_unwind() -> None:
    """By the time the caller sees the timeout, the coroutine's `finally` has run."""
    unwound: list[bool] = []

    async def slow() -> None:
        try:
            await asyncio.sleep(10)
        finally:
            unwound.append(True)

    with pytest.raises(TimeoutError):
        run_sync(slow(), timeout=0.05)

    assert unwound == [True]


def test_closing_a_loop_closes_every_closable_resource_and_logs_a_refusal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One refusal doesn't keep the rest open; a cache without `close` is left alone."""

    async def refuse(resource: _Resource) -> None:
        raise OSError(f"{resource.name} is already gone")

    stuck: LoopCache[str, _Resource] = LoopCache(close=refuse)
    first: LoopCache[str, _Resource] = LoopCache(close=_close)
    second: LoopCache[str, _Resource] = LoopCache(close=_close)
    plain: LoopCache[str, _Resource] = LoopCache()

    async def open_and_close() -> list[_Resource]:
        stuck.get("a", lambda: _Resource("stuck"))
        opened = [
            first.get("a", lambda: _Resource("a")),
            first.get("b", lambda: _Resource("b")),
            second.get("a", lambda: _Resource("a")),
        ]
        untouched = plain.get("a", lambda: _Resource("plain"))
        await _loop._close_resources(asyncio.get_running_loop())
        return [*opened, untouched]

    with caplog.at_level(logging.WARNING, logger="runa"):
        *closed, untouched = asyncio.run(open_and_close())

    assert all(resource.closed for resource in closed)
    assert not untouched.closed
    assert "closing" in caplog.text


def test_a_plain_asyncio_run_closes_nothing() -> None:
    """Only the background loop's resources are closed at exit; any other loop keeps its own."""
    cache: LoopCache[str, _Resource] = LoopCache(close=_close)
    resource = _Resource("a")

    async def open_one() -> None:
        cache.get("a", lambda: resource)

    asyncio.run(open_one())

    assert not resource.closed
