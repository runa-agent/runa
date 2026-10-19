"""Tests for `runa._loop`: one resource per event loop, and `run` closing what must be closed."""

import asyncio
import logging

import pytest

from runa import _loop
from runa._loop import LoopCache


class _Resource:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False


async def _close(resource: _Resource) -> None:
    resource.closed = True


def test_run_closes_every_closable_resource_opened_on_its_loop() -> None:
    """Every key of every closable cache is closed, whichever cache or caller opened it."""
    first: LoopCache[str, _Resource] = LoopCache(close=_close)
    second: LoopCache[str, _Resource] = LoopCache(close=_close)

    async def open_three() -> list[_Resource]:
        return [
            first.get("a", lambda: _Resource("a")),
            first.get("b", lambda: _Resource("b")),
            second.get("a", lambda: _Resource("a")),
        ]

    opened = _loop.run(open_three())

    assert all(resource.closed for resource in opened)


def test_run_closes_even_when_the_coroutine_raises() -> None:
    """A failing coroutine still takes its resources down; its exception is the caller's."""
    cache: LoopCache[str, _Resource] = LoopCache(close=_close)
    resource = _Resource("a")

    async def open_then_fail() -> None:
        cache.get("a", lambda: resource)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _loop.run(open_then_fail())

    assert resource.closed


def test_a_resource_that_fails_to_close_is_logged_and_does_not_stop_the_rest(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The coroutine's result is still returned, and the other resources are still closed."""

    async def refuse(resource: _Resource) -> None:
        raise OSError(f"{resource.name} is already gone")

    stuck: LoopCache[str, _Resource] = LoopCache(close=refuse)
    fine: LoopCache[str, _Resource] = LoopCache(close=_close)
    resource = _Resource("fine")

    async def open_both() -> str:
        stuck.get("a", lambda: _Resource("stuck"))
        fine.get("a", lambda: resource)
        return "done"

    with caplog.at_level(logging.WARNING, logger="runa"):
        assert _loop.run(open_both()) == "done"

    assert resource.closed
    assert "closing" in caplog.text


def test_a_cache_without_close_leaves_its_resources_alone() -> None:
    """Only a cache built with `close` is closed; the rest fall away with their loop."""
    cache: LoopCache[str, _Resource] = LoopCache()
    resource = _Resource("a")

    async def open_one() -> None:
        cache.get("a", lambda: resource)

    _loop.run(open_one())

    assert not resource.closed


def test_a_plain_asyncio_run_closes_nothing() -> None:
    """Closing is `run`'s, for a loop its caller owns; any other loop keeps what it opened."""
    cache: LoopCache[str, _Resource] = LoopCache(close=_close)
    resource = _Resource("a")

    async def open_one() -> None:
        cache.get("a", lambda: resource)

    asyncio.run(open_one())

    assert not resource.closed
