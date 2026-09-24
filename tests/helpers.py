"""Shared helpers for the test suite."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any


def run[T](awaitable: Awaitable[T]) -> T:
    """Run any awaitable to completion, not just a coroutine.

    `asyncio.run` takes a coroutine specifically. Several things under test are declared as
    returning `Awaitable` (a tool's `on_invoke_tool`, a guardrail function), which is the right
    public contract and a type error to hand straight to `asyncio.run` on Python 3.12. Awaiting
    it inside a real coroutine keeps the contract broad and the call site honest.
    """

    async def _await() -> Any:
        return await awaitable

    return asyncio.run(_await())
