"""Shared helpers for the test suite."""

import asyncio
from collections.abc import Awaitable
from typing import Any

from runa._types import Usage
from runa.run import Run
from runa.tracing import Trace


def finished_run(output: Any = "ok", *, trace: Trace | None = None, **overrides: Any) -> Run:
    """A finished `Run`, for a test that grades or renders one without driving the turn loop.

    `Run`'s required fields are the turn loop's to fill (`run_internal/run_loop._finish`), and a
    test asserting on an eval metric or an evaluation page cares about one or two of them. Pass
    `status=`/`error=`/`_transcript=` for the rest, or `trace=` for a test that reads the trace
    itself -- a run always has one, so an anonymous stand-in stands in when it doesn't matter.
    """
    stand_in = Trace(id="trace_stub", name="stub", start_time=0.0)
    return Run(
        output=output, trace=trace if trace is not None else stand_in, usage=Usage(), **overrides
    )


def run[T](awaitable: Awaitable[T]) -> T:
    """Run any awaitable to completion, not just a coroutine.

    `asyncio.run` takes a coroutine specifically. Several things under test are declared as
    returning `Awaitable` (a tool's `on_invoke_tool`, a guardrail function), which is the right
    public contract and a type error to hand straight to `asyncio.run`. Awaiting
    it inside a real coroutine keeps the contract broad and the call site honest.
    """

    async def _await() -> Any:
        return await awaitable

    return asyncio.run(_await())
