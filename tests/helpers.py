"""Shared helpers for the test suite."""

import asyncio
from collections.abc import Awaitable
from typing import Any

from runa._types import RunContextWrapper, Usage
from runa.run import Run
from runa.tracing import Trace


def finished_run(output: Any = "ok", *, trace: Trace | None = None, **overrides: Any) -> Run:
    """A finished `Run`, for a test that grades or renders one without driving the turn loop.

    `Run`'s required fields are the turn loop's to fill (`run_internal/run_loop._finish`), and a
    test asserting on an eval metric or an evaluation page cares about one or two of them. Pass
    `status=`/`error=`/`_tool_calls=` for the rest.
    """
    return Run(output=output, trace=trace, usage=Usage(), **overrides)


def trace_of(result: Run) -> Trace:
    """`result.trace`, narrowed away from its public `| None`.

    A run that reached the turn loop always finishes with a trace. `Run.trace` is optional
    because `Agent.run` also reports a failure that never got that far (see `Agent._failed`),
    so a test that did reach the loop asserts the difference away rather than carrying it.
    """
    assert result.trace is not None
    return result.trace


def context_of(result: Run) -> RunContextWrapper:
    """`result._context_wrapper`, narrowed, for a test reading what the loop accumulated on it.

    Private on `Run` because a caller has `usage` and the guardrail audit lists already; the loop's
    own tests read the wrapper itself to check usage accounting and the call-id replay guard.
    """
    assert result._context_wrapper is not None
    return result._context_wrapper


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
