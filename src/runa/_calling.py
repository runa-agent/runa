"""_calling.py: how Runa calls an app's own plain function, the one rule every caller shares.

`call_app` runs a function the way every hook point does: a coroutine function is awaited on the
loop, anything else runs in a worker thread, so a sync function doing blocking I/O (a sync HTTP
call, a blocking DB driver) can't stall every other concurrent run. Whatever comes back awaitable
(an `async` lambda, a callable object with an `async __call__`) is awaited too.

`by_name` adds the binding a tool call's arguments get, for `@tool` and `@approval` alike: each
parameter is looked up by name in the call's parsed arguments, a parameter named in
`RESERVED_PARAMS` receives the run context or call id instead, and one the model didn't supply is
left out so its default applies (or Python's own `TypeError` names it, when it has none).

Imports nothing from `runa` at runtime: `runa.approval` and `runa.tool` both import it.
"""

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from runa._types import RunContext

RESERVED_PARAMS = ("ctx", "call_id")
"""The parameter names that receive the run context and the call id, never a tool argument."""


async def call_app(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call app code `func`, off the loop unless it's a coroutine function; await its result."""
    if inspect.iscoroutinefunction(func):
        result = await func(*args, **kwargs)
    else:
        result = await asyncio.to_thread(func, *args, **kwargs)
    return await result if inspect.isawaitable(result) else result


def by_name(
    func: Callable[..., Any],
) -> Callable[[RunContext[Any], dict[str, Any], str], Awaitable[Any]]:
    """`func` as a `(ctx, args, call_id)` callable, bound by name and run by `call_app`."""
    names = list(inspect.signature(func).parameters)

    async def invoke(ctx: RunContext[Any], args: dict[str, Any], call_id: str) -> Any:
        reserved = dict(zip(RESERVED_PARAMS, (ctx, call_id), strict=True))
        bound = {name: reserved[name] for name in names if name in reserved}
        bound |= {name: args[name] for name in names if name not in reserved and name in args}
        return await call_app(func, **bound)

    return invoke
