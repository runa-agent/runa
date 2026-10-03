"""`@approval` decorator that turns a plain predicate into a `needs_approval` callable."""

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from runa._types import RunContextWrapper

_NeedsApproval = Callable[[RunContextWrapper[Any], dict[str, Any], str], Awaitable[bool]]


def approval(func: Callable[..., bool | Awaitable[bool]]) -> _NeedsApproval:
    """Turn a predicate into a `@tool(needs_approval=...)` callable.

    Write the predicate like the tool it guards: parameters are looked up by name from the
    tool call's parsed arguments. Name a parameter `ctx` or `call_id` to receive the run
    context or call id instead of a tool argument. Works on `def`, `async def`, and `lambda`.
    """
    names = list(inspect.signature(func).parameters)

    async def wrapper(ctx: RunContextWrapper[Any], params: dict[str, Any], call_id: str) -> bool:
        reserved = {"ctx": ctx, "call_id": call_id}
        bound = {name: reserved[name] if name in reserved else params[name] for name in names}
        result = func(**bound)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)

    return wrapper


__all__ = ["approval"]
