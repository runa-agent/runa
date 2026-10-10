"""`runa.lifecycle`: `Hooks`, and live `logging`-module output for a run.

For a structured, persisted, queryable record of a run instead, see `runa.tracing`. That runs
unconditionally regardless of `hooks`; this module is just console/log lines while a run happens.

These hooks are the default, so they log to whatever handler the application configured. At INFO
they name what happened and never carry content: an agent's answer and a tool's result are user
data, and a production app running at INFO should not be writing them to stdout. Content is logged
at DEBUG only, and even there it goes through `runa.tracing`'s redact/truncate policy, so
`observe(redact=[...])` and `observe(capture_outputs=False)` govern these lines too.
"""

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from runa._items import ConversationItem
from runa._types import RunContextWrapper
from runa.tool import FunctionTool

logger = logging.getLogger("runa")

__all__ = [
    "Hooks",
    "LoggingHooks",
]


def _content(value: Any, *, limit: int | None = None) -> str:
    """Render `value` for a DEBUG log line under `runa.tracing`'s active privacy policy.

    Imported lazily: `runa.tracing.manual` reaches back into this module for `logger`, so a
    top-level import here would close the cycle.
    """
    from runa.tracing.config import apply_policy, output_limit

    return repr(apply_policy(value, max_bytes=limit if limit is not None else output_limit()))


class Hooks[TContext]:
    """Lifecycle callbacks for a run; every method is a no-op unless overridden.

    Scope is where an instance is put, not which class it is:

    * `Agent.run(hooks=...)` fires for every agent in that one run, the ones reached by
      handoff/delegate included. Use it for cross-cutting concerns -- metrics, a single audit
      log for the whole run.
    * An `Agent` subclass's `hooks` attribute fires only for that agent, in every run it takes
      part in. Use it for a concern that belongs to one agent's identity.

    Both scopes fire for the same event, run-scoped first, and read the same arguments.
    `on_handoff` is the one event an agent-scoped instance doesn't see for every turn of its
    own agent: it notifies the target, the agent the run was handed *to*, and never the sender.

    `LoggingHooks` (below) is what a run with no `hooks=` gets.
    """

    async def on_agent_start(self, context: RunContextWrapper[TContext], agent: Any) -> None:
        """Called right before `agent` starts (or resumes, after a handoff) running."""

    async def on_agent_end(
        self, context: RunContextWrapper[TContext], agent: Any, output: Any
    ) -> None:
        """Called once `agent` has produced its final output for the run."""

    async def on_handoff(
        self, context: RunContextWrapper[TContext], from_agent: Any, to_agent: Any
    ) -> None:
        """Called when `from_agent` hands the run off to `to_agent`."""

    async def on_tool_start(
        self, context: RunContextWrapper[TContext], agent: Any, tool: FunctionTool
    ) -> None:
        """Called right before `tool` runs."""

    async def on_tool_end(
        self, context: RunContextWrapper[TContext], agent: Any, tool: FunctionTool, result: object
    ) -> None:
        """Called once `tool` has returned (or raised) `result`."""

    async def on_llm_start(
        self,
        context: RunContextWrapper[TContext],
        agent: Any,
        system_prompt: str | None,
        input_items: list[ConversationItem],
    ) -> None:
        """Called right before `agent` calls the model."""

    async def on_llm_end(
        self, context: RunContextWrapper[TContext], agent: Any, response: Any
    ) -> None:
        """Called once the model call has returned `response`."""


_EVENTS = frozenset(name for name in vars(Hooks) if name.startswith("on_"))


class _Dispatch[TContext]:
    """Fires one lifecycle event at both scopes: the run's hooks, then the agent's own.

    `runa.run_internal` holds one of these for a whole run and calls it exactly where it would
    call `Hooks`, so the points that fire an event stay ignorant of there being two scopes.
    One `Hooks` class for both scopes is what keeps this a forwarder over `Hooks`' own method
    names instead of a second copy of them: an event added to `Hooks` dispatches with no edit
    here, and an event this doesn't know raises `AttributeError` rather than firing nowhere.

    The one thing it knows per event is which agent's own `hooks` to fire, read out of that
    call's arguments: the agent acting, except for a handoff, which notifies the target.

    Run-scoped hooks fire first, so a run-wide audit log records an event before any one agent's
    callback can raise out of it.
    """

    _NOTIFIED = {"on_handoff": 1}
    """Whose `hooks` an event fires, as an index into its arguments past `context`; default 0."""

    def __init__(self, run_hooks: Hooks[TContext]) -> None:
        self.run_hooks = run_hooks

    def __getattr__(self, event: str) -> Callable[..., Coroutine[Any, Any, None]]:
        """A `Hooks` event as one callable, firing it at the run scope then the agent's own."""
        if event not in _EVENTS:
            raise AttributeError(event)

        async def fire(context: RunContextWrapper[TContext], *args: Any) -> None:
            await getattr(self.run_hooks, event)(context, *args)
            agent = args[self._NOTIFIED.get(event, 0)]
            if (own := getattr(agent, "hooks", None)) is not None:
                await getattr(own, event)(context, *args)

        return fire


class LoggingHooks(Hooks[Any]):
    """Logs each lifecycle event through the standard `logging` module, under the `runa` logger.

    `Agent.run`/`run_sync` use an instance of this as the default `hooks`, so every run is
    logged without the caller having to ask; passing an explicit `hooks` overrides it. Assigned
    to an `Agent` subclass's `hooks` instead, it logs that one agent's events.
    """

    async def on_agent_start(self, context: RunContextWrapper[Any], agent: Any) -> None:
        """Log that `agent` is about to run."""
        logger.info("agent start: %s", agent.name)

    async def on_agent_end(self, context: RunContextWrapper[Any], agent: Any, output: Any) -> None:
        """Log that `agent` finished: its name at INFO, its output at DEBUG if policy allows."""
        from runa.tracing.config import capture_outputs, output_limit

        logger.info("agent end: %s", agent.name)
        if capture_outputs() and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "agent output: %s -> %s", agent.name, _content(output, limit=output_limit())
            )

    async def on_handoff(
        self, context: RunContextWrapper[Any], from_agent: Any, to_agent: Any
    ) -> None:
        """Log a handoff as `from_agent -> to_agent`."""
        logger.info("handoff: %s -> %s", from_agent.name, to_agent.name)

    async def on_tool_start(
        self, context: RunContextWrapper[Any], agent: Any, tool: FunctionTool
    ) -> None:
        """Log that `tool` is about to run."""
        logger.info("tool start: %s (%s)", tool.name, agent.name)

    async def on_tool_end(
        self, context: RunContextWrapper[Any], agent: Any, tool: FunctionTool, result: object
    ) -> None:
        """Log that `tool` finished; its result only at DEBUG, under the tracing policy."""
        from runa.tracing.config import capture_outputs, tool_result_limit

        logger.info("tool end: %s (%s)", tool.name, agent.name)
        if capture_outputs() and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "tool output: %s -> %s", tool.name, _content(result, limit=tool_result_limit())
            )

    async def on_llm_start(
        self,
        context: RunContextWrapper[Any],
        agent: Any,
        system_prompt: str | None,
        input_items: list[ConversationItem],
    ) -> None:
        """Log that `agent` is about to call the model."""
        logger.debug("llm start: %s", agent.name)

    async def on_llm_end(self, context: RunContextWrapper[Any], agent: Any, response: Any) -> None:
        """Log that `agent`'s model call returned."""
        logger.debug("llm end: %s", agent.name)
