"""`runa.lifecycle`: `Hooks`, and live `logging`-module output for a run.

For a structured, persisted, queryable record of a run instead, see `runa.tracing`. That runs
unconditionally regardless of `hooks`; this module is just console/log lines while a run happens.

These hooks are the default, so they log to whatever handler the application configured. At INFO
they name what happened and never carry content: an agent's answer and a tool's result are user
data, and a production app running at INFO should not be writing them to stdout. Content is logged
at DEBUG only, and even there it goes through `runa.tracing`'s redact/truncate policy, so
`observe(redact=[...])` and `observe(capture_outputs=False)` govern these lines too.

The `Logger` these write to is `runa._log`'s, not this module's: `logger` is a leaf every layer
can import without taking `Hooks` with it, and keeping it there is what lets this module read
the tracing policy at the top of the file rather than inside each method.

The `agent` every callback is handed is the `Agent` itself. That import is type-only, and has to
be: an `Agent` carries the `hooks` below.
"""

import logging
from typing import TYPE_CHECKING, Any

from runa._items import ConversationItem
from runa._log import logger
from runa._types import RunContext
from runa.tool import FunctionTool
from runa.tracing.config import kept

if TYPE_CHECKING:
    from runa.agent import Agent

__all__ = [
    "Hooks",
    "LoggingHooks",
]


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

    async def on_agent_start(self, context: RunContext[TContext], agent: Agent) -> None:
        """Called right before `agent` starts (or resumes, after a handoff) running."""

    async def on_agent_end(self, context: RunContext[TContext], agent: Agent, output: Any) -> None:
        """Called once `agent` has produced its final output for the run."""

    async def on_handoff(
        self, context: RunContext[TContext], from_agent: Agent, to_agent: Agent
    ) -> None:
        """Called when `from_agent` hands the run off to `to_agent`."""

    async def on_tool_start(
        self, context: RunContext[TContext], agent: Agent, tool: FunctionTool
    ) -> None:
        """Called right before `tool` runs."""

    async def on_tool_end(
        self,
        context: RunContext[TContext],
        agent: Agent,
        tool: FunctionTool,
        result: object,
    ) -> None:
        """Called once `tool` has returned (or raised) `result`."""

    async def on_llm_start(
        self,
        context: RunContext[TContext],
        agent: Agent,
        system_prompt: str | None,
        input_items: list[ConversationItem],
    ) -> None:
        """Called right before `agent` calls the model."""

    async def on_llm_end(self, context: RunContext[TContext], agent: Agent, response: Any) -> None:
        """Called once the model call has returned `response`."""


class _Dispatch[TContext](Hooks[TContext]):
    """Fires each lifecycle event at both scopes: the run's hooks, then the agent's own.

    `runa.run_internal` holds one of these for a whole run and calls it exactly where it would
    call `Hooks`. Every event is overridden here, so an event added to `Hooks` needs its own
    override below or it fires at neither scope.

    Each event fires the `hooks` of the agent acting, except a handoff, which notifies the target.
    Run-scoped hooks fire first, so a run-wide audit log records an event before any one agent's
    callback can raise out of it.
    """

    def __init__(self, run_hooks: Hooks[TContext]) -> None:
        self.run_hooks = run_hooks

    def _scopes(self, agent: Agent) -> list[Hooks[TContext]]:
        """The hooks an event about `agent` fires, in order: the run's, then `agent`'s own."""
        return [self.run_hooks] if agent.hooks is None else [self.run_hooks, agent.hooks]

    async def on_agent_start(self, context: RunContext[TContext], agent: Agent) -> None:
        for hooks in self._scopes(agent):
            await hooks.on_agent_start(context, agent)

    async def on_agent_end(self, context: RunContext[TContext], agent: Agent, output: Any) -> None:
        for hooks in self._scopes(agent):
            await hooks.on_agent_end(context, agent, output)

    async def on_handoff(
        self, context: RunContext[TContext], from_agent: Agent, to_agent: Agent
    ) -> None:
        for hooks in self._scopes(to_agent):
            await hooks.on_handoff(context, from_agent, to_agent)

    async def on_tool_start(
        self, context: RunContext[TContext], agent: Agent, tool: FunctionTool
    ) -> None:
        for hooks in self._scopes(agent):
            await hooks.on_tool_start(context, agent, tool)

    async def on_tool_end(
        self, context: RunContext[TContext], agent: Agent, tool: FunctionTool, result: object
    ) -> None:
        for hooks in self._scopes(agent):
            await hooks.on_tool_end(context, agent, tool, result)

    async def on_llm_start(
        self,
        context: RunContext[TContext],
        agent: Agent,
        system_prompt: str | None,
        input_items: list[ConversationItem],
    ) -> None:
        for hooks in self._scopes(agent):
            await hooks.on_llm_start(context, agent, system_prompt, input_items)

    async def on_llm_end(self, context: RunContext[TContext], agent: Agent, response: Any) -> None:
        for hooks in self._scopes(agent):
            await hooks.on_llm_end(context, agent, response)


class LoggingHooks(Hooks[Any]):
    """Logs each lifecycle event through the standard `logging` module, under the `runa` logger.

    `Agent.run`/`run_sync` use an instance of this as the default `hooks`, so every run is
    logged without the caller having to ask; passing an explicit `hooks` overrides it. Assigned
    to an `Agent` subclass's `hooks` instead, it logs that one agent's events.
    """

    async def on_agent_start(self, context: RunContext[Any], agent: Agent) -> None:
        """Log that `agent` is about to run."""
        logger.info("agent start: %s", agent.name)

    async def on_agent_end(self, context: RunContext[Any], agent: Agent, output: Any) -> None:
        """Log that `agent` finished: its name at INFO, its output at DEBUG if policy allows."""
        logger.info("agent end: %s", agent.name)
        if logger.isEnabledFor(logging.DEBUG) and (shown := kept(output, "output")) is not None:
            logger.debug("agent output: %s -> %r", agent.name, shown)

    async def on_handoff(
        self, context: RunContext[Any], from_agent: Agent, to_agent: Agent
    ) -> None:
        """Log a handoff as `from_agent -> to_agent`."""
        logger.info("handoff: %s -> %s", from_agent.name, to_agent.name)

    async def on_tool_start(
        self, context: RunContext[Any], agent: Agent, tool: FunctionTool
    ) -> None:
        """Log that `tool` is about to run."""
        logger.info("tool start: %s (%s)", tool.name, agent.name)

    async def on_tool_end(
        self, context: RunContext[Any], agent: Agent, tool: FunctionTool, result: object
    ) -> None:
        """Log that `tool` finished; its result only at DEBUG, under the tracing policy."""
        logger.info("tool end: %s (%s)", tool.name, agent.name)
        if (
            logger.isEnabledFor(logging.DEBUG)
            and (shown := kept(result, "tool_result")) is not None
        ):
            logger.debug("tool output: %s -> %r", tool.name, shown)

    async def on_llm_start(
        self,
        context: RunContext[Any],
        agent: Agent,
        system_prompt: str | None,
        input_items: list[ConversationItem],
    ) -> None:
        """Log that `agent` is about to call the model."""
        logger.debug("llm start: %s", agent.name)

    async def on_llm_end(self, context: RunContext[Any], agent: Agent, response: Any) -> None:
        """Log that `agent`'s model call returned."""
        logger.debug("llm end: %s", agent.name)
