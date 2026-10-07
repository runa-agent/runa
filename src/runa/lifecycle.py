"""`runa.lifecycle`: `RunHooks`/`AgentHooks`, and live `logging`-module output for a run.

For a structured, persisted, queryable record of a run instead, see `runa.tracing`. That runs
unconditionally regardless of `hooks`; this module is just console/log lines while a run happens.

These hooks are the default, so they log to whatever handler the application configured. At INFO
they name what happened and never carry content: an agent's answer and a tool's result are user
data, and a production app running at INFO should not be writing them to stdout. Content is logged
at DEBUG only, and even there it goes through `runa.tracing`'s redact/truncate policy, so
`observe(redact=[...])` and `observe(capture_outputs=False)` govern these lines too.
"""

import logging
from typing import Any

from runa._items import ConversationItem
from runa._types import RunContextWrapper
from runa.tool import FunctionTool

logger = logging.getLogger("runa")

__all__ = [
    "AgentHooks",
    "LoggingAgentHooks",
    "LoggingRunHooks",
    "RunHooks",
]


def _content(value: Any, *, limit: int | None = None) -> str:
    """Render `value` for a DEBUG log line under `runa.tracing`'s active privacy policy.

    Imported lazily: `runa.tracing.manual` reaches back into this module for `logger`, so a
    top-level import here would close the cycle.
    """
    from runa.tracing.config import apply_policy, output_limit

    return repr(apply_policy(value, max_bytes=limit if limit is not None else output_limit()))


def _log_agent_start(agent: Any) -> None:
    """Log that `agent` is about to run; the one event both hook scopes name differently."""
    logger.info("agent start: %s", agent.name)


def _log_agent_end(agent: Any, output: Any) -> None:
    """Log that `agent` finished: its name at INFO, its output at DEBUG if the policy allows."""
    from runa.tracing.config import capture_outputs, output_limit

    logger.info("agent end: %s", agent.name)
    if capture_outputs() and logger.isEnabledFor(logging.DEBUG):
        logger.debug("agent output: %s -> %s", agent.name, _content(output, limit=output_limit()))


def _log_handoff(source: Any, target: Any) -> None:
    """Log a handoff as `source -> target`, whichever scope's argument order it arrived in."""
    logger.info("handoff: %s -> %s", source.name, target.name)


class RunHooks[TContext]:
    """Lifecycle callbacks for one run; every method is a no-op unless overridden.

    Passed as `hooks=` to `Agent.run`/`run_sync`/`run_streamed`, `runa.run_internal` calls these
    as the run progresses. `LoggingRunHooks` (below) is the default when no `hooks` is given.
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


class AgentHooks[TContext]:
    """Lifecycle callbacks scoped to one `Agent` subclass, via its `hooks` class attribute.

    Every method is a no-op unless overridden. Unlike `RunHooks` (passed per-call), this fires
    only for the agent it's assigned to, not for every agent in a run with handoffs/delegates.
    """

    async def on_start(self, context: RunContextWrapper[TContext], agent: Any) -> None:
        """Called right before this agent starts (or resumes, after a handoff) running."""

    async def on_end(self, context: RunContextWrapper[TContext], agent: Any, output: Any) -> None:
        """Called once this agent has produced its final output for the run."""

    async def on_handoff(
        self, context: RunContextWrapper[TContext], agent: Any, source: Any
    ) -> None:
        """Called when `source` hands the run off to this agent."""

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
        """Called right before this agent calls the model."""

    async def on_llm_end(
        self, context: RunContextWrapper[TContext], agent: Any, response: Any
    ) -> None:
        """Called once the model call has returned `response`."""


class _Dispatch[TContext]:
    """Fires one lifecycle event at both scopes: the run's hooks, then the agent's own.

    `runa.run_internal` holds one of these for a whole run and calls it exactly where it used
    to call `RunHooks`, so the eight points that fire an event stay ignorant of there being two
    scopes. Pairing them is knowledge rather than a loop over method names: the two classes name
    the same event differently (`on_agent_start` against `on_start`), and `on_handoff` points
    opposite ways -- the run is told `(from_agent, to_agent)`, while the agent notified is the
    target, told who handed off to it.

    Run-scoped hooks fire first, so a run-wide audit log records an event before any one agent's
    callback can raise out of it.
    """

    def __init__(self, run_hooks: RunHooks[TContext]) -> None:
        self.run_hooks = run_hooks

    @staticmethod
    def _own(agent: Any) -> AgentHooks[Any] | None:
        """The `AgentHooks` assigned to `agent`; `None` covers both unset and a duck-typed agent."""
        return getattr(agent, "hooks", None)

    async def on_agent_start(self, context: RunContextWrapper[TContext], agent: Any) -> None:
        await self.run_hooks.on_agent_start(context, agent)
        if (own := self._own(agent)) is not None:
            await own.on_start(context, agent)

    async def on_agent_end(
        self, context: RunContextWrapper[TContext], agent: Any, output: Any
    ) -> None:
        await self.run_hooks.on_agent_end(context, agent, output)
        if (own := self._own(agent)) is not None:
            await own.on_end(context, agent, output)

    async def on_handoff(
        self, context: RunContextWrapper[TContext], from_agent: Any, to_agent: Any
    ) -> None:
        await self.run_hooks.on_handoff(context, from_agent, to_agent)
        if (own := self._own(to_agent)) is not None:
            await own.on_handoff(context, to_agent, from_agent)

    async def on_tool_start(
        self, context: RunContextWrapper[TContext], agent: Any, tool: FunctionTool
    ) -> None:
        await self.run_hooks.on_tool_start(context, agent, tool)
        if (own := self._own(agent)) is not None:
            await own.on_tool_start(context, agent, tool)

    async def on_tool_end(
        self, context: RunContextWrapper[TContext], agent: Any, tool: FunctionTool, result: object
    ) -> None:
        await self.run_hooks.on_tool_end(context, agent, tool, result)
        if (own := self._own(agent)) is not None:
            await own.on_tool_end(context, agent, tool, result)

    async def on_llm_start(
        self,
        context: RunContextWrapper[TContext],
        agent: Any,
        system_prompt: str | None,
        input_items: list[ConversationItem],
    ) -> None:
        await self.run_hooks.on_llm_start(context, agent, system_prompt, input_items)
        if (own := self._own(agent)) is not None:
            await own.on_llm_start(context, agent, system_prompt, input_items)

    async def on_llm_end(
        self, context: RunContextWrapper[TContext], agent: Any, response: Any
    ) -> None:
        await self.run_hooks.on_llm_end(context, agent, response)
        if (own := self._own(agent)) is not None:
            await own.on_llm_end(context, agent, response)


class _LoggedEvents:
    """The callbacks `RunHooks` and `AgentHooks` name identically, logged once for both scopes.

    A tool call and a model call belong to no scope in particular, so the two `Logging*` classes
    below mix this in and define only what the scopes genuinely spell differently: an agent's
    start and end (`on_agent_start`/`on_start`), and which way a handoff's arguments point.
    """

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


class LoggingRunHooks(_LoggedEvents, RunHooks[Any]):
    """Logs each lifecycle event of a run through the standard `logging` module.

    `Agent.run`/`run_sync` use an instance of this as the default `hooks`, so every run is
    logged without the caller having to ask; passing an explicit `hooks` overrides it. A tool
    call and a model call are logged by `_LoggedEvents`, shared with `LoggingAgentHooks`.
    """

    async def on_agent_start(self, context: RunContextWrapper[Any], agent: Any) -> None:
        """Log that `agent` is about to run."""
        _log_agent_start(agent)

    async def on_agent_end(self, context: RunContextWrapper[Any], agent: Any, output: Any) -> None:
        """Log that `agent` finished; its output only at DEBUG, under the tracing policy."""
        _log_agent_end(agent, output)

    async def on_handoff(
        self, context: RunContextWrapper[Any], from_agent: Any, to_agent: Any
    ) -> None:
        """Log a handoff between agents."""
        _log_handoff(from_agent, to_agent)


class LoggingAgentHooks(_LoggedEvents, AgentHooks[Any]):
    """Logs the lifecycle events of a single agent through the standard `logging` module.

    Assign an instance to an `Agent` subclass's `hooks` class attribute to log that agent's
    own callbacks; unlike `LoggingRunHooks`, this is scoped to one agent rather than a run.
    """

    async def on_start(self, context: RunContextWrapper[Any], agent: Any) -> None:
        """Log that `agent` is about to run."""
        _log_agent_start(agent)

    async def on_end(self, context: RunContextWrapper[Any], agent: Any, output: Any) -> None:
        """Log that `agent` finished; its output only at DEBUG, under the tracing policy."""
        _log_agent_end(agent, output)

    async def on_handoff(self, context: RunContextWrapper[Any], agent: Any, source: Any) -> None:
        """Log that `source` handed off to `agent`."""
        _log_handoff(source, agent)
