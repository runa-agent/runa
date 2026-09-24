"""`runa.lifecycle`: `RunHooks`/`AgentHooks`, and live `logging`-module output for a run.

For a structured, persisted, queryable record of a run instead, see `runa.tracing`. That runs
unconditionally regardless of `hooks`; this module is just console/log lines while a run happens.

These hooks are the default, so they log to whatever handler the application configured. At INFO
they name what happened and never carry content: an agent's answer and a tool's result are user
data, and a production app running at INFO should not be writing them to stdout. Content is logged
at DEBUG only, and even there it goes through `runa.tracing`'s redact/truncate policy, so
`observe(redact=[...])` and `observe(capture_outputs=False)` govern these lines too.
"""

from __future__ import annotations

import logging
from typing import Any

from runa._types import RunContextWrapper, TResponseInputItem
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


def _log_output(label: str, name: str, value: Any) -> None:
    """Log one completed step: its name at INFO, its output at DEBUG if the policy allows."""
    from runa.tracing.config import capture_outputs, output_limit

    logger.info("%s end: %s", label, name)
    if capture_outputs() and logger.isEnabledFor(logging.DEBUG):
        logger.debug("%s output: %s -> %s", label, name, _content(value, limit=output_limit()))


def _log_tool_output(agent_name: str, tool_name: str, result: object) -> None:
    """Log a finished tool call: names at INFO, the result at DEBUG if the policy allows."""
    from runa.tracing.config import capture_outputs, tool_result_limit

    logger.info("tool end: %s (%s)", tool_name, agent_name)
    if capture_outputs() and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "tool output: %s -> %s", tool_name, _content(result, limit=tool_result_limit())
        )


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
        input_items: list[TResponseInputItem],
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
        input_items: list[TResponseInputItem],
    ) -> None:
        """Called right before this agent calls the model."""

    async def on_llm_end(
        self, context: RunContextWrapper[TContext], agent: Any, response: Any
    ) -> None:
        """Called once the model call has returned `response`."""


class LoggingRunHooks(RunHooks[Any]):
    """Logs each lifecycle event of a run through the standard `logging` module.

    `Agent.run`/`run_sync` use an instance of this as the default `hooks`, so every run is
    logged without the caller having to ask; passing an explicit `hooks` overrides it.
    """

    async def on_agent_start(self, context: RunContextWrapper[Any], agent: Any) -> None:
        """Log that `agent` is about to run."""
        logger.info("agent start: %s", agent.name)

    async def on_agent_end(self, context: RunContextWrapper[Any], agent: Any, output: Any) -> None:
        """Log that `agent` finished; its output only at DEBUG, under the tracing policy."""
        _log_output("agent", agent.name, output)

    async def on_handoff(
        self, context: RunContextWrapper[Any], from_agent: Any, to_agent: Any
    ) -> None:
        """Log a handoff between agents."""
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
        _log_tool_output(agent.name, tool.name, result)

    async def on_llm_start(
        self,
        context: RunContextWrapper[Any],
        agent: Any,
        system_prompt: str | None,
        input_items: list[TResponseInputItem],
    ) -> None:
        """Log that `agent` is about to call the model."""
        logger.debug("llm start: %s", agent.name)

    async def on_llm_end(self, context: RunContextWrapper[Any], agent: Any, response: Any) -> None:
        """Log that `agent`'s model call returned."""
        logger.debug("llm end: %s", agent.name)


class LoggingAgentHooks(AgentHooks[Any]):
    """Logs the lifecycle events of a single agent through the standard `logging` module.

    Assign an instance to an `Agent` subclass's `hooks` class attribute to log that agent's
    own callbacks; unlike `LoggingRunHooks`, this is scoped to one agent rather than a run.
    """

    async def on_start(self, context: RunContextWrapper[Any], agent: Any) -> None:
        """Log that `agent` is about to run."""
        logger.info("agent start: %s", agent.name)

    async def on_end(self, context: RunContextWrapper[Any], agent: Any, output: Any) -> None:
        """Log that `agent` finished; its output only at DEBUG, under the tracing policy."""
        _log_output("agent", agent.name, output)

    async def on_handoff(self, context: RunContextWrapper[Any], agent: Any, source: Any) -> None:
        """Log that `source` handed off to `agent`."""
        logger.info("handoff: %s -> %s", source.name, agent.name)

    async def on_tool_start(
        self, context: RunContextWrapper[Any], agent: Any, tool: FunctionTool
    ) -> None:
        """Log that `tool` is about to run."""
        logger.info("tool start: %s (%s)", tool.name, agent.name)

    async def on_tool_end(
        self, context: RunContextWrapper[Any], agent: Any, tool: FunctionTool, result: object
    ) -> None:
        """Log that `tool` finished; its result only at DEBUG, under the tracing policy."""
        _log_tool_output(agent.name, tool.name, result)

    async def on_llm_start(
        self,
        context: RunContextWrapper[Any],
        agent: Any,
        system_prompt: str | None,
        input_items: list[TResponseInputItem],
    ) -> None:
        """Log that `agent` is about to call the model."""
        logger.debug("llm start: %s", agent.name)

    async def on_llm_end(self, context: RunContextWrapper[Any], agent: Any, response: Any) -> None:
        """Log that `agent`'s model call returned."""
        logger.debug("llm end: %s", agent.name)
