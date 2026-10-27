"""dispatch.py: `_Dispatch`, the `Hooks` a run calls, firing each event at both scopes."""

from typing import TYPE_CHECKING, Any

from runa._items import ConversationItem
from runa._types import RunContext
from runa.lifecycle import Hooks
from runa.run_internal.agent_shape import shape_of
from runa.tool import Tool

if TYPE_CHECKING:
    from runa.agent import Agent


class _Dispatch[TContext](Hooks[TContext]):
    """Fires each lifecycle event at both scopes: the run's hooks, then the agent's own.

    The turn loop holds one of these for a whole run and calls it exactly where it would call
    `Hooks`. Every event is overridden here, so an event added to `Hooks` needs its own override
    below or it fires at neither scope.

    Each event fires the `hooks` of the agent acting, except a handoff, which notifies the target.
    Run-scoped hooks fire first, so a run-wide audit log records an event before any one agent's
    callback can raise out of it.
    """

    def __init__(self, run_hooks: Hooks[TContext]) -> None:
        self.run_hooks = run_hooks

    def _scopes(self, agent: Agent) -> list[Hooks[TContext]]:
        """The hooks an event about `agent` fires, in order: the run's, then `agent`'s own."""
        own = shape_of(agent).hooks
        return [self.run_hooks] if own is None else [self.run_hooks, own]

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

    async def on_tool_start(self, context: RunContext[TContext], agent: Agent, tool: Tool) -> None:
        for hooks in self._scopes(agent):
            await hooks.on_tool_start(context, agent, tool)

    async def on_tool_end(
        self, context: RunContext[TContext], agent: Agent, tool: Tool, result: object
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


__all__ = ["_Dispatch"]
