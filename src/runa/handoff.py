"""handoff.py: `Handoff` (agent-as-a-switch) and `agent_as_tool` (agent-as-a-tool).

Both wrap a sub-`Agent` as something the model can call, differing in what happens next: calling a
`Handoff`'s tool switches `run_internal`'s current agent for the rest of the run (`Agent.subagents`'
`.handoff` mode); calling an `agent_as_tool()` tool runs the sub-agent to completion and hands its
output back to the *calling* agent, which keeps going (`.delegate` mode). See `runa.agent.Subagent`
for how a class wires either mode up from its `subagents` list.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from runa._types import RunContextWrapper, Usage
from runa.tool import FunctionTool

_DELEGATE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {"input": {"type": "string"}},
    "required": ["input"],
}


def _slugify(name: str) -> str:
    """Turn an agent's `name` into a valid, lowercase, underscore-separated tool name."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


@dataclass
class Handoff:
    """A sub-agent registered as a switch: calling its tool hands the run over to it.

    `run_internal` sees a `Handoff` in an agent's `handoffs` list and recognizes a call to
    `tool_name` as a request to switch `current_agent` to `agent`, rather than a normal tool call.
    """

    agent: Any
    tool_name: str
    tool_description: str

    @classmethod
    def from_agent(cls, agent: Any) -> Handoff:
        """Build a `Handoff` for `agent` with a name/description derived from it."""
        return cls(
            agent=agent,
            tool_name=f"transfer_to_{_slugify(agent.name)}",
            tool_description=f"Transfer the conversation to {agent.name}.",
        )


def agent_as_tool(agent: Any, tool_name: str | None, tool_description: str | None) -> FunctionTool:
    """Wrap `agent` as a `FunctionTool` that runs it on a generated `input` string.

    The nested run shares the caller's `context` (so tools/guardrails/`instructions` see the same
    object) but not its conversation history: the calling agent generates fresh input for it, the
    same way any other tool call's arguments are generated. A nested run that errors surfaces its
    error message as the tool's return value instead of raising, so the calling agent's turn can
    still continue and decide how to respond.

    The delegate's context is `ctx.fork()`ed, not just `ctx.context` unwrapped: this shares the
    sticky approval ledger, the call-id replay guard, and the guardrail-result audit trail with
    the delegate (and back), and merges the delegate's usage into the caller's afterward.

    A delegate run that pauses for approval raises `DelegatePaused`: the caller's run pauses on
    the same interruptions, and the nested `RunState` waits in `ctx.paused_delegates` under this
    call's id, so resuming the caller resumes the delegate instead of starting it over.
    """
    resolved_name = tool_name or _slugify(agent.name)
    resolved_description = tool_description or f"Delegate a task to {agent.name}."

    async def on_invoke_tool(ctx: RunContextWrapper, arguments_json: str, call_id: str) -> Any:
        paused = ctx.paused_delegates.pop(call_id, None)
        if paused is not None:
            forked = paused.context_wrapper
            run = await agent.run(paused)
        else:
            args = json.loads(arguments_json) if arguments_json else {}
            forked = ctx.fork()
            run = await agent.run(args.get("input", ""), _context_wrapper=forked)
        ctx.usage.add(forked.usage)
        if run.status == "paused":
            forked.usage = Usage()  # already merged into the caller's
            state = run.to_state()
            ctx.paused_delegates[call_id] = state
            raise DelegatePaused(state.pending)
        return run.output if run.status == "completed" else f"error: {run.error}"

    return FunctionTool(
        name=resolved_name,
        description=resolved_description,
        params_json_schema=_DELEGATE_INPUT_SCHEMA,
        on_invoke_tool=on_invoke_tool,
        delegate=agent,
    )


class DelegatePaused(Exception):  # noqa: N818 -- a signal, not an error
    """Raised by a delegate tool whose nested run paused: carries its `interruptions` up."""

    def __init__(self, interruptions: list[Any]) -> None:
        """Hold the nested run's pending `Interruption`s for the caller's run to surface."""
        super().__init__(f"delegate paused on {len(interruptions)} approval(s)")
        self.interruptions = interruptions


__all__ = ["DelegatePaused", "Handoff", "agent_as_tool"]
