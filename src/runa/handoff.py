"""handoff.py: `Handoff` (agent-as-a-switch) and `Delegate` (agent-as-a-tool).

Both wrap a sub-`Agent` as something the model can call, differing in what happens next: calling a
`Handoff`'s tool switches `run_internal`'s current agent for the rest of the run (`Agent.subagents`'
`.handoff` mode); calling a `Delegate`'s tool runs the sub-agent to completion, as a nested run of
the caller's, and hands its output back to the *calling* agent, which keeps going (`.delegate`
mode). A `Subagent` is one entry of an agent's `subagents` list: a class and the mode it is wired
up in.

A delegate is a tool and a handoff is not. A delegate's call is gated, run and recorded the way a
`@tool`'s is, so it takes the same `needs_approval` and `guardrails`
(`Researcher.delegate(needs_approval=True)`). A handoff's call is the switch itself, with nothing
to approve or guard and no result to record.

The two modes are the only edges between agents, so this module names them (`Mode`); walking them
is `runa.run_internal.agent_shape`'s, which builds the edges.

`Agent` is a type-only import, so the runtime dependency still points from `runa.agent` to this
module.
"""

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from runa._types import RunContext
from runa.approval import predicate
from runa.exceptions import UserError
from runa.guardrail import GuardrailsList, flatten_guardrails
from runa.tool import FunctionTool, _NeedsApproval

if TYPE_CHECKING:
    from runa.agent import Agent

Mode = Literal["handoff", "delegate"]
"""Which of the two mechanisms above wires a subagent up: `Agent.subagents`' `.handoff`/`.delegate`.

Named here, beside both of them, rather than in `runa.agent`: a declaration spells the mode and
`Subagent` carries it, but what each one *means* is this module's.
"""


@dataclass(frozen=True)
class Subagent:
    """A wired-up subagent, attached as a handoff or a delegate tool."""

    agent: type[Agent]
    mode: Mode
    tool_name: str | None = None
    tool_description: str | None = None
    needs_approval: _NeedsApproval = False
    """For a delegate, whether a call to it waits on a human, as `@tool(needs_approval=...)`."""
    guardrails: tuple[Any, ...] = ()
    """For a delegate, the guardrails on a call to it, as `@tool(guardrails=[...])`."""

    def __call__(
        self,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
        needs_approval: _NeedsApproval = False,
        guardrails: GuardrailsList | None = None,
    ) -> Subagent:
        """Return a copy with the tool's name, description and policy overridden.

        A handoff given `needs_approval` or `guardrails` is a `UserError`: it is a switch, not a
        call, so nothing would ever read them.
        """
        if self.mode == "handoff" and (needs_approval is not False or guardrails):
            name = self.agent.__name__
            raise UserError(
                f"{name}.handoff takes no needs_approval or guardrails: a handoff switches agents "
                f"rather than calling one. Put them on {name}.delegate(...), or give {name} "
                f"guardrails of its own."
            )
        return replace(
            self,
            tool_name=tool_name,
            tool_description=tool_description,
            needs_approval=needs_approval,
            guardrails=tuple(guardrails or ()),
        )


def _slugify(name: str) -> str:
    """Turn an agent's `name` into a valid, lowercase, underscore-separated tool name."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


@dataclass
class Handoff:
    """A sub-agent registered as a switch: calling its tool hands the run over to it.

    `run_internal` sees a `Handoff` in an agent's `handoffs` list and recognizes a call to
    `tool_name` as a request to switch `current_agent` to `agent`, rather than a normal tool call.
    """

    agent: Agent
    tool_name: str
    tool_description: str

    @classmethod
    def from_agent(
        cls, agent: Agent, tool_name: str | None = None, tool_description: str | None = None
    ) -> Handoff:
        """Build a `Handoff` for `agent`, its name/description derived from it unless given."""
        return cls(
            agent=agent,
            tool_name=tool_name or f"transfer_to_{_slugify(agent.name)}",
            tool_description=tool_description or f"Transfer the conversation to {agent.name}.",
        )


_DELEGATE_INPUT = {
    "type": "object",
    "properties": {"input": {"type": "string"}},
    "required": ["input"],
}
"""What the model calls a delegate with: the one `input` string it is handed as its turn."""


@dataclass(frozen=True)
class Delegate:
    """A sub-agent registered as a tool: calling it runs `agent` on the `input` the model wrote.

    A call to it is looked up to the `Delegate` itself, not its `tool`, and runs as a nested run
    of the caller's, whose output goes back to the caller as the call's result.
    """

    agent: Agent
    tool: FunctionTool
    """The delegate as the model is offered it and a run gates it: its name, description, `input`
    schema, `needs_approval` and guardrails. It is also what a hook is handed as the call's
    `tool`. Its `on_invoke_tool` refuses to run, because a delegate's call is a nested run of its
    caller's and only that run can start it."""

    @classmethod
    def from_agent(
        cls,
        agent: Agent,
        tool_name: str | None = None,
        tool_description: str | None = None,
        *,
        needs_approval: _NeedsApproval = False,
        guardrails: GuardrailsList = (),
    ) -> Delegate:
        """Build a `Delegate` for `agent`, its name/description derived from it unless given."""
        name = tool_name or _slugify(agent.name)

        async def on_invoke_tool(ctx: RunContext, args: dict[str, Any], call_id: str) -> Any:
            raise UserError(
                f"{name} delegates to {agent.name}, which runs only as a call of its caller's "
                f"run. Run {type(agent).__name__}() itself to call it directly."
            )

        return cls(
            agent=agent,
            tool=FunctionTool(
                name=name,
                description=tool_description or f"Delegate a task to {agent.name}.",
                params_json_schema=_DELEGATE_INPUT,
                on_invoke_tool=on_invoke_tool,
                guardrails=flatten_guardrails(guardrails, tool=True),
                needs_approval=predicate(needs_approval),
            ),
        )


__all__ = ["Delegate", "Handoff", "Mode", "Subagent"]
