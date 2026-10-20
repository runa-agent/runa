"""handoff.py: `Handoff` (agent-as-a-switch) and `Delegate` (agent-as-a-tool).

Both wrap a sub-`Agent` as something the model can call, differing in what happens next: calling a
`Handoff`'s tool switches `run_internal`'s current agent for the rest of the run (`Agent.subagents`'
`.handoff` mode); calling a `Delegate`'s tool runs the sub-agent to completion, as a nested run of
the caller's, and hands its output back to the *calling* agent, which keeps going (`.delegate`
mode). A `Subagent` is one entry of an agent's `subagents` list: a class and the mode it is wired
up in.

The two modes are the only edges between agents, so this module names them (`Mode`); walking them
is `runa.run_internal.agent_shape`'s, which builds the edges.

`Agent` is a type-only import, so the runtime dependency still points from `runa.agent` to this
module.
"""

import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

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

    def __call__(
        self, *, tool_name: str | None = None, tool_description: str | None = None
    ) -> Subagent:
        """Return a copy with the tool name/description overridden."""
        return replace(self, tool_name=tool_name, tool_description=tool_description)


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


@dataclass(frozen=True)
class Delegate:
    """A sub-agent registered as a tool: calling it runs `agent` on the `input` the model wrote.

    `run_internal` sees a `Delegate` in an agent's `delegates` list and runs a call to `tool_name`
    as a nested run of the caller's, whose output goes back to the caller as the call's result.
    """

    agent: Agent
    tool_name: str
    tool_description: str
    params_json_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        }
    )
    """What the model calls it with: the one `input` string the delegate is handed as its turn."""

    @classmethod
    def from_agent(
        cls, agent: Agent, tool_name: str | None = None, tool_description: str | None = None
    ) -> Delegate:
        """Build a `Delegate` for `agent`, its name/description derived from it unless given."""
        return cls(
            agent=agent,
            tool_name=tool_name or _slugify(agent.name),
            tool_description=tool_description or f"Delegate a task to {agent.name}.",
        )


__all__ = ["Delegate", "Handoff", "Mode", "Subagent"]
