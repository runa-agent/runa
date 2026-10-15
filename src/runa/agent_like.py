"""agent_like.py: `AgentLike`, what Runa reads off an agent, written as types.

"An agent" is the framework's most central seam, and almost nothing that reads one can import the
class that implements it: `Agent` is assembled *out of* `runa.handoff`, `runa.run_state`,
`runa.lifecycle` and `runa.run_internal`, so each of those importing `runa.agent` back would be a
cycle. The dependency has to point this way. What it must not cost is the definition: an agent is
one shape, and this module is where it is written down, so every layer that reads one -- and a
`Hooks` subclass in an application, handed an agent by every callback -- names the same type
instead of taking `Any` and asking an object for an attribute it may not have.

That default is the failure this exists to remove. `getattr(agent, "handoffs", None) or []` reads
a renamed or misspelled attribute as "this agent has no subagents" and lets the run proceed --
`Agent._reject_unknown_settings`'s bug one layer down, where there is no class body left to
check. Written as a type, the same mistake is a type error at `Agent.run`, where the class hands
itself to the loop.

**The read side only**, and read-only in fact: every member below is a property, because a
subclass is where the values come from and a mutable attribute would have to match this
declaration exactly rather than merely satisfy it -- `instructions = "Be helpful."` is a `str`,
and only a type nobody can assign through accepts it as one of the three things `instructions`
may be. Which names are settings at all is `Agent.__init__`'s to say (`agent._AGENT_FIELDS`);
these are the resolved attributes a run then reads, so this is no second way to configure an
agent.

Three names carry the irregularity of the class they describe. `guardrails` and `compact` stay
the declaration's, so their resolved forms answer to `bound_guardrails` and `compactor`; and
`memory`/`knowledge` are the one pair whose declaration and resolution are different *kinds* of
value under one name (`"auto"`, or the object it built), so a type checker reading a subclass and
one reading a run would each see a different half. They are typed `Any` and `Agent.__init__` is
the guarantee instead: by the time a run reads either, it is an object or `None`.

`Agent` is the one implementation. `run_internal.agent_shape.AgentShape` is deliberately not a
second one: it is an agent's shape *resolved*, folding `mcp_servers` into `tools`, which is why
the two places that take either say so (`AgentShape.of`, `run_loop._run_async`). What no
annotation can state is the dynamic half, that `AgentShape.of` copies each member below into the
field named for it, and that stays a test (`tests/test_agent.py`).
"""

from collections.abc import Callable
from typing import Any, Protocol

from runa._models import Model
from runa._types import ModelSettings
from runa.compact import Compactor
from runa.guardrail import BoundGuardrail, Phase
from runa.lifecycle import Hooks
from runa.tool import FunctionTool


class AgentLike(Protocol):
    """Every attribute Runa reads off an agent: the other side of the seam, as types."""

    @property
    def name(self) -> str:
        """What this agent is called, everywhere one agent has to be told from another."""
        ...

    @property
    def instructions(self) -> str | Callable[..., Any] | None:
        """Its system prompt: a string, a callable the run resolves, or none at all."""
        ...

    @property
    def model(self) -> str | Model:
        """The model name a run resolves through its provider, or a `Model` built directly."""
        ...

    @property
    def model_settings(self) -> ModelSettings:
        """How that model is sampled."""
        ...

    @property
    def tools(self) -> list[FunctionTool]:
        """The tools it offers the model, its delegates among them (see `runa.handoff`)."""
        ...

    @property
    def mcp_servers(self) -> list[Any]:
        """The MCP servers whose tools join `tools` once a run connects to them."""
        ...

    @property
    def handoffs(self) -> list[Any]:
        """Its handoff targets, as `Handoff.by_tool_name` reads them (`runa.handoff`)."""
        ...

    @property
    def bound_guardrails(self) -> dict[Phase, list[BoundGuardrail]]:
        """Its guardrails by the phase each one runs in: the resolved form of `guardrails`."""
        ...

    @property
    def output_type(self) -> type | None:
        """The type its final answer is parsed into, or `None` for text."""
        ...

    @property
    def memory(self) -> Any:
        """Its long-term memory: a `MemoryLike` or `None` once resolved (see this module)."""
        ...

    @property
    def knowledge(self) -> Any:
        """Its document knowledge: a `KnowledgeLike` or `None` once resolved."""
        ...

    @property
    def compactor(self) -> Compactor | None:
        """What trims a long conversation, if anything: the resolved form of `compact`."""
        ...

    @property
    def hooks(self) -> Hooks[Any] | None:
        """The lifecycle callbacks that belong to this agent rather than to a run."""
        ...


__all__ = ["AgentLike"]
