"""agent_like.py: `AgentLike`, what Runa reads off an agent, written as types.

Most layers that read an agent (`runa.handoff`, `runa.run_state`, `runa.lifecycle`,
`runa.run_internal`) can't import `runa.agent`, which is built out of them. They, and an
application's `Hooks` subclass, name this `Protocol` instead of taking `Any`, so a renamed or
misspelled attribute is a type error rather than a `getattr` default that reads as "not set".

**The read side only.** Every member is a property, so a subclass's plain class attribute
satisfies it. Which names are settings at all is `Agent.__init__`'s signature to say; these are
the resolved attributes a run reads, not a second way to configure an agent.

Each member has the name of the setting it comes from. `guardrails` and `compact` hold the
setting as declared or passed; `memory`/`knowledge` change kind (`"auto"` becomes the object it
built, `"llm"` a search tool and `None`).

`Agent` is the one implementation. What a run resolves from these (MCP tools, keyed handoffs and
delegates, guardrails by phase, the compactor, the model) is
`runa.run_internal.agent_shape.AgentShape`'s.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from runa._models import Model
from runa._types import ModelSettings
from runa.compact import Compactor
from runa.guardrail import GuardrailsList
from runa.lifecycle import Hooks
from runa.tool import FunctionTool

if TYPE_CHECKING:
    from runa.handoff import Delegate, Handoff  # which read agents through this protocol
    from runa.knowledge import KnowledgeLike
    from runa.memory import MemoryLike


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
        """The functions it offers the model: its `@tool`s, and memory's or knowledge's search."""
        ...

    @property
    def delegates(self) -> list[Delegate]:
        """The subagents it calls like a tool, each run as a nested run (see `runa.handoff`)."""
        ...

    @property
    def mcp_servers(self) -> list[Any]:
        """The MCP servers whose tools join `tools` once a run connects to them."""
        ...

    @property
    def handoffs(self) -> list[Handoff]:
        """The subagents it hands the run over to, each a switch tool (see `runa.handoff`)."""
        ...

    @property
    def guardrails(self) -> GuardrailsList:
        """Its guardrails, in the order they run."""
        ...

    @property
    def output_type(self) -> type | None:
        """The type its final answer is parsed into, or `None` for text."""
        ...

    @property
    def memory(self) -> MemoryLike | None:
        """Its long-term memory, or `None`: the resolved form of `memory`."""
        ...

    @property
    def knowledge(self) -> KnowledgeLike | None:
        """Its document knowledge, or `None`: the resolved form of `knowledge`."""
        ...

    @property
    def compact(self) -> bool | Compactor | None:
        """What trims a long conversation: `True` for `default_compactor`, or its own."""
        ...

    @property
    def max_turns(self) -> int:
        """How many model calls one run of it may make."""
        ...

    @property
    def max_tokens(self) -> int | None:
        """How many tokens one run of it may spend in total, or `None` for no ceiling."""
        ...

    @property
    def timeout(self) -> float | None:
        """How many wall-clock seconds one run of it may take, or `None` for no ceiling."""
        ...

    @property
    def hooks(self) -> Hooks[Any] | None:
        """The lifecycle callbacks that belong to this agent rather than to a run."""
        ...


__all__ = ["AgentLike"]
