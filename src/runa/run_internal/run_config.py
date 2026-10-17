"""run_config.py: `RunConfig`, per-call configuration for one `Agent` run."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from runa._models import ModelProvider
from runa.session import Session
from runa.tracing.manual import current_trace

if TYPE_CHECKING:
    from runa.agent import Agent

DEFAULT_MAX_TURNS = 10

_MODEL_PROVIDER = ModelProvider()


@dataclass
class RunConfig:
    """Per-call configuration for `Agent.run`/`run_sync`/`run_streamed`.

    `workflow_name` names the `Trace` this run produces; `group_id`/`trace_metadata` are recorded
    on it verbatim. `model_provider` resolves an `Agent.model` string to a `Model`, irrelevant
    when `Agent.model` is already a `Model` instance (as Runa's own tests do, to script one).

    `max_turns`/`max_tokens`/`timeout` are the three ceilings on one run: model calls, tokens
    spent, and wall-clock seconds. Each is `Agent.max_turns`/`.max_tokens`/`.timeout` by the
    time it gets here; `None` means no ceiling of that kind.
    """

    model_provider: ModelProvider = field(default_factory=ModelProvider)
    workflow_name: str = "Agent"
    group_id: str | None = None
    trace_metadata: dict[str, Any] | None = None
    max_turns: int = DEFAULT_MAX_TURNS
    max_tokens: int | None = None
    timeout: float | None = None

    @classmethod
    def of(cls, agent: Agent, session: Session | None = None) -> RunConfig:
        """`agent`'s run, grouped under an enclosing `tracing.trace` block, else its `session`.

        The one place an `Agent` becomes a `RunConfig`, for `Agent.run`. A `.delegate`
        subagent's nested run is its caller's config with the delegate's own three ceilings.
        """
        outer = current_trace()
        return cls(
            model_provider=_MODEL_PROVIDER,
            workflow_name=type(agent).__name__,
            group_id=outer.id if outer else (session.session_id if session is not None else None),
            max_turns=agent.max_turns,
            max_tokens=agent.max_tokens,
            timeout=agent.timeout,
        )


__all__ = ["DEFAULT_MAX_TURNS", "RunConfig"]
