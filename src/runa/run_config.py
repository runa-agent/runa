"""run_config.py: `RunConfig`, per-call configuration for `Runner.run`/`run_sync`/`run_streamed`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from runa._models import ModelProvider

DEFAULT_MAX_TURNS = 10


@dataclass
class RunConfig:
    """Per-call configuration for `Runner.run`/`run_sync`/`run_streamed`.

    `workflow_name` names the `Trace` this run produces; `group_id`/`trace_metadata` are recorded
    on it verbatim. `model_provider` resolves an `Agent.model` string to a `Model`, irrelevant
    when `Agent.model` is already a `Model` instance (as Runa's own tests do, to script one).
    """

    model_provider: ModelProvider = field(default_factory=ModelProvider)
    workflow_name: str = "Agent"
    group_id: str | None = None
    trace_metadata: dict[str, Any] | None = None
    max_turns: int = DEFAULT_MAX_TURNS


__all__ = ["DEFAULT_MAX_TURNS", "RunConfig"]
