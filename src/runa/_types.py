"""_types.py: the provider-neutral request/response shapes Runa's own runtime is built on.

No OpenAI (or Anthropic) SDK type leaks past `_models`, everywhere else in Runa speaks these
types instead: a plain dict for one turn of conversation, a token-usage tally, and per-call model
settings. `run_internal` builds and consumes these; each `Model` implementation translates them to
and from whatever shape its own provider's wire format actually wants.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

TResponseInputItem = dict[str, Any]
"""One turn of conversation history: a `{"role": ..., "content": ...}` message, a tool call, or a
tool result. Plain JSON, never a provider SDK type, an output item becomes tomorrow's input item
once appended to history, so this one shape serves both directions.
"""

TResponseOutputItem = TResponseInputItem
"""What a model call produces, before it's appended to history, the same shape as
`TResponseInputItem`; see that alias for why one shape covers both.
"""

MessageContent = str | Sequence[str | dict[str, Any]]
"""One user message's `content`: plain text, or a list for a multimodal message. Each list item
is either a bare string (auto-detected as text or an image by `runa.content.parts`) or an
already-built content part dict (`runa.content.text`/`.image`, an escape hatch for a string the
heuristic can't classify). `runa._models.openai_chatcompletions` passes the resulting parts
straight through; `runa._models.anthropic` translates them into Claude's own content blocks.

`Sequence`, not `list`, so a `list[dict[str, Any]]` of already-built parts type-checks too --
`list` is invariant, `Sequence` is covariant.
"""

TResponseStreamEvent = dict[str, Any]
"""One raw provider streaming event, passed through to callers as-is via
`RawResponsesStreamEvent.data`. Opaque to Runa itself: each `Model` decides what to put in it.
"""

ToolChoice = Literal["auto", "required", "none"] | str | None
"""`"auto"`/`"required"`/`"none"`, a specific tool name to force, or `None` for the provider's
default.
"""


@dataclass
class InputTokensDetails:
    """A breakdown of `Usage.input_tokens` into cached and cache-write tokens."""

    cached_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class OutputTokensDetails:
    """A breakdown of `Usage.output_tokens` into reasoning tokens."""

    reasoning_tokens: int = 0


@dataclass
class Usage:
    """Token usage for one or more model calls.

    `Agent.run`/`run_sync` accumulate every call's `Usage` into `Agent.usage` via `.add()`, and
    record the latest one on `Agent.last_usage`.
    """

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: InputTokensDetails = field(default_factory=InputTokensDetails)
    output_tokens_details: OutputTokensDetails = field(default_factory=OutputTokensDetails)

    def add(self, other: Usage) -> None:
        """Accumulate `other`'s counts into this `Usage`, in place."""
        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.input_tokens_details.cached_tokens += other.input_tokens_details.cached_tokens
        self.input_tokens_details.cache_write_tokens += (
            other.input_tokens_details.cache_write_tokens
        )
        self.output_tokens_details.reasoning_tokens += other.output_tokens_details.reasoning_tokens


@dataclass
class Reasoning:
    """A reasoning-model's effort/summary settings, for providers that support them."""

    effort: Literal["low", "medium", "high"] | None = None
    summary: Literal["auto", "concise", "detailed"] | None = None


@dataclass
class ModelSettings:
    """Per-call model parameters; a `Model` implementation uses whichever of these its API takes."""

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    tool_choice: ToolChoice = None
    parallel_tool_calls: bool | None = None
    reasoning: Reasoning | None = None


@dataclass
class ModelResponse:
    """What a `Model.get_response()` call returns: the model's output items, plus usage."""

    output: list[TResponseOutputItem]
    usage: Usage
    response_id: str | None = None


@dataclass
class RunContextWrapper[TContext]:
    """Wraps the `context` object passed to `Agent.run()`/`run_sync()`, plus usage-so-far.

    `context` is never sent to the model; it's how tools, guardrails, `needs_approval`, and a
    single-argument `instructions` callable receive whatever the caller passed to `run`/`run_sync`.

    `approval_ledger`/`approval_ledger_messages` hold sticky ("always approve"/"always reject")
    per-tool-name decisions; `executed_call_ids` guards against executing the same tool-call id
    twice (e.g. from resuming a stale `RunState`). `paused_delegates` maps a delegate tool call's
    id to the nested `RunState` it paused on, so resuming the caller resumes the delegate too.
    The four `*_guardrail_results` lists are every `GuardrailResult` produced this run, tripped
    or not -- an audit trail, not just the one that stopped the run.
    """

    context: TContext = None  # pyright: ignore[reportAssignmentType]
    usage: Usage = field(default_factory=Usage)
    approval_ledger: dict[str, bool] = field(default_factory=dict)
    approval_ledger_messages: dict[str, str] = field(default_factory=dict)
    executed_call_ids: set[str] = field(default_factory=set)
    paused_delegates: dict[str, Any] = field(default_factory=dict)
    input_guardrail_results: list[Any] = field(default_factory=list)
    output_guardrail_results: list[Any] = field(default_factory=list)
    tool_input_guardrail_results: list[Any] = field(default_factory=list)
    tool_output_guardrail_results: list[Any] = field(default_factory=list)

    def fork(self) -> RunContextWrapper[TContext]:
        """Build a child context for a nested delegate-agent call (`agent_as_tool`).

        Shares `context` and every governance list/dict by reference -- so an approval, a
        replayed call id, or a guardrail result recorded in either the parent or the delegate is
        visible to both -- but starts `usage` at zero: the caller merges the delegate's usage
        back explicitly (`ctx.usage.add(forked.usage)`), so it isn't double-counted against
        `Agent.run`'s own usage accumulation.
        """
        return RunContextWrapper(
            context=self.context,
            usage=Usage(),
            approval_ledger=self.approval_ledger,
            approval_ledger_messages=self.approval_ledger_messages,
            executed_call_ids=self.executed_call_ids,
            paused_delegates=self.paused_delegates,
            input_guardrail_results=self.input_guardrail_results,
            output_guardrail_results=self.output_guardrail_results,
            tool_input_guardrail_results=self.tool_input_guardrail_results,
            tool_output_guardrail_results=self.tool_output_guardrail_results,
        )


__all__ = [
    "InputTokensDetails",
    "MessageContent",
    "ModelResponse",
    "ModelSettings",
    "OutputTokensDetails",
    "Reasoning",
    "RunContextWrapper",
    "TResponseInputItem",
    "TResponseOutputItem",
    "TResponseStreamEvent",
    "ToolChoice",
    "Usage",
]
