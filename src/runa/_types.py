"""_types.py: the request/response shapes Runa's own runtime is built on, apart from the item.

No OpenAI (or Anthropic) SDK type leaks past `_models`, everywhere else in Runa speaks these
types instead: a token-usage tally, per-call model settings, the context a run carries. The one
shape they're all in service of, the `ConversationItem`, lives in `runa._items` with the
accessors that read one -- it is chat-completions-shaped by choice, not a neutral format, and
that module is where the choice is argued. `run_internal` builds and consumes everything here;
each `Model` implementation translates it to and from its own provider's wire format.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from runa._items import ConversationItem

MessageContent = str | Sequence[str | Path | dict[str, Any]]
"""One user message's `content`: plain text, or a list for a multimodal message. Each list item
is a bare string (text, or an image when it is an image URL or `data:` URI -- `runa.content.parts`
refuses to read a string as a local path, since the list may carry user input), a `Path` (a local
image, read off disk), or an already-built content part dict (`runa.content.text`/`.image`, an
escape hatch for a string the heuristic can't classify). `runa._models.chat_completions` passes
the resulting parts straight through; `runa._models.anthropic` translates them into Claude's own
content blocks.

`Sequence`, not `list`, so a `list[dict[str, Any]]` of already-built parts type-checks too --
`list` is invariant, `Sequence` is covariant.
"""

ToolChoice = Literal["auto", "required", "none"] | str | None
"""`"auto"`/`"required"`/`"none"`, a specific tool name to force, or `None` for the provider's
default.
"""

ReasoningEffort = Literal["low", "medium", "high"] | None
"""How hard a reasoning model should think before answering, or `None` for the provider's default.

Three levels, not each provider's own ladder: `low`/`medium`/`high` is the vocabulary both
backends share, so an agent keeps meaning the same thing after `model` changes vendor. A provider
that can go further (Claude's `xhigh`/`max`) is reached by passing a `Model`, the same escape
hatch an unnamed endpoint takes.
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
class ModelSettings:
    """Per-call model parameters; a `Model` implementation uses whichever of these its API takes.

    "Whichever" is a closed set, not a courtesy: every field here reaches both backends' request
    body, unless it is one a backend deliberately doesn't take.
    A setting that configures nothing is the failure `check_model_name` exists to prevent, one
    field over -- the bill and the caller's mental model disagree, with no signal anywhere.
    """

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    tool_choice: ToolChoice = None
    parallel_tool_calls: bool | None = None
    reasoning_effort: ReasoningEffort = None
    max_retries: int | None = None
    """How many times to retry a failed request (connection errors, 408/409/429/5xx), on top of
    the first attempt. `None` uses the backend default of 2; `0` disables retrying outright, for
    an app that does its own. Honored identically by both backends."""


@dataclass
class ModelResponse:
    """What a `Model.get_response()` call returns: the model's output items, plus usage."""

    output: list[ConversationItem]
    usage: Usage
    response_id: str | None = None


@dataclass
class RunContext[TContext]:
    """What user code is told about the run calling it: its `context`, its user, and usage so far.

    `context` is whatever was passed to `Agent.run()`/`run_sync()`, never sent to the model; it's
    how a tool's `ctx`, an `@approval` predicate's `ctx`, a `Hooks` method, and a single-argument
    `instructions` callable reach it. `user_id` is who the run is for, the `user_id` passed to
    `Agent.run()` and shared by its delegates; it's the one scope every `Memory`
    read and write the run makes uses, `"auto"` and `"llm"` alike. The run sets it; user code
    reads it. `usage` is what this run has spent so far.

    Nothing else: how the run governs itself (its approval ledger, its guardrail audit trail) is
    held beside this, out of an app's reach.
    """

    context: TContext = None  # pyright: ignore[reportAssignmentType]
    user_id: str | None = None
    usage: Usage = field(default_factory=Usage)


__all__ = [
    "InputTokensDetails",
    "MessageContent",
    "ModelResponse",
    "ModelSettings",
    "OutputTokensDetails",
    "ReasoningEffort",
    "RunContext",
    "ToolChoice",
    "Usage",
]
