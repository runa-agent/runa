"""exceptions.py: Runa's own run-time exception hierarchy.

`RunaError` is the base every run-ending failure raises: a tripped guardrail, `MaxTurnsExceeded`, a
model behaving unexpectedly, or a `UserError` in how the framework itself was used. `Agent.run`/
`run_sync` catch `RunaError` (not each subclass individually) and translate it into
`Run(status="error", ...)`, see `runa.agent`.
"""

from dataclasses import dataclass, field
from typing import Any

from runa._types import RunContextWrapper
from runa.guardrail import GuardrailAudit, GuardrailResult, GuardrailResults


@dataclass
class RunErrorDetails(GuardrailAudit):
    """Whatever a run had accumulated when a `RunaError` cut it short.

    `context_wrapper.usage` is what `Agent.run`/`run_sync` read to still record token usage for a
    run that errored instead of completing; the rest is here for a caller that wants more detail
    than `Run.error`'s message.
    """

    input: str | list[Any]
    new_items: list[Any]
    raw_responses: list[Any]
    last_agent: Any
    context_wrapper: RunContextWrapper
    guardrail_results: GuardrailResults = field(default_factory=GuardrailResults)
    trace: Any = None


class RunaError(Exception):
    """Base class for every exception a Runa agent run can end with."""

    run_data: RunErrorDetails | None

    def __init__(self, *args: object) -> None:
        """Initialize with `run_data` unset; the runner fills it in as the run unwinds."""
        super().__init__(*args)
        self.run_data = None


class MaxTurnsExceeded(RunaError):
    """Raised when a run reaches `max_turns` without producing a final output."""

    def __init__(self, message: str) -> None:
        """Store `message` as both the exception's args and its `.message`."""
        self.message = message
        super().__init__(message)


class MaxTokensExceeded(RunaError):
    """Raised when a run's accumulated token usage passes `Agent.max_tokens`.

    The spend ceiling to `MaxTurnsExceeded`'s call ceiling: `max_turns` bounds how many times a
    run may call the model, this bounds how much those calls may cost. Checked after each model
    call, so the run stops at the first call that crosses the line rather than before it.
    """

    def __init__(self, message: str) -> None:
        """Store `message` as both the exception's args and its `.message`."""
        self.message = message
        super().__init__(message)


class RunTimeout(RunaError):
    """Raised when a run passes `Agent.timeout` seconds without finishing.

    The wall-clock ceiling `max_turns`/`max_tokens` can't give: a single tool call or model
    response that hangs would otherwise stall a run indefinitely. Like every `RunaError` this
    comes back as `Run(status="error")` rather than propagating, and the partial trace is still
    exported, so a timed-out run is as inspectable as a failed one.
    """

    def __init__(self, message: str) -> None:
        """Store `message` as both the exception's args and its `.message`."""
        self.message = message
        super().__init__(message)


class ModelBehaviorError(RunaError):
    """Raised when the model does something a `Model` implementation can't make sense of.

    For example: calling a tool that isn't in the request, or returning malformed tool-call JSON.
    """

    def __init__(self, message: str) -> None:
        """Store `message` as both the exception's args and its `.message`."""
        self.message = message
        super().__init__(message)


class UserError(RunaError):
    """Raised when the caller has misused the framework itself (bad config, missing credentials)."""

    def __init__(self, message: str) -> None:
        """Store `message` as both the exception's args and its `.message`."""
        self.message = message
        super().__init__(message)


class GuardrailTripwireTriggered(RunaError):
    """Raised when a guardrail's tripwire trips, whichever `Phase` it ran in.

    One exception for the 2x2 rather than four classes that differed only in which phase they
    named in their message: `phase` says where it tripped, and `guardrail`/`output` are the entry
    and the verdict `guardrail_result` pairs up.
    """

    def __init__(self, guardrail_result: GuardrailResult) -> None:
        """Store the triggering result, and name its phase and guardrail in the message."""
        self.guardrail_result = guardrail_result
        self.phase = guardrail_result.phase
        self.guardrail = guardrail_result.guardrail
        self.output = guardrail_result.output
        super().__init__(
            f"{guardrail_result.phase.title} "
            f"{guardrail_result.guardrail.get_name()} triggered tripwire"
        )


class DuplicateToolCallError(RunaError):
    """Raised when a tool-call id that already executed once is submitted for execution again.

    Guards against silently re-running a tool -- e.g. resuming the same `RunState` twice, or a
    model retry that reuses a call id.
    """

    def __init__(self, call_id: str, tool_name: str) -> None:
        """Store the offending `call_id`/`tool_name` and build a message from them."""
        self.call_id = call_id
        self.tool_name = tool_name
        super().__init__(f"tool call {call_id!r} for {tool_name!r} was already executed")


__all__ = [
    "DuplicateToolCallError",
    "GuardrailTripwireTriggered",
    "MaxTokensExceeded",
    "MaxTurnsExceeded",
    "ModelBehaviorError",
    "RunErrorDetails",
    "RunTimeout",
    "RunaError",
    "UserError",
]
