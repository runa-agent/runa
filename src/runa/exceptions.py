"""exceptions.py: Runa's own run-time exception hierarchy.

`RunaError` is the base every run-ending failure raises: a tripped guardrail, `MaxTurnsExceeded`, a
model behaving unexpectedly, or a `UserError` in how the framework itself was used. `Agent.run`/
`run_sync` catch `RunaError` (not each subclass individually) and translate it into
`Run(status="error", ...)`, see `runa.agent`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from runa._types import RunContextWrapper

if TYPE_CHECKING:
    from runa.guardrail import ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolOutputGuardrail


@dataclass
class RunErrorDetails:
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
    input_guardrail_results: list[Any]
    output_guardrail_results: list[Any]
    tool_input_guardrail_results: list[Any] = field(default_factory=list)
    tool_output_guardrail_results: list[Any] = field(default_factory=list)
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


class InputGuardrailTripwireTriggered(RunaError):
    """Raised when an `Agent.guardrails` input guardrail's tripwire trips."""

    def __init__(self, guardrail_result: Any) -> None:
        """Store the triggering `guardrail_result` and build a message from its guardrail's name."""
        self.guardrail_result = guardrail_result
        super().__init__(f"Guardrail {guardrail_result.guardrail.name} triggered tripwire")


class OutputGuardrailTripwireTriggered(RunaError):
    """Raised when an `Agent.guardrails` output guardrail's tripwire trips."""

    def __init__(self, guardrail_result: Any) -> None:
        """Store the triggering `guardrail_result` and build a message from its guardrail's name."""
        self.guardrail_result = guardrail_result
        super().__init__(f"Guardrail {guardrail_result.guardrail.name} triggered tripwire")


class ToolInputGuardrailTripwireTriggered(RunaError):
    """Raised when a `@tool(guardrails=...)` input guardrail's tripwire trips."""

    def __init__(
        self, guardrail: ToolInputGuardrail[Any], output: ToolGuardrailFunctionOutput
    ) -> None:
        """Store the triggering `guardrail`/`output` and build a message from its name."""
        self.guardrail = guardrail
        self.output = output
        super().__init__(f"Tool input guardrail {guardrail.name} triggered tripwire")


class ToolOutputGuardrailTripwireTriggered(RunaError):
    """Raised when a `@tool(guardrails=...)` output guardrail's tripwire trips."""

    def __init__(
        self, guardrail: ToolOutputGuardrail[Any], output: ToolGuardrailFunctionOutput
    ) -> None:
        """Store the triggering `guardrail`/`output` and build a message from its name."""
        self.guardrail = guardrail
        self.output = output
        super().__init__(f"Tool output guardrail {guardrail.name} triggered tripwire")


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
    "InputGuardrailTripwireTriggered",
    "MaxTokensExceeded",
    "MaxTurnsExceeded",
    "ModelBehaviorError",
    "OutputGuardrailTripwireTriggered",
    "RunErrorDetails",
    "RunTimeout",
    "RunaError",
    "ToolInputGuardrailTripwireTriggered",
    "ToolOutputGuardrailTripwireTriggered",
    "UserError",
]
