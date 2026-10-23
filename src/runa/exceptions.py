"""exceptions.py: Runa's two exception roots.

`RunaError` is the base every run-ending failure raises: a tripped guardrail, `MaxTurnsExceeded`, a
model behaving unexpectedly, a service not answering, or a `UserError` in how the framework itself
was used. They are raised freely inside a run and caught at its boundary, which reports each as
`Run(status="error", ...)`; anything else a run touches raises (an app's own hook, `Session` or
`Compactor`) is wrapped in `UnexpectedError` there. None of them carry a payload for the caller.

`OperatorError` is the other root: the person running a Runa app typed something wrong, mostly
on a surface around a run (the CLI, the dashboard). It is not a `RunaError`, and it is the one
exception a run lets through.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runa.guardrail import GuardrailResult


class OperatorError(Exception):
    """Base class for a failure that is the operator's input rather than a bug.

    A mistyped session id, a command run outside a Runa project, an unparseable
    `RUNA_DATABASE_URL`. Every surface that fronts someone's app owes these a plain message
    instead of a traceback: `cli/main.py` prints one line and exits 1, `runa ui` renders a 400
    page. Both catch this base, so a new subclass is handled by every surface the moment it is
    declared, rather than when each surface remembers to name it.

    Subclass this, not `Exception`, whenever the module raising already knows the failure is the
    operator's fault -- which is where that knowledge lives. Keep a bare `Exception` when it
    isn't the operator's fault: `runa.project.AppLoadError` is a bug in the developer's own
    `main.py`, and earns a different message for saying so.
    """


class RunaError(Exception):
    """Base class for every exception a Runa agent run can end with.

    One message, carried as both the exception's args and its `.message`, and nothing else: what a
    failed run accumulated is reported on the `"error"` `Run` returned in its place. Subclasses
    built from structured fields (`GuardrailTripwireTriggered`, `DuplicateToolCallError`) compose
    the message in their own `__init__` and hand it here.
    """

    def __init__(self, message: str) -> None:
        """Store `message` as both the exception's args and its `.message`."""
        super().__init__(message)
        self.message = message


class MaxTurnsExceeded(RunaError):
    """Raised when a run reaches `max_turns` without producing a final output."""


class MaxTokensExceeded(RunaError):
    """Raised when a run's accumulated token usage passes `Agent.max_tokens`.

    The spend ceiling to `MaxTurnsExceeded`'s call ceiling: `max_turns` bounds how many times a
    run may call the model, this bounds how much those calls may cost. Checked after each model
    call, so the run stops at the first call that crosses the line rather than before it.
    """


class RunTimeout(RunaError):
    """Raised when a run passes `Agent.timeout` seconds without finishing.

    The wall-clock ceiling `max_turns`/`max_tokens` can't give: a single tool call or model
    response that hangs would otherwise stall a run indefinitely. Like every `RunaError` this
    comes back as `Run(status="error")` rather than propagating, and the partial trace is still
    exported, so a timed-out run is as inspectable as a failed one.
    """


class ModelBehaviorError(RunaError):
    """Raised when the model does something a `Model` implementation can't make sense of.

    For example: calling a tool that isn't in the request, or returning malformed tool-call JSON.
    """


class ServiceUnavailable(RunaError):
    """Raised when a service a run calls doesn't answer: it is down, slow, or shedding load.

    A connection that can't be made, a timeout, or a 408/409/429/5xx still failing after the
    retries. The one failure a context source (`runa.context_source.ContextSource`) is skipped
    for rather than ending the run; a custom source raises it to be skipped the same way.
    """


class ToolError(RunaError):
    """Raised by a tool to fail its call on purpose, with a message that says what went wrong.

    The call fails, not the run, as it does for any exception a tool raises: the model is answered
    `"error: <message>"` and the failure is the call's `ToolCall.error`. What this one changes is
    the log: a warning without a traceback, since the tool already said what went wrong.
    """


class UserError(RunaError):
    """Raised when the caller has misused the framework itself (bad config, missing credentials)."""


class UnexpectedError(RunaError):
    """Raised in place of an exception a run ended with that is none of Runa's own kinds.

    A `Session` that can't read its database, a `Hooks` method with a bug in it, a `Compactor`
    that divides by zero: the extension points a run calls are application code, and the run
    boundary owes them the same contract it gives a tripped guardrail (`Run(status="error")`,
    with the trace around the failure) rather than one that holds only for failures Runa itself
    declared. The message names the original's type as well as its text, since neither alone
    identifies it, and the original is logged with its traceback where it was caught, so nothing
    about it is lost by coming back as a result.

    Not raised for `OperatorError` (a sibling root, deliberately let through) nor for anything
    deriving from `BaseException` alone -- `asyncio.CancelledError` above all, which a caller
    asking a run to stop must get back as itself.
    """

    def __init__(self, exc: BaseException) -> None:
        """Name `exc`'s type and message, since neither alone identifies what went wrong."""
        detail = str(exc)
        super().__init__(f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__)


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

    Guards against silently re-running a tool within one run, such as a model retry that reuses a
    call id. Resuming a pause twice is `RunAlreadyResumed`.
    """

    def __init__(self, call_id: str, tool_name: str) -> None:
        """Store the offending `call_id`/`tool_name` and build a message from them."""
        self.call_id = call_id
        self.tool_name = tool_name
        super().__init__(f"tool call {call_id!r} for {tool_name!r} was already executed")


class RunAlreadyResumed(RunaError):
    """Raised when a paused run is resumed after it already was: its `run_id` is claimed.

    A pause resumes once, from the `RunState` it paused with or from `RunState.load` in any
    process sharing its database, so the calls it approved run at most once.
    """

    def __init__(self, run_id: str) -> None:
        """Store the claimed `run_id` and name it in the message."""
        self.run_id = run_id
        super().__init__(f"paused run {run_id!r} was already resumed")


__all__ = [
    "DuplicateToolCallError",
    "GuardrailTripwireTriggered",
    "MaxTokensExceeded",
    "MaxTurnsExceeded",
    "ModelBehaviorError",
    "OperatorError",
    "RunAlreadyResumed",
    "RunTimeout",
    "RunaError",
    "ServiceUnavailable",
    "ToolError",
    "UnexpectedError",
    "UserError",
]
