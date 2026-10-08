"""Tests for the `@approval` decorator and the `ApprovalLedger` that decides each call."""

import asyncio
from typing import Any, cast

import pytest

from runa import approval, tool
from runa._types import RunContextWrapper
from runa.approval import ApprovalDecision, ApprovalLedger


def _run(predicate: Any, params: dict[str, Any], call_id: str = "call_1") -> bool:
    """Call a wrapped `needs_approval` callable directly, awaiting its always-async wrapper."""
    ctx = cast(RunContextWrapper[Any], RunContextWrapper(context=None))
    return asyncio.run(predicate(ctx, params, call_id))


def _decide(
    ledger: ApprovalLedger,
    *,
    needs_approval: bool = True,
    tool_name: str = "issue_refund",
    call_id: str = "call_1",
    approvals: dict[str, bool] | None = None,
    rejection_messages: dict[str, str] | None = None,
) -> ApprovalDecision:
    """Ask `ledger` about one call, with `needs_approval` standing in for the tool's predicate."""

    async def predicate() -> bool:
        return needs_approval

    return asyncio.run(
        ledger.decide(
            tool_name,
            call_id,
            needs_approval=predicate,
            approvals=approvals,
            rejection_messages=rejection_messages,
        )
    )


def test_approval_binds_params_by_name() -> None:
    """A predicate's parameters are looked up by name from the tool's parsed arguments."""

    @approval
    def requires_review(subject: str) -> bool:
        """Trip for refund-related subjects."""
        return "refund" in subject.lower()

    assert _run(requires_review, {"subject": "Refund request", "body": "..."})
    assert not _run(requires_review, {"subject": "Hello", "body": "..."})


def test_approval_works_on_lambda() -> None:
    """A bare lambda works the same as a `def`."""
    requires_review = approval(lambda subject: "refund" in subject.lower())

    assert _run(requires_review, {"subject": "Refund request"})
    assert not _run(requires_review, {"subject": "Hello"})


def test_approval_exposes_ctx_and_call_id_by_name() -> None:
    """Naming a parameter `ctx`/`call_id` receives the run context/call id, not a tool arg."""
    seen: dict[str, Any] = {}

    @approval
    def requires_review(ctx: RunContextWrapper[Any], call_id: str, subject: str) -> bool:
        seen["ctx"] = ctx
        seen["call_id"] = call_id
        return bool(subject)

    assert _run(requires_review, {"subject": "x"}, call_id="call_42")
    assert seen["call_id"] == "call_42"
    assert isinstance(seen["ctx"], RunContextWrapper)


def test_approval_supports_async_predicates() -> None:
    """An `async def` predicate is awaited before its result is used."""

    @approval
    async def requires_review(subject: str) -> bool:
        return "refund" in subject.lower()

    assert _run(requires_review, {"subject": "Refund request"})
    assert not _run(requires_review, {"subject": "Hello"})


def test_approval_wires_into_tool() -> None:
    """`approval(...)` plugs straight into `@tool(needs_approval=...)`."""

    @tool(needs_approval=approval(lambda subject: "refund" in subject.lower()))
    async def send_email(subject: str, body: str) -> str:
        """Send an email."""
        return f"Sent '{subject}'"

    assert callable(send_email.needs_approval)
    assert _run(send_email.needs_approval, {"subject": "Refund request", "body": "hi"})


def test_approval_unknown_param_raises_key_error() -> None:
    """A predicate parameter that matches no tool argument fails loudly, not silently."""

    @approval
    def requires_review(subjct: str) -> bool:  # typo, on purpose
        return bool(subjct)

    with pytest.raises(KeyError):
        _run(requires_review, {"subject": "x"})


def test_a_call_with_no_decision_yet_interrupts() -> None:
    """A tool that needs approval and has no decision recorded pauses the run."""
    assert _decide(ApprovalLedger()).action == "interrupt"


def test_a_tool_that_needs_no_approval_runs() -> None:
    """The tool's own predicate saying no is the end of it: no pause, no ledger lookup."""
    assert _decide(ApprovalLedger(), needs_approval=False).action == "run"


def test_a_per_call_decision_settles_a_call_that_needs_approval() -> None:
    """A resolved `RunState`'s per-call verdicts decide the calls they name."""
    ledger = ApprovalLedger()

    assert _decide(ledger, approvals={"call_1": True}).action == "run"
    assert _decide(ledger, approvals={"call_1": False}).action == "reject"


def test_sticky_beats_the_tools_own_predicate() -> None:
    """An "always" answer covers every later call to that tool, predicate or not."""
    approved = ApprovalLedger()
    approved.record("issue_refund", approved=True)
    rejected = ApprovalLedger()
    rejected.record("issue_refund", approved=False)

    assert _decide(approved).action == "run"
    assert _decide(rejected).action == "reject"


def test_sticky_beats_a_conflicting_per_call_decision() -> None:
    """Precedence is sticky, then the predicate, then per-call -- not the other way around."""
    ledger = ApprovalLedger()
    ledger.record("issue_refund", approved=False)

    assert _decide(ledger, approvals={"call_1": True}).action == "reject"


def test_a_rejection_with_no_message_falls_back_to_the_default() -> None:
    """Sticky and per-call rejections answer with the same default text, from one constant."""
    sticky = ApprovalLedger()
    sticky.record("issue_refund", approved=False)

    assert _decide(sticky).message == ApprovalLedger.DEFAULT_REJECTION
    assert (
        _decide(ApprovalLedger(), approvals={"call_1": False}).message
        == ApprovalLedger.DEFAULT_REJECTION
    )


def test_a_rejection_message_is_fed_back_instead_of_the_default() -> None:
    """A custom message replaces the default, whether it was recorded sticky or per call."""
    sticky = ApprovalLedger()
    sticky.record("issue_refund", approved=False, message="not authorized")

    assert _decide(sticky).message == "not authorized"
    assert (
        _decide(
            ApprovalLedger(),
            approvals={"call_1": False},
            rejection_messages={"call_1": "not this one"},
        ).message
        == "not this one"
    )


def test_recording_a_decision_clears_the_previous_ones_message() -> None:
    """A decision owns its message slot: a later one can't answer with the earlier one's text."""
    ledger = ApprovalLedger()
    ledger.record("issue_refund", approved=False, message="not authorized")
    ledger.record("issue_refund", approved=False)

    assert _decide(ledger).message == ApprovalLedger.DEFAULT_REJECTION

    ledger.record("issue_refund", approved=True)
    assert ledger.sticky_messages == {}


def test_a_claimed_call_id_cannot_be_claimed_twice() -> None:
    """The replay guard: an executed call id never runs again, until it's released."""
    ledger = ApprovalLedger()

    assert ledger.claim("call_1") is True
    assert ledger.claim("call_1") is False

    ledger.release("call_1")
    assert ledger.claim("call_1") is True


def test_a_forked_context_shares_one_ledger() -> None:
    """A delegate's decisions and claims are the caller's: one ledger, shared by reference."""
    ctx: RunContextWrapper[Any] = RunContextWrapper(context=None)
    forked = ctx.fork()

    forked.approval_ledger.record("issue_refund", approved=True)
    forked.approval_ledger.claim("call_1")

    assert ctx.approval_ledger.sticky == {"issue_refund": True}
    assert ctx.approval_ledger.claim("call_1") is False
