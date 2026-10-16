"""_governance.py: `_Governance`, how one run governs itself, out of user code's reach.

Three protocols make a run more than a loop of model calls, and each is already one object that
owns its rules: the approval ledger (`runa.approval.ApprovalLedger`), the paused-delegate stash
(`runa.paused_delegates.PausedDelegates`), and the guardrail audit trail
(`runa.guardrail.GuardrailResults`). This holds the three together, so the loop, `RunState` and a
delegate pass one value around rather than three.

It is deliberately not on `RunContext`, which is what a tool, an `@approval` predicate, a hook and
an `instructions` callable are handed. Those are application code; this is the framework's own
bookkeeping, and a rename inside the ledger must not be a breaking change for an app that was
legitimately given it.

A `.delegate` subagent's run shares its caller's `_Governance` by reference -- so the caller's
"always" covers the delegate's matching tool, a call id replayed at either depth is refused, and
every guardrail that ran at any depth lands in one trail -- while getting a `RunContext` of its own
over the same `context`, so its `usage` starts at zero and is merged back explicitly by the caller.
That pair is the whole of what "forking" a run means, which is why there is no `fork()` here: the
shared half is this very object.
"""

from dataclasses import dataclass, field

from runa.approval import ApprovalLedger
from runa.guardrail import GuardrailResults
from runa.paused_delegates import PausedDelegates


@dataclass
class _Governance:
    """One run's approval ledger, paused delegates and guardrail audit trail, shared by delegates.

    `approval_ledger` is sticky per-tool decisions and the replay guard; `paused_delegates` is
    which delegate runs this one paused on and whose decision each of their interruptions is;
    `guardrail_results` is every `GuardrailResult` produced, tripped or not, keyed by `Phase`.
    """

    approval_ledger: ApprovalLedger = field(default_factory=ApprovalLedger)
    paused_delegates: PausedDelegates = field(default_factory=PausedDelegates)
    guardrail_results: GuardrailResults = field(default_factory=GuardrailResults)


__all__ = ["_Governance"]
