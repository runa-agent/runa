"""_governance.py: `_Governance`, how one run governs itself, out of user code's reach.

The approval ledger, the paused-delegate stash and the guardrail audit trail, held together so
the loop, `RunState` and a delegate pass one value around. Not on `RunContext`, which is what
application code is handed. A `.delegate` subagent's run shares its caller's by reference.
"""

from dataclasses import dataclass, field

from runa.approval import ApprovalLedger
from runa.guardrail import GuardrailResults
from runa.paused_delegates import PausedDelegates


@dataclass
class _Governance:
    """One run's approval ledger, paused delegates and guardrail audit trail."""

    approval_ledger: ApprovalLedger = field(default_factory=ApprovalLedger)
    paused_delegates: PausedDelegates = field(default_factory=PausedDelegates)
    guardrail_results: GuardrailResults = field(default_factory=GuardrailResults)


__all__ = ["_Governance"]
