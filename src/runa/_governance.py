"""_governance.py: `_Governance`, how one run governs itself, out of user code's reach.

The approval ledger, the paused-delegate stash and the guardrail audit trail, held together so
the loop, `RunState` and a delegate pass one value around. Not on `RunContext`, which is what
application code is handed. A `.delegate` subagent's run shares its caller's by reference.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from runa.approval import ApprovalLedger
from runa.guardrail import GuardrailResults

if TYPE_CHECKING:
    from runa.run_state import RunState


@dataclass
class _Governance:
    """One run's approval ledger, paused delegates and guardrail audit trail.

    `paused_delegates` maps a `.delegate` tool call's id to the nested `RunState` that call paused
    on, set aside until resuming the caller reaches the same call again.
    """

    approval_ledger: ApprovalLedger = field(default_factory=ApprovalLedger)
    paused_delegates: dict[str, RunState] = field(default_factory=dict)
    guardrail_results: GuardrailResults = field(default_factory=GuardrailResults)


__all__ = ["_Governance"]
