"""run_state: `RunState`, the public handle on a paused run, to resolve and resume it.

What a pause is, and how it is kept in `runa.db` and claimed to resume once, is the runtime's
(`runa.run_internal.record`); this is what a caller reads and decides on.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from runa._items import ConversationItem
from runa._loop import run_sync
from runa._types import RunContext
from runa.guardrail import GuardrailAudit, GuardrailResults
from runa.run_internal.record import Interruption, Pause
from runa.tracing.traces import Trace
from runa.transcript import Transcript

if TYPE_CHECKING:
    from runa.agent import Agent


@dataclass
class RunState(GuardrailAudit):
    """Enough of a paused run to resume it once its `interruptions` are approved or rejected.

    The run itself, and what every run in its tree shares with it: the `trace`, the approval
    ledger `approve`/`reject` write to, and the guardrail audit trail. `transcript` is the
    conversation as it stood at the pause (see `Transcript`): its `items` end with the assistant
    message that requested the `pending` calls. `run_context` is the paused run's `RunContext`,
    its `context`, user and usage so far.

    It carries the session the run started in too, so resuming saves the turn there with no
    `session=` passed: by id, resolved through `runa.db.session`, or, resumed in the process that
    paused, as the very `Session` object the run was given.

    `run_id` is the handle the pause is kept under in `runa.db`, and a pause resumes once: the
    resumed run claims it, and resuming it again, from this object or from `load`, is an error
    `Run`. `load(agent, run_id)` is how another process, or a later request, gets it back.

    `pause` is the runtime's own record of all of this, which every property here reads.
    """

    pause: Pause

    @property
    def trace(self) -> Trace:
        """The paused run's trace, which the resumed run goes on adding spans to."""
        return self.pause.trace

    @property
    def guardrail_results(self) -> GuardrailResults:
        """Every guardrail that ran before the pause, keyed by the `Phase` it ran in."""
        return self.pause.guardrail_results

    @property
    def run_id(self) -> str:
        """The handle the pause is kept under in `runa.db`, for `load`."""
        return self.pause.run_id

    @property
    def agent(self) -> Agent:
        """The agent the run paused in, the starting one unless a handoff switched it."""
        return self.pause.record.agent

    @property
    def pending(self) -> list[Interruption]:
        """The calls waiting on a decision."""
        return self.pause.record.pending

    @property
    def transcript(self) -> Transcript:
        """The conversation as it stood at the pause."""
        return self.pause.record.transcript

    @property
    def run_context(self) -> RunContext[Any]:
        """The paused run's `context`, user and usage so far."""
        return self.pause.record.run_context

    @property
    def session_id(self) -> str | None:
        """The id of the session the run started in; `None` for a session-less run."""
        return self.pause.record.session_id

    @property
    def paused_message(self) -> ConversationItem:
        """The assistant message whose calls paused the run: the transcript's last, always."""
        return self.pause.record.paused_message

    def approve(self, interruption: Interruption, *, always: bool = False) -> None:
        """Mark `interruption` approved; its tool runs when the run is resumed.

        `always=True` also records a sticky decision on the run's approval ledger: every future
        call to this tool (by name), in this run or a nested delegate call sharing it, skips the
        approval prompt entirely.
        """
        if always:
            self.pause.approvals.record(interruption.name, approved=True)
        self.pause.approvals.record_call(interruption.call_id, approved=True)

    def reject(
        self,
        interruption: Interruption,
        *,
        always: bool = False,
        rejection_message: str | None = None,
    ) -> None:
        """Mark `interruption` rejected; its tool is skipped when the run is resumed.

        `rejection_message`, if given, is fed back to the model instead of
        `ApprovalLedger.DEFAULT_REJECTION`. `always=True` sticks the rejection (and message, if
        any) for every future call to this tool, the same way `approve(always=True)` does.
        """
        if always:
            self.pause.approvals.record(
                interruption.name, approved=False, message=rejection_message
            )
        self.pause.approvals.record_call(
            interruption.call_id, approved=False, message=rejection_message
        )

    @classmethod
    async def load(cls, agent: Agent, run_id: str) -> RunState | None:
        """The pause `agent`'s run is kept under as `run_id`; `None` if there is none.

        `agent` is a fresh instance of the agent the run started with, and a pause is found only
        through it. The agent running at the pause (a handoff may have switched it) and each
        pending interruption's agent are found again from it by name. Loading resolves no tools,
        so it never reaches an agent's MCP servers, and it spends nothing: the pause is claimed
        when the run resumes, so a decision about a call the pause doesn't have can still be
        refused first. A pause already resumed is `None`, like one that never existed.

        A `context` that was a dataclass comes back as a plain dict, not its original class.
        Raises `UserError` for a pause this Runa can no longer rebuild: an unknown schema
        version, a malformed field, or an agent name no longer reachable from `agent`.
        """
        pause = await Pause.load(agent, run_id)
        return cls(pause) if pause is not None else None

    @classmethod
    def load_sync(cls, agent: Agent, run_id: str) -> RunState | None:
        """Synchronous `load`, for a caller resuming with `run_sync`."""
        return run_sync(cls.load(agent, run_id))


__all__ = ["Interruption", "RunState"]
