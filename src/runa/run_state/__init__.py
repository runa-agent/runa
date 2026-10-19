"""run_state.py: `RunState`, enough of a paused run to resume it once approvals are resolved.

A run is one `_RunRecord`, the facts about it that outlive the call running it, held by the turn
loop as it goes and serialized as is. A fact a run starts carrying is a field there, and nowhere
else: it reaches a paused state, its JSON and a resumed run with no further edit.
"""

import copy
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    BaseModel,
    Field,
    PlainSerializer,
    PlainValidator,
    ValidationError,
    ValidationInfo,
)

from runa._items import ConversationItem
from runa._types import RunContext
from runa.approval import ApprovalLedger
from runa.exceptions import UserError
from runa.guardrail import GuardrailAudit, GuardrailResults
from runa.handoff import reachable
from runa.tracing.traces import Trace
from runa.transcript import Transcript

if TYPE_CHECKING:
    from runa.agent import Agent

_SCHEMA_VERSION = 5


def _find_agent_by_name(root: Agent, name: str) -> Agent:
    """`root`, or the agent reachable from it, that is named `name`.

    Deserialization carries an agent as a name string, so resuming has to find the instance
    again: a handoff may have switched the current agent before the pause, or the paused call
    may belong to a delegate, so the match isn't necessarily `root` itself. What reachable means
    is `runa.handoff`'s to say, not this module's -- resuming and `Agent.graph` walk one graph.
    """
    for candidate in reachable(root):
        if candidate.name == name:
            return candidate
    raise UserError(f"no agent named {name!r} reachable from {root.name!r}")


def _agent_from_json(value: Any, info: ValidationInfo) -> Any:
    """A live agent passes through; a name is found again from the `root` `from_json` was given."""
    if not isinstance(value, str):
        return value
    return _find_agent_by_name((info.context or {})["root"], value)


def _trace_from_json(value: Any) -> Trace:
    """A live trace passes through; its identity becomes a trace with no spans yet."""
    return value if isinstance(value, Trace) else Trace(**value)


def _trace_to_json(trace: Trace) -> dict[str, Any]:
    """What identifies `trace`; its spans are exported on their own."""
    return {
        "id": trace.id,
        "name": trace.name,
        "start_time": trace.start_time,
        "session_id": trace.session_id,
        "metadata": trace.metadata,
    }


if TYPE_CHECKING:
    _AgentRef = Agent
    _TraceRef = Trace
else:
    _AgentRef = Annotated[
        Any, PlainValidator(_agent_from_json), PlainSerializer(lambda agent: agent.name)
    ]
    """An `Agent`, written as its name: a live one can't round-trip through JSON."""
    _TraceRef = Annotated[Any, PlainValidator(_trace_from_json), PlainSerializer(_trace_to_json)]


@dataclass
class Interruption:
    """One tool call paused on `needs_approval`, surfaced to the caller to resolve.

    `name` is the tool's own name, which is all a decision needs: the sticky ledger is keyed by
    it, and the resumed run looks the tool itself up again off the agent's shape. Carrying the
    `FunctionTool` here as well would mean `from_json` had to rebuild one from a name, which for
    an MCP tool means connecting to the server just to read a deserialized blob.

    Which run a paused call belongs to isn't here either, and needn't be: a call surfaced by a
    `.delegate` subagent appears in its caller's `interruptions` as the very same `Interruption`,
    and its decision is recorded by `call_id` on the approval ledger every run in the tree
    shares, where the run that executes the call reads it.
    """

    name: str
    arguments: str
    call_id: str
    agent: _AgentRef


class _RunRecord(BaseModel):
    """One run, as it would be resumed: everything it is that outlives the call running it.

    The turn loop works on one directly; a paused `RunState` holds it, and so does a paused
    delegate's entry in its caller's `delegates`. Every field is data or written as data (an
    agent by its name), so all of it is serialized. What only the call running it holds -- its
    hooks, its session, its stream, its open span -- isn't here, and isn't resumed.

    * `agent`: the agent running now, the same as the starting one until a handoff.
    * `run_context`: what user code is told, its `context`, `user_id` and usage so far.
    * `elapsed`: the seconds the run has spent running, across its resumes: what its `timeout`
      is measured against. Time spent paused doesn't count.
    * `pending`: the calls a paused run waits on, all in its `paused_message`. Empty while the
      run is going.
    * `ready_results`: results already computed for that message's calls that needed no
      approval, carried forward rather than run again.
    * `delegates`: each paused `.delegate` call's own record, by call id, set aside until
      resuming this run reaches the same call again.
    """

    agent: _AgentRef
    transcript: Transcript = Field(default_factory=Transcript)
    run_context: RunContext[Any] = Field(default_factory=RunContext)
    elapsed: float = 0.0
    pending: list[Interruption] = Field(default_factory=list)
    ready_results: list[ConversationItem] = Field(default_factory=list)
    delegates: dict[str, _RunRecord] = Field(default_factory=dict)

    @property
    def paused_message(self) -> ConversationItem:
        """The assistant message whose calls paused the run: the transcript's last, always."""
        return self.transcript.items[-1]

    def resumed(self) -> _RunRecord:
        """A copy for a resumed run to work on, so this one still reads as it paused."""
        usage = copy.deepcopy(self.run_context.usage)
        return self.model_copy(
            update={
                "transcript": self.transcript.copy(),
                "run_context": RunContext(
                    self.run_context.context, self.run_context.user_id, usage
                ),
                "delegates": dict(self.delegates),
            }
        )


class _RunStateSchema(BaseModel):
    """The JSON envelope a `RunState` serializes through: its record, and what its tree shares.

    The guardrail audit trail isn't serialized: each result is already a span in the trace.
    """

    schema_version: Literal[5]
    trace: _TraceRef
    approvals: ApprovalLedger
    run: _RunRecord


@dataclass
class RunState(GuardrailAudit):
    """Enough of a paused run to resume it once its `interruptions` are approved or rejected.

    The run itself, and what every run in its tree shares with it: the `trace`, the approval
    ledger `approve`/`reject` write to, and the guardrail audit trail. `transcript` is the
    conversation as it stood at the pause (see `Transcript`): its `items` end with the assistant
    message that requested the `pending` calls. `run_context` is the paused run's `RunContext`,
    its `context`, user and usage so far.

    `to_json()`/`to_string()`/`from_json()`/`from_string()` let a paused run survive a process
    restart -- see their docstrings for what is (and isn't) preserved.
    """

    _record: _RunRecord
    trace: Trace
    _approvals: ApprovalLedger
    guardrail_results: GuardrailResults = field(default_factory=GuardrailResults)

    @property
    def agent(self) -> Agent:
        """The agent the run paused in, the starting one unless a handoff switched it."""
        return self._record.agent

    @property
    def pending(self) -> list[Interruption]:
        """The calls waiting on a decision."""
        return self._record.pending

    @property
    def transcript(self) -> Transcript:
        """The conversation as it stood at the pause."""
        return self._record.transcript

    @property
    def run_context(self) -> RunContext[Any]:
        """The paused run's `context`, user and usage so far."""
        return self._record.run_context

    @property
    def paused_message(self) -> ConversationItem:
        """The assistant message whose calls paused the run: the transcript's last, always."""
        return self._record.paused_message

    def approve(self, interruption: Interruption, *, always: bool = False) -> None:
        """Mark `interruption` approved; its tool runs when the run is resumed.

        `always=True` also records a sticky decision on the run's approval ledger: every future
        call to this tool (by name), in this run or a nested delegate call sharing it, skips the
        approval prompt entirely.
        """
        if always:
            self._approvals.record(interruption.name, approved=True)
        self._approvals.record_call(interruption.call_id, approved=True)

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
            self._approvals.record(interruption.name, approved=False, message=rejection_message)
        self._approvals.record_call(interruption.call_id, approved=False, message=rejection_message)

    def _to_schema(self) -> _RunStateSchema:
        return _RunStateSchema(
            schema_version=_SCHEMA_VERSION,
            trace=self.trace,
            approvals=self._approvals,
            run=self._record,
        )

    def to_json(self) -> dict[str, Any]:
        """Serialize this paused run to a JSON-compatible dict, to persist and resume later.

        Everything the run carries is included, agents by name. Not included: the guardrail
        audit trail and the trace's spans, both already exported as spans.
        """
        return self._to_schema().model_dump(mode="json")

    def to_string(self) -> str:
        """As `to_json()`, but as a JSON string."""
        return self._to_schema().model_dump_json()

    @classmethod
    def from_json(cls, initial_agent: Agent, state_json: dict[str, Any]) -> RunState:
        """Rebuild a paused `RunState` from `to_json()`'s output.

        `initial_agent` is a fresh instance of the agent the run started with; the current agent
        (possibly switched by a handoff before the pause) and each pending interruption's agent
        are re-resolved from it by name -- a live `Agent` can't round-trip through JSON, so it
        was never serialized as an object in the first place. Reading a blob is pure: the tools
        themselves are resolved by the resumed run, off the shape it builds anyway, so rebuilding
        a state never reaches out to an agent's MCP servers.

        A `context` that was a dataclass comes back as a plain dict, not reinstantiated as its
        original class -- there's no type registry to reverse that with. Raises `UserError` for
        an unknown schema version, a malformed field, or an agent name that can no longer be
        found -- never a raw `pydantic.ValidationError`.
        """
        try:
            schema = _RunStateSchema.model_validate(state_json, context={"root": initial_agent})
        except ValidationError as exc:
            raise UserError(f"invalid RunState JSON: {exc}") from exc
        return cls(schema.run, schema.trace, schema.approvals)

    @classmethod
    def from_string(cls, initial_agent: Agent, state_string: str) -> RunState:
        """As `from_json()`, but from a JSON string produced by `to_string()`."""
        try:
            state_json = json.loads(state_string)
        except json.JSONDecodeError as exc:
            raise UserError(f"invalid RunState JSON: {exc}") from exc
        return cls.from_json(initial_agent, state_json)


__all__ = ["Interruption", "RunState"]
