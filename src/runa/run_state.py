"""run_state.py: `RunState`, enough of a paused run to resume it once approvals are resolved."""

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from runa._governance import _Governance
from runa._items import ConversationItem
from runa._types import InputTokensDetails, OutputTokensDetails, RunContext, Usage
from runa.agent_like import AgentLike
from runa.approval import ApprovalLedger
from runa.exceptions import UserError
from runa.guardrail import GuardrailAudit, GuardrailResults
from runa.handoff import reachable
from runa.tracing.traces import Trace
from runa.transcript import Transcript, _TranscriptSchema

_SCHEMA_VERSION = 4


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
    agent: AgentLike


# --- JSON schema (Pydantic) -------------------------------------------------------------
#
# `RunState`'s own dataclass fields hold live objects (an `Agent` instance, a `Trace`'s spans)
# that can't round-trip through JSON. These Pydantic models are the actual JSON
# envelope `to_json`/`to_string`/`from_json`/`from_string` serialize through -- kept separate
# from the runtime dataclasses above, which stay plain dataclasses (hot-path, mutated in place,
# and partly non-serializable by nature).


class _UsageSchema(BaseModel):
    """JSON-safe mirror of `Usage`."""

    requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: dict[str, int]
    output_tokens_details: dict[str, int]


class _InterruptionSchema(BaseModel):
    """JSON-safe mirror of `Interruption`; `agent` becomes a name string to resolve later."""

    name: str
    arguments: str
    call_id: str
    agent_name: str


class _ApprovalLedgerSchema(BaseModel):
    """JSON-safe mirror of `ApprovalLedger`; `executed` becomes a sorted list."""

    sticky: dict[str, bool]
    sticky_messages: dict[str, str]
    calls: dict[str, bool]
    call_messages: dict[str, str]
    executed: list[str]


class _RunStateSchema(BaseModel):
    """The JSON envelope a `RunState` actually serializes through.

    Every field is required, so a field `_to_schema` forgets is a type error there, and a blob
    missing a key is a `UserError` from `from_json`. The guardrail audit trail isn't serialized:
    each result is already a span in the trace `trace_id` names. `approval_ledger` and
    `delegates` are the run's governance, read back from the outermost state only.
    """

    schema_version: Literal[4]
    agent_name: str
    transcript: _TranscriptSchema
    ready_results: list[dict[str, Any]]
    pending: list[_InterruptionSchema]
    context: Any
    usage: _UsageSchema
    approval_ledger: _ApprovalLedgerSchema
    trace_id: str
    trace_name: str
    trace_start_time: float
    delegates: dict[str, _RunStateSchema]


def _context_to_json(context: Any) -> Any:
    """A dataclass `context` becomes a plain dict; anything else passes through as-is."""
    if dataclasses.is_dataclass(context) and not isinstance(context, type):
        return dataclasses.asdict(context)
    return context


def _find_agent_by_name(root: AgentLike, name: str) -> AgentLike:
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


@dataclass
class RunState(GuardrailAudit):
    """Enough of a paused run to resume it once its `interruptions` are approved or rejected.

    `transcript` is the conversation as it stood at the pause (see `Transcript`): its `items` end
    with the assistant message that requested the paused calls. `ready_results` holds results
    already computed this turn for calls in that same message that *didn't* need approval:
    they're carried forward rather than re-executed on resume.

    `run_context` is the paused run's `RunContext`, its `context` and usage so far. How the run
    governs itself -- the approval ledger, the paused delegates -- is `_governance`, which a
    caller reaches only through `approve`/`reject`, never directly (see `runa._governance`).

    `to_json()`/`to_string()`/`from_json()`/`from_string()` let a paused run survive a process
    restart -- see their docstrings for what is (and isn't) preserved.
    """

    agent: AgentLike
    transcript: Transcript
    ready_results: list[ConversationItem]
    pending: list[Interruption]
    run_context: RunContext
    _governance: _Governance
    trace: Trace
    guardrail_results: GuardrailResults = field(default_factory=GuardrailResults)

    @property
    def paused_message(self) -> ConversationItem:
        """The assistant message whose calls paused the run: the transcript's last, always.

        Where a resumed run picks the turn back up -- it finishes this one message's calls before
        calling the model again, with `ready_results` as the results it already has. The turn
        loop reads both from here rather than from a resumed-turn shape of its own.
        """
        return self.transcript.items[-1]

    def approve(self, interruption: Interruption, *, always: bool = False) -> None:
        """Mark `interruption` approved; its tool runs when the run is resumed.

        `always=True` also records a sticky decision on the run's approval ledger: every future
        call to this tool (by name), in this run or a nested delegate call sharing it (see
        `runa._governance`), skips the approval prompt entirely.
        """
        ledger = self._governance.approval_ledger
        if always:
            ledger.record(interruption.name, approved=True)
        ledger.record_call(interruption.call_id, approved=True)

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
        ledger = self._governance.approval_ledger
        if always:
            ledger.record(interruption.name, approved=False, message=rejection_message)
        ledger.record_call(interruption.call_id, approved=False, message=rejection_message)

    def _to_schema(self, *, nested: bool = False) -> _RunStateSchema:
        """This state as JSON; `nested` for a delegate's, whose governance the caller holds."""
        usage = self.run_context.usage
        ledger = self._governance.approval_ledger
        return _RunStateSchema(
            schema_version=_SCHEMA_VERSION,
            agent_name=self.agent.name,
            transcript=self.transcript._to_schema(),
            ready_results=self.ready_results,
            pending=[
                _InterruptionSchema(
                    name=i.name, arguments=i.arguments, call_id=i.call_id, agent_name=i.agent.name
                )
                for i in self.pending
            ],
            context=_context_to_json(self.run_context.context),
            usage=_UsageSchema(
                requests=usage.requests,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                input_tokens_details=dataclasses.asdict(usage.input_tokens_details),
                output_tokens_details=dataclasses.asdict(usage.output_tokens_details),
            ),
            approval_ledger=_ApprovalLedgerSchema(
                sticky={} if nested else ledger.sticky,
                sticky_messages={} if nested else ledger.sticky_messages,
                calls={} if nested else ledger.calls,
                call_messages={} if nested else ledger.call_messages,
                executed=[] if nested else sorted(ledger.executed),
            ),
            trace_id=self.trace.id,
            trace_name=self.trace.name,
            trace_start_time=self.trace.start_time,
            delegates={}
            if nested
            else {
                call_id: state._to_schema(nested=True)
                for call_id, state in self._governance.paused_delegates.items()
            },
        )

    def to_json(self) -> dict[str, Any]:
        """Serialize this paused run to a JSON-compatible dict, to persist and resume later.

        Not included: `agent` and `Interruption.agent` are recorded by name and re-resolved by
        `from_json`/`from_string` against a fresh agent instance, since a live `Agent` can't
        round-trip through JSON; the guardrail audit trail and trace spans aren't included
        either (see `RunState`'s docstring and `_RunStateSchema`'s).
        """
        return self._to_schema().model_dump(mode="json")

    def to_string(self) -> str:
        """As `to_json()`, but as a JSON string."""
        return self._to_schema().model_dump_json()

    @classmethod
    def from_json(cls, initial_agent: AgentLike, state_json: dict[str, Any]) -> RunState:
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
            schema = _RunStateSchema.model_validate(state_json)
        except ValidationError as exc:
            raise UserError(f"invalid RunState JSON: {exc}") from exc

        ledger = schema.approval_ledger
        governance = _Governance(
            approval_ledger=ApprovalLedger(
                sticky=dict(ledger.sticky),
                sticky_messages=dict(ledger.sticky_messages),
                calls=dict(ledger.calls),
                call_messages=dict(ledger.call_messages),
                executed=set(ledger.executed),
            ),
        )
        governance.paused_delegates.update(
            (call_id, cls._from_schema(initial_agent, delegate, governance))
            for call_id, delegate in schema.delegates.items()
        )
        return cls._from_schema(initial_agent, schema, governance)

    @classmethod
    def _from_schema(
        cls, initial_agent: AgentLike, schema: _RunStateSchema, governance: _Governance
    ) -> RunState:
        """Rebuild one state (the caller's, or a paused delegate's) onto the shared `governance`.

        Each gets a `RunContext` of its own, over the one `context` they all share, the way a
        delegate's nested run did before the pause: its usage is its own to merge back.
        """
        state = cls(
            agent=_find_agent_by_name(initial_agent, schema.agent_name),
            transcript=Transcript._from_schema(schema.transcript),
            ready_results=schema.ready_results,
            pending=[],
            run_context=RunContext(context=schema.context),
            _governance=governance,
            trace=Trace(
                id=schema.trace_id, name=schema.trace_name, start_time=schema.trace_start_time
            ),
        )
        state.run_context.usage = Usage(
            requests=schema.usage.requests,
            input_tokens=schema.usage.input_tokens,
            output_tokens=schema.usage.output_tokens,
            total_tokens=schema.usage.total_tokens,
            input_tokens_details=InputTokensDetails(**schema.usage.input_tokens_details),
            output_tokens_details=OutputTokensDetails(**schema.usage.output_tokens_details),
        )
        state.pending = [
            Interruption(
                name=item.name,
                arguments=item.arguments,
                call_id=item.call_id,
                agent=_find_agent_by_name(initial_agent, item.agent_name),
            )
            for item in schema.pending
        ]
        return state

    @classmethod
    def from_string(cls, initial_agent: AgentLike, state_string: str) -> RunState:
        """As `from_json()`, but from a JSON string produced by `to_string()`."""
        try:
            state_json = json.loads(state_string)
        except json.JSONDecodeError as exc:
            raise UserError(f"invalid RunState JSON: {exc}") from exc
        return cls.from_json(initial_agent, state_json)


__all__ = ["Interruption", "RunState"]
