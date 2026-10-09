"""run_state.py: `RunState`, enough of a paused run to resume it once approvals are resolved."""

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from runa._items import ConversationItem
from runa._types import InputTokensDetails, OutputTokensDetails, RunContextWrapper, Usage
from runa.approval import ApprovalLedger
from runa.exceptions import UserError
from runa.guardrail import GuardrailAudit, GuardrailResults
from runa.run_internal.agent_shape import _normalized_handoffs
from runa.tool import ToolCall
from runa.tracing.traces import Trace

_SCHEMA_VERSION = 2


@dataclass
class Interruption:
    """One tool call paused on `needs_approval`, surfaced to the caller to resolve.

    `name` is the tool's own name, which is all a decision needs: the sticky ledger is keyed by
    it, and the resumed run looks the tool itself up again off the agent's shape. Carrying the
    `FunctionTool` here as well would mean `from_json` had to rebuild one from a name, which for
    an MCP tool means connecting to the server just to read a deserialized blob.

    Which run a paused call belongs to isn't here either: a call surfaced by a `.delegate`
    subagent appears in its caller's `interruptions` as the very same `Interruption`, so it is
    the stash of paused delegates that knows whose it is (`runa.paused_delegates`), not the
    interruption and not whichever `RunState` happens to be holding it.
    """

    name: str
    arguments: str
    call_id: str
    agent: Any


# --- JSON schema (Pydantic) -------------------------------------------------------------
#
# `RunState`'s own dataclass fields hold live objects (an `Agent` instance, a `Trace`'s spans)
# that can't round-trip through JSON. These Pydantic models are the actual JSON
# envelope `to_json`/`to_string`/`from_json`/`from_string` serialize through -- kept separate
# from the runtime dataclasses above, which stay plain dataclasses (hot-path, mutated in place,
# and partly non-serializable by nature). This is Runa's only use of Pydantic; everywhere else
# in the codebase is dataclass-only.


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


class _RunStateSchema(BaseModel):
    """The JSON envelope a `RunState` actually serializes through.

    `context` and the guardrail audit trail (see `GuardrailResults`) aren't included: a
    guardrail's `output_info` isn't guaranteed JSON-safe, and a dataclass `context` loses its
    original type on the way back out (see `RunState.from_json`'s docstring).

    `delegates` and `delegate_owners` are the paused-delegate stash's two halves, straight off
    `PausedDelegates` and handed back to it by `from_json` -- which run owns a pending call is
    recorded here rather than left to be inferred from the shape of what came back.
    """

    schema_version: Literal[2]
    agent_name: str
    original_input: list[dict[str, Any]]
    generated_items: list[dict[str, Any]]
    ready_results: list[dict[str, Any]]
    new_items: list[dict[str, Any]] = []
    session_input: list[dict[str, Any]] = []
    pending: list[_InterruptionSchema]
    approvals: dict[str, bool]
    rejection_messages: dict[str, str] = {}
    context: Any = None
    usage: _UsageSchema
    approval_ledger: dict[str, bool] = {}
    approval_ledger_messages: dict[str, str] = {}
    executed_call_ids: list[str] = []
    trace_id: str
    trace_name: str
    trace_start_time: float
    delegates: dict[str, _RunStateSchema] = {}
    delegate_owners: dict[str, str] = {}


def _context_to_json(context: Any) -> Any:
    """A dataclass `context` becomes a plain dict; anything else passes through as-is."""
    if dataclasses.is_dataclass(context) and not isinstance(context, type):
        return dataclasses.asdict(context)
    return context


def _find_agent_by_name(root: Any, name: str) -> Any:
    """BFS `root` and everything reachable via its handoffs and delegates, matching on `.name`.

    Deserialization carries an agent as a name string, so resuming has to find the instance
    again: a handoff may have switched the current agent before the pause, or the paused call
    may belong to a delegate, so the match isn't necessarily `root` itself.
    """
    seen: set[int] = set()
    queue: list[Any] = [root]
    while queue:
        candidate = queue.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if getattr(candidate, "name", None) == name:
            return candidate
        queue.extend(
            handoff.agent
            for handoff in _normalized_handoffs(getattr(candidate, "handoffs", [])).values()
        )
        queue.extend(
            tool.delegate
            for tool in getattr(candidate, "tools", [])
            if getattr(tool, "delegate", None) is not None
        )
    raise UserError(f"no agent named {name!r} reachable from {getattr(root, 'name', root)!r}")


@dataclass
class RunState(GuardrailAudit):
    """Enough of a paused run to resume it once its `interruptions` are approved or rejected.

    `generated_items` ends with the assistant message that requested the paused calls;
    `ready_results` holds results already computed this turn for calls in that same message that
    *didn't* need approval: they're carried forward rather than re-executed on resume.
    `new_items` is what this run generated before the pause, and `session_input` the user turn
    a session-backed run still has to persist: both are saved once the resumed run finishes.

    `to_json()`/`to_string()`/`from_json()`/`from_string()` let a paused run survive a process
    restart -- see their docstrings for what is (and isn't) preserved.
    """

    agent: Any
    original_input: list[ConversationItem]
    generated_items: list[ConversationItem]
    ready_results: list[ConversationItem]
    pending: list[Interruption]
    context_wrapper: RunContextWrapper
    trace: Trace
    new_items: list[ConversationItem] = field(default_factory=list)
    session_input: list[ConversationItem] = field(default_factory=list)
    approvals: dict[str, bool] = field(default_factory=dict)
    rejection_messages: dict[str, str] = field(default_factory=dict)
    guardrail_results: GuardrailResults = field(default_factory=GuardrailResults)
    tool_calls: list[ToolCall] = field(default_factory=list)
    """The calls that already ran before the pause, so the resumed run's `Run` reports the whole
    turn's. Not serialized, for the same reason `trace`'s spans aren't: a restored run reports
    what it did after being restored."""

    def approve(self, interruption: Interruption, *, always: bool = False) -> None:
        """Mark `interruption` approved; its tool runs when the run is resumed.

        `always=True` also records a sticky decision on `context_wrapper.approval_ledger`:
        every future call to this tool (by name), in this run or a nested delegate call sharing
        this context (see `RunContextWrapper.fork()`), skips the approval prompt entirely.
        """
        if always:
            self.context_wrapper.approval_ledger.record(interruption.name, approved=True)
        self._deciding(interruption).approvals[interruption.call_id] = True

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
            self.context_wrapper.approval_ledger.record(
                interruption.name, approved=False, message=rejection_message
            )
        deciding = self._deciding(interruption)
        deciding.approvals[interruption.call_id] = False
        if rejection_message is not None:
            deciding.rejection_messages[interruption.call_id] = rejection_message

    def _deciding(self, interruption: Interruption) -> RunState:
        """The state a decision about `interruption` belongs on: a paused delegate's, or this one.

        A `.delegate` subagent's paused call rides up into its caller's `interruptions`, but it is
        the delegate's own run that will execute it, so the decision is recorded there and this
        state keeps only the calls it will run itself. Which is which is the paused-delegate
        stash's to say (`runa.paused_delegates`).
        """
        return self.context_wrapper.paused_delegates.owner_of(interruption) or self

    def _to_schema(self, *, nested: bool = False) -> _RunStateSchema:
        """This state as JSON; `nested` for a delegate's, whose shared context the caller holds."""
        usage = self.context_wrapper.usage
        ledger = self.context_wrapper.approval_ledger
        delegates = self.context_wrapper.paused_delegates
        return _RunStateSchema(
            schema_version=_SCHEMA_VERSION,
            agent_name=self.agent.name,
            original_input=self.original_input,
            generated_items=self.generated_items,
            ready_results=self.ready_results,
            new_items=self.new_items,
            session_input=self.session_input,
            pending=[
                _InterruptionSchema(
                    name=i.name, arguments=i.arguments, call_id=i.call_id, agent_name=i.agent.name
                )
                for i in self.pending
            ],
            approvals=self.approvals,
            rejection_messages=self.rejection_messages,
            context=_context_to_json(self.context_wrapper.context),
            usage=_UsageSchema(
                requests=usage.requests,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                input_tokens_details=dataclasses.asdict(usage.input_tokens_details),
                output_tokens_details=dataclasses.asdict(usage.output_tokens_details),
            ),
            approval_ledger=ledger.sticky,
            approval_ledger_messages=ledger.sticky_messages,
            executed_call_ids=sorted(ledger.executed),
            trace_id=self.trace.id,
            trace_name=self.trace.name,
            trace_start_time=self.trace.start_time,
            delegates={}
            if nested
            else {
                call_id: state._to_schema(nested=True)
                for call_id, state in delegates.waiting.items()
            },
            delegate_owners={} if nested else dict(delegates.owners),
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
    def from_json(cls, initial_agent: Any, state_json: dict[str, Any]) -> RunState:
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

        context_wrapper = RunContextWrapper(
            context=schema.context,
            approval_ledger=ApprovalLedger(
                sticky=dict(schema.approval_ledger),
                sticky_messages=dict(schema.approval_ledger_messages),
                executed=set(schema.executed_call_ids),
            ),
        )
        context_wrapper.paused_delegates.restore(
            {
                call_id: cls._from_schema(initial_agent, delegate, context_wrapper.fork())
                for call_id, delegate in schema.delegates.items()
            },
            schema.delegate_owners,
        )
        return cls._from_schema(initial_agent, schema, context_wrapper)

    @classmethod
    def _from_schema(
        cls, initial_agent: Any, schema: _RunStateSchema, context_wrapper: RunContextWrapper
    ) -> RunState:
        """Rebuild one state (the caller's, or a paused delegate's) onto `context_wrapper`."""
        state = cls(
            agent=_find_agent_by_name(initial_agent, schema.agent_name),
            original_input=schema.original_input,
            generated_items=schema.generated_items,
            ready_results=schema.ready_results,
            pending=[],
            context_wrapper=context_wrapper,
            trace=Trace(
                id=schema.trace_id, name=schema.trace_name, start_time=schema.trace_start_time
            ),
            new_items=schema.new_items,
            session_input=schema.session_input,
            approvals=dict(schema.approvals),
            rejection_messages=dict(schema.rejection_messages),
        )
        context_wrapper.usage = Usage(
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
    def from_string(cls, initial_agent: Any, state_string: str) -> RunState:
        """As `from_json()`, but from a JSON string produced by `to_string()`."""
        try:
            state_json = json.loads(state_string)
        except json.JSONDecodeError as exc:
            raise UserError(f"invalid RunState JSON: {exc}") from exc
        return cls.from_json(initial_agent, state_json)


__all__ = ["Interruption", "RunState"]
