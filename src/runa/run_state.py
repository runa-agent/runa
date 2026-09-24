"""run_state.py: `RunState`, enough of a paused run to resume it once approvals are resolved."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from runa._types import (
    InputTokensDetails,
    OutputTokensDetails,
    RunContextWrapper,
    TResponseInputItem,
    Usage,
)
from runa.exceptions import UserError
from runa.tool import FunctionTool
from runa.tracing.traces import Trace

_SCHEMA_VERSION = 1


@dataclass
class Interruption:
    """One tool call paused on `needs_approval`, surfaced to the caller to resolve.

    `owner` is the `RunState` the call belongs to: the caller's own, or a delegate's nested one
    when the paused call came from a `.delegate` subagent. Resolving it records the decision on
    both, so resuming the caller resumes the delegate right where it stopped.
    """

    name: str
    arguments: str
    call_id: str
    tool: FunctionTool
    agent: Any
    owner: RunState | None = field(default=None, repr=False)


# --- JSON schema (Pydantic) -------------------------------------------------------------
#
# `RunState`'s own dataclass fields hold live objects (an `Agent` instance, a `FunctionTool`
# closure) that can't round-trip through JSON. These Pydantic models are the actual JSON
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
    """JSON-safe mirror of `Interruption`; `tool`/`agent` become a name string to resolve later."""

    name: str
    arguments: str
    call_id: str
    agent_name: str
    owner_call_id: str | None = None


class _RunStateSchema(BaseModel):
    """The JSON envelope a `RunState` actually serializes through.

    `context` and the four guardrail-result lists (see `GuardrailResult`) aren't included: a
    guardrail's `output_info` isn't guaranteed JSON-safe, and a dataclass `context` loses its
    original type on the way back out (see `RunState.from_json`'s docstring).
    """

    schema_version: Literal[1]
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


def _context_to_json(context: Any) -> Any:
    """A dataclass `context` becomes a plain dict; anything else passes through as-is."""
    if dataclasses.is_dataclass(context) and not isinstance(context, type):
        return dataclasses.asdict(context)
    return context


@dataclass
class RunState:
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
    original_input: list[TResponseInputItem]
    generated_items: list[TResponseInputItem]
    ready_results: list[TResponseInputItem]
    pending: list[Interruption]
    context_wrapper: RunContextWrapper
    trace: Trace
    new_items: list[TResponseInputItem] = field(default_factory=list)
    session_input: list[TResponseInputItem] = field(default_factory=list)
    approvals: dict[str, bool] = field(default_factory=dict)
    rejection_messages: dict[str, str] = field(default_factory=dict)
    input_guardrail_results: list[Any] = field(default_factory=list)
    output_guardrail_results: list[Any] = field(default_factory=list)
    tool_input_guardrail_results: list[Any] = field(default_factory=list)
    tool_output_guardrail_results: list[Any] = field(default_factory=list)

    def approve(self, interruption: Interruption, *, always: bool = False) -> None:
        """Mark `interruption` approved; its tool runs when the run is resumed.

        `always=True` also records a sticky decision on `context_wrapper.approval_ledger`:
        every future call to this tool (by name), in this run or a nested delegate call sharing
        this context (see `RunContextWrapper.fork()`), skips the approval prompt entirely.
        """
        self.approvals[interruption.call_id] = True
        if always:
            self.context_wrapper.approval_ledger[interruption.tool.name] = True
            self.context_wrapper.approval_ledger_messages.pop(interruption.tool.name, None)
        if interruption.owner not in (None, self):
            interruption.owner.approve(interruption, always=always)

    def reject(
        self,
        interruption: Interruption,
        *,
        always: bool = False,
        rejection_message: str | None = None,
    ) -> None:
        """Mark `interruption` rejected; its tool is skipped when the run is resumed.

        `rejection_message`, if given, is fed back to the model instead of the default
        "rejected by the operator" text. `always=True` sticks the rejection (and message, if
        any) for every future call to this tool, the same way `approve(always=True)` does.
        """
        self.approvals[interruption.call_id] = False
        if rejection_message is not None:
            self.rejection_messages[interruption.call_id] = rejection_message
        if always:
            self.context_wrapper.approval_ledger[interruption.tool.name] = False
            if rejection_message is not None:
                self.context_wrapper.approval_ledger_messages[interruption.tool.name] = (
                    rejection_message
                )
        if interruption.owner not in (None, self):
            interruption.owner.reject(
                interruption, always=always, rejection_message=rejection_message
            )

    def _to_schema(self, *, nested: bool = False) -> _RunStateSchema:
        """This state as JSON; `nested` for a delegate's, whose shared context the caller holds."""
        usage = self.context_wrapper.usage
        delegates = self.context_wrapper.paused_delegates
        owners = {id(state): call_id for call_id, state in delegates.items()}
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
                    name=i.name,
                    arguments=i.arguments,
                    call_id=i.call_id,
                    agent_name=i.agent.name,
                    owner_call_id=owners.get(id(i.owner)),
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
            approval_ledger=self.context_wrapper.approval_ledger,
            approval_ledger_messages=self.context_wrapper.approval_ledger_messages,
            executed_call_ids=sorted(self.context_wrapper.executed_call_ids),
            trace_id=self.trace.id,
            trace_name=self.trace.name,
            trace_start_time=self.trace.start_time,
            delegates={}
            if nested
            else {call_id: state._to_schema(nested=True) for call_id, state in delegates.items()},
        )

    def to_json(self) -> dict[str, Any]:
        """Serialize this paused run to a JSON-compatible dict, to persist and resume later.

        Not included: `Interruption.tool`/`.agent` and `agent` are recorded by name and
        re-resolved by `from_json`/`from_string` against a fresh agent instance, since a
        `FunctionTool`'s closure and a live `Agent` can't round-trip through JSON; the four
        `*_guardrail_results` lists and trace spans aren't included either (see `RunState`'s
        docstring and `_RunStateSchema`'s).
        """
        return self._to_schema().model_dump(mode="json")

    def to_string(self) -> str:
        """As `to_json()`, but as a JSON string."""
        return self._to_schema().model_dump_json()

    @classmethod
    async def from_json(cls, initial_agent: Any, state_json: dict[str, Any]) -> RunState:
        """Rebuild a paused `RunState` from `to_json()`'s output.

        `initial_agent` is a fresh instance of the agent the run started with; the current agent
        (possibly switched by a handoff before the pause) and each pending interruption's tool
        are re-resolved from it by name -- a `FunctionTool`'s closure and a live `Agent` can't
        round-trip through JSON, so they were never serialized as objects in the first place.

        A `context` that was a dataclass comes back as a plain dict, not reinstantiated as its
        original class -- there's no type registry to reverse that with. Raises `UserError` for
        an unknown schema version, a malformed field, or a tool/agent name that can no longer be
        found -- never a raw `pydantic.ValidationError`.
        """
        try:
            schema = _RunStateSchema.model_validate(state_json)
        except ValidationError as exc:
            raise UserError(f"invalid RunState JSON: {exc}") from exc

        context_wrapper = RunContextWrapper(
            context=schema.context,
            approval_ledger=dict(schema.approval_ledger),
            approval_ledger_messages=dict(schema.approval_ledger_messages),
            executed_call_ids=set(schema.executed_call_ids),
        )
        for call_id, delegate in schema.delegates.items():
            context_wrapper.paused_delegates[call_id] = await cls._from_schema(
                initial_agent, delegate, context_wrapper.fork()
            )
        return await cls._from_schema(initial_agent, schema, context_wrapper)

    @classmethod
    async def _from_schema(
        cls, initial_agent: Any, schema: _RunStateSchema, context_wrapper: RunContextWrapper
    ) -> RunState:
        """Rebuild one state (the caller's, or a paused delegate's) onto `context_wrapper`."""
        from runa.run_internal.agent_runner_helpers import (
            _agent_tools,
            _find_agent_by_name,
            _find_tool,
        )

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
        for item in schema.pending:
            item_agent = _find_agent_by_name(initial_agent, item.agent_name)
            tool = _find_tool(await _agent_tools(item_agent), item.name)
            if tool is None:
                raise UserError(
                    f"tool {item.name!r} not found on agent {item_agent.name!r} while resuming"
                )
            owner = (
                context_wrapper.paused_delegates.get(item.owner_call_id)
                if item.owner_call_id is not None
                else state
            )
            state.pending.append(
                Interruption(
                    name=item.name,
                    arguments=item.arguments,
                    call_id=item.call_id,
                    tool=tool,
                    agent=item_agent,
                    owner=owner,
                )
            )
        return state

    @classmethod
    async def from_string(cls, initial_agent: Any, state_string: str) -> RunState:
        """As `from_json()`, but from a JSON string produced by `to_string()`."""
        try:
            state_json = json.loads(state_string)
        except json.JSONDecodeError as exc:
            raise UserError(f"invalid RunState JSON: {exc}") from exc
        return await cls.from_json(initial_agent, state_json)


__all__ = ["Interruption", "RunState"]
