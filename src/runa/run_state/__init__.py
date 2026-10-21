"""run_state.py: `RunState`, enough of a paused run to resume it once approvals are resolved.

A run is one `_RunRecord`, the facts about it that outlive the call running it, held by the turn
loop as it goes and serialized as is. A fact a run starts carrying is a field there, and nowhere
else: it reaches a paused state, its stored row and a resumed run with no further edit.

A pause is kept in `runa.db.paused_runs()` the moment it happens, under a `run_id`, and resuming
it claims that row: an atomic delete, so of any number of resumes of one pause, in one process
or across replicas, exactly one runs. The blob a row holds is this module's own format, never
handed out, so no copy of a pause exists that could be resumed a second time.
"""

import copy
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
from runa._loop import run_sync
from runa._types import RunContext
from runa.approval import ApprovalLedger
from runa.exceptions import UserError
from runa.guardrail import GuardrailAudit, GuardrailResults
from runa.run_internal.agent_shape import reachable
from runa.session import Session
from runa.tracing.traces import Trace
from runa.transcript import Transcript

if TYPE_CHECKING:
    from runa.agent import Agent

_SCHEMA_VERSION = 7


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
    """A live agent passes through; a name is found again from the `root` a load was given."""
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
    `FunctionTool` here as well would mean `load` had to rebuild one from a name, which for
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
    hooks, its `Session` object, its stream, its open span -- isn't here, and isn't resumed.

    * `start`: the agent the run began with. The run as a whole is its: the turn, token and time
      ceilings, the context sources, the compaction of history, and the key a pause is kept under.
    * `agent`: the agent running now, the same as `start` until a handoff.
    * `run_context`: what user code is told, its `context`, `user_id` and usage so far.
    * `session_id`: the id of the session the run started in, which its turn is saved to once it
      completes. `None` for a session-less run, and for every delegate's.
    * `turns`: the model calls the turn loop has made, across its resumes: what `max_turns` is
      measured against.
    * `elapsed`: the seconds the run has spent running, across its resumes: what its `timeout`
      is measured against. Time spent paused doesn't count.
    * `pending`: the calls a paused run waits on, all in its `paused_message`. Empty while the
      run is going.
    * `ready_results`: results already computed for that message's calls that needed no
      approval, carried forward rather than run again.
    * `delegates`: each paused `.delegate` call's own record, by call id, set aside until
      resuming this run reaches the same call again.
    """

    start: _AgentRef
    agent: _AgentRef
    transcript: Transcript = Field(default_factory=Transcript)
    run_context: RunContext[Any] = Field(default_factory=RunContext)
    session_id: str | None = None
    turns: int = 0
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
    """The JSON envelope a `RunState` is kept as: its record, and what its tree shares.

    The guardrail audit trail isn't serialized: each result is already a span in the trace. Nor
    is `run_id`, the key the row is kept under beside the starting agent's name, nor the run's
    `Session` object, whose id the record carries.
    """

    schema_version: Literal[7]
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

    It carries the session the run started in too, so resuming saves the turn there with no
    `session=` passed: by id, resolved through `runa.db.session`, or, resumed in the process that
    paused, as the very `Session` object the run was given.

    `run_id` is the handle the pause is kept under in `runa.db`, and a pause resumes once: the
    resumed run claims it, and resuming it again, from this object or from `load`, is an error
    `Run`. `load(agent, run_id)` is how another process, or a later request, gets it back.
    """

    _record: _RunRecord
    trace: Trace
    _approvals: ApprovalLedger
    guardrail_results: GuardrailResults = field(default_factory=GuardrailResults)
    run_id: str = ""
    _session: Session | None = field(default=None, repr=False)

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
    def session_id(self) -> str | None:
        """The id of the session the run started in; `None` for a session-less run."""
        return self._record.session_id

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
        from runa import db

        state_json = await db.paused_runs().load(agent.name, run_id)
        if state_json is None:
            return None
        state = cls._from_json(agent, state_json)
        state.run_id = run_id
        return state

    @classmethod
    def load_sync(cls, agent: Agent, run_id: str) -> RunState | None:
        """Synchronous `load`, for a caller resuming with `run_sync`."""
        return run_sync(cls.load(agent, run_id))

    async def _keep(self) -> None:
        """Save this pause as its starting agent's, under the new `run_id` it is resumed by."""
        from runa import db

        self.run_id = await db.paused_runs().save(self._record.start.name, self._to_json())

    async def _claim(self) -> bool:
        """Take this pause off the store to resume it; `False` if it was already resumed."""
        from runa import db

        return await db.paused_runs().claim(self._record.start.name, self.run_id)

    def _session_for(self, passed: Session | str | None) -> Session | None:
        """The session to resume this run in: its own, or the `Session` object `passed` for it.

        A `Session` object is accepted only for this run's own `session_id`: it names the store
        a session lives in when `runa.db.session` doesn't reach it, an object a pause loaded in
        another process doesn't have. Anything else is refused rather than used, since the turn
        belongs to the conversation it paused in.
        """
        session_id = self._record.session_id
        if passed is None:
            if self._session is not None or session_id is None:
                return self._session
            from runa import db

            return db.session(session_id, self._record.start.name)
        if isinstance(passed, str) or passed.session_id != session_id:
            given = passed if isinstance(passed, str) else passed.session_id
            raise UserError(
                f"a RunState already carries the session of the run it paused ({session_id!r}), "
                f"so `session={given!r}` would save its turn somewhere else: resume it without "
                "`session=`, or with the `Session` object of that same session."
            )
        return passed

    def _to_json(self) -> dict[str, Any]:
        """The row this pause is kept as: everything the run carries, agents by name.

        Not included: the guardrail audit trail and the trace's spans, both exported as spans.
        """
        schema = _RunStateSchema(
            schema_version=_SCHEMA_VERSION,
            trace=self.trace,
            approvals=self._approvals,
            run=self._record,
        )
        return schema.model_dump(mode="json")

    @classmethod
    def _from_json(cls, root: Agent, state_json: dict[str, Any]) -> RunState:
        """Rebuild the pause `_to_json` wrote, its agents found again from `root` by name."""
        try:
            schema = _RunStateSchema.model_validate(state_json, context={"root": root})
        except ValidationError as exc:
            raise UserError(f"invalid paused run: {exc}") from exc
        return cls(schema.run, schema.trace, schema.approvals)


__all__ = ["Interruption", "RunState"]
