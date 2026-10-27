"""record.py: `RunRecord`, what a run is as it would be resumed, and `Pause`, one kept in `runa.db`.

A run is one `RunRecord`, the facts about it that outlive the call running it, held by the turn
loop as it goes and serialized as is. What every run in one tree shares, the root's and each
delegate's alike, is one `RunTree`. A fact a run starts carrying is a field on one of the two, and
nowhere else: it reaches a paused state, its stored row and a resumed run with no further edit.
A caller reads it through `RunView`, which `runa.Run` and `runa.RunState` both are: one property
there shows it on a finished run and a paused one alike.

A pause is kept in `runa.db.paused_runs()` the moment it happens, under a `run_id`, and resuming
it claims that row (see `PausedRunStore.claim`). The blob a row holds is this module's own
format, never handed out (see `runa.RunState` for why).

`runa.RunState` is the public handle over a `Pause`, and holds it out of a caller's reach: this
module builds a state (`paused`) and is the only one that reads the pause back out of one
(`resume_session`, `keep`, `resuming`), so nothing else can save, copy or resume a pause by hand.
"""

import copy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from runa._items import ConversationItem
from runa._types import RunContext, Usage
from runa.approval import ApprovalLedger
from runa.exceptions import UserError
from runa.guardrail import GuardrailAudit, GuardrailResults
from runa.run_internal.agent_shape import Agents
from runa.session import Session
from runa.tool import ToolCall
from runa.tracing.traces import Trace
from runa.transcript import Transcript

if TYPE_CHECKING:
    from runa.agent import Agent
    from runa.run_state import RunState

SCHEMA_VERSION = 11
"""The version a pause is written at; a row from `READABLE_SINCE` up to it is read as is.

A field added with a default is no new version: an older row reads with the default, so its pause
still resumes. Bump this, and `READABLE_SINCE` to match, only when a stored field's meaning
changes, which strands every pause kept before it.
"""

READABLE_SINCE = 11
"""The oldest `schema_version` this Runa still resumes."""


def _agent_from_json(value: Any, info: ValidationInfo) -> Any:
    """A live agent passes through; a name is found again among the `agents` a load was given.

    The match isn't necessarily the root: a handoff may have switched the current agent before
    the pause, or the paused call may belong to a delegate.
    """
    if not isinstance(value, str):
        return value
    return (info.context or {})["agents"].named(value)


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


class Answered(BaseModel):
    """A call answered with `item`, reused rather than run again."""

    kind: Literal["answered"] = "answered"
    item: ConversationItem


class Waiting(BaseModel):
    """A call waiting on a human to decide `interruption`."""

    kind: Literal["waiting"] = "waiting"
    interruption: Interruption


class Suspended(BaseModel):
    """A `.delegate` call whose run paused, kept as `record` to continue where it stopped."""

    kind: Literal["suspended"] = "suspended"
    record: RunRecord


type CallStanding = Annotated[Answered | Waiting | Suspended, Field(discriminator="kind")]
"""Where one call of a paused message stands, tagged by `kind`."""


class RunRecord(BaseModel):
    """One run, as it would be resumed: everything it is that outlives the call running it.

    The turn loop works on one directly; a `Pause` holds it, and so does a paused delegate's
    `Suspended` standing in its caller's record. Every field is data or written as data (an
    agent by its name), so all of it is serialized. What only the call running it holds -- its
    hooks, its `Session` object, its stream, its open span -- isn't here, and isn't resumed.

    * `start`: the agent the run began with. The run as a whole is its: the turn, token and time
      ceilings, the context sources, compaction, and the key a pause is kept under.
    * `agent`: the agent running now, the same as `start` until a handoff.
    * `run_context`: what user code is told, its `context`, `user_id` and usage so far.
    * `tool_calls`: every tool call the run answered, across its resumes, in the order they
      finished, each added through `record_call`. A fact about the run, not its conversation: a
      run that fails still reports it.
    * `session_id`: the id of the session the run started in, which its turn is saved to once it
      completes. `None` for a session-less run, and for every delegate's.
    * `turns`: the model calls the turn loop has made, across its resumes: what `max_turns` is
      measured against.
    * `elapsed`: the seconds the run has spent running, across its resumes: what its `timeout`
      is measured against. Time spent paused doesn't count.
    * `standings`: where the calls of the message being answered stand, by call id, each one
      `Answered`, `Waiting` or `Suspended`. A call with no standing yet is gated afresh, and a
      handoff never has one: a run pauses before making one, so resuming makes it.

    Standings are written through `answer`, `wait` and `suspend` as the message's calls settle,
    and cleared by `settle_message` once it is answered, so only a paused run's are read; an
    errored run's may show the message it failed on part-way.
    """

    start: _AgentRef
    agent: _AgentRef
    transcript: Transcript = Field(default_factory=Transcript)
    run_context: RunContext[Any] = Field(default_factory=RunContext)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    session_id: str | None = None
    turns: int = 0
    elapsed: float = 0.0
    standings: dict[str, CallStanding] = Field(default_factory=dict)

    @property
    def paused_message(self) -> ConversationItem:
        """The assistant message whose calls paused the run: the transcript's last, always."""
        return self.transcript.items[-1]

    @property
    def pending(self) -> list[Interruption]:
        """The calls the run waits on, each suspended delegate's in its place among them."""
        waiting: list[Interruption] = []
        for standing in self.standings.values():
            if isinstance(standing, Waiting):
                waiting.append(standing.interruption)
            elif isinstance(standing, Suspended):
                waiting.extend(standing.record.pending)
        return waiting

    def record_call(self, call: ToolCall) -> None:
        """Record `call` as the latest of the run's answered `tool_calls`."""
        self.tool_calls.append(call)

    def answer(self, call_id: str, item: ConversationItem) -> None:
        """Record call `call_id` as answered with `item`."""
        self.standings[call_id] = Answered(item=item)

    def wait(self, interruption: Interruption) -> None:
        """Record `interruption`'s call as waiting on a human."""
        self.standings[interruption.call_id] = Waiting(interruption=interruption)

    def suspend(self, call_id: str, record: RunRecord) -> None:
        """Record delegate call `call_id` as paused, its run kept as `record`."""
        self.standings[call_id] = Suspended(record=record)

    def answered(self, call_id: str) -> ConversationItem | None:
        """What call `call_id` was already answered with, if it was."""
        standing = self.standings.get(call_id)
        return standing.item if isinstance(standing, Answered) else None

    def suspended(self, call_id: str) -> RunRecord | None:
        """The run of delegate call `call_id` if it paused, to continue where it stopped."""
        standing = self.standings.get(call_id)
        return standing.record if isinstance(standing, Suspended) else None

    def settle_message(self) -> None:
        """Forget the standings of a message now answered."""
        self.standings = {}

    def resumed(self) -> RunRecord:
        """A copy for a resumed run to work on, so this one still reads as it paused.

        A call still `Waiting` keeps its place until it is gated again, which a resumed run does
        before it reads `pending`.
        """
        usage = copy.deepcopy(self.run_context.usage)
        return self.model_copy(
            update={
                "transcript": self.transcript.copy(),
                "tool_calls": list(self.tool_calls),
                "run_context": RunContext(
                    self.run_context.context, self.run_context.user_id, usage
                ),
                "standings": dict(self.standings),
            }
        )

    def snapshot(self) -> RunRecord:
        """A copy for a finished `Run` to read: what the loop records after it doesn't reach it.

        Its transcript and usage are this record's own, which nothing changes once the run has
        reported: a resumed run works on `resumed`'s copies.
        """
        return self.model_copy(update={"tool_calls": list(self.tool_calls)})


class RunTree(BaseModel):
    """What every run in one tree shares by reference: the root run's and each delegate's.

    * `trace`: the one trace every run's spans go in.
    * `approvals`: the ledger a decision is written to, and each run reads its calls' from.
    * `claimed_calls`: the id of every call that started running, written through `claim`.
    * `guardrail_results`: the audit trail of every guardrail any of them ran. Not serialized:
      each result is already a span in the trace.
    * `agents`: the instance each agent class runs as, the root run's starting agent among them.
      Not serialized: a load is given its root, and builds the rest again.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    trace: _TraceRef
    agents: Agents = Field(exclude=True)
    approvals: ApprovalLedger = Field(default_factory=ApprovalLedger)
    claimed_calls: set[str] = Field(default_factory=set)
    guardrail_results: GuardrailResults = Field(default_factory=GuardrailResults, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _loaded_agents(cls, data: Any, info: ValidationInfo) -> Any:
        """A tree read back from a pause runs the `agents` its load was given."""
        if isinstance(data, dict) and info.context is not None and "agents" not in data:
            return {**data, "agents": info.context["agents"]}
        return data

    def claim(self, call_id: str) -> bool:
        """Claim `call_id` to run it; `False` if it already ran, so it must not run again."""
        if call_id in self.claimed_calls:
            return False
        self.claimed_calls.add(call_id)
        return True

    def snapshot(self) -> RunTree:
        """The same trace, ledger and claims, with a copy of the audit trail as it stands now."""
        return self.model_copy(update={"guardrail_results": self.guardrail_results.snapshot()})


class RunView(GuardrailAudit):
    """What a caller reads about a run, through to its record and tree, never copied out of them.

    `runa.Run` and `runa.RunState` are both one, each answering `_record` and `_tree` with its
    own: a fact a caller sees on both is a property here, and nowhere else.
    """

    if TYPE_CHECKING:

        @property
        def _record(self) -> RunRecord: ...

        @property
        def _tree(self) -> RunTree: ...

    @property
    def trace(self) -> Trace:
        """The run's `Trace`, the one every run in its tree adds spans to."""
        return self._tree.trace

    @property
    def guardrail_results(self) -> GuardrailResults:
        """Every guardrail that ran, a delegate's included, keyed by the `Phase` it ran in."""
        return self._tree.guardrail_results

    @property
    def usage(self) -> Usage:
        """The run's token usage, every model and embedding call made on its behalf included."""
        return self._record.run_context.usage

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """Every tool call the run answered, across its resumes, in the order they finished."""
        return tuple(self._record.tool_calls)


class PauseSchema(BaseModel):
    """The JSON envelope a `Pause` is kept as: its record and its tree.

    Not serialized: `run_id`, the key the row is kept under beside the starting agent's name, and
    the run's `Session` object, whose id the record carries.
    """

    schema_version: int
    tree: RunTree
    run: RunRecord

    @field_validator("schema_version")
    @classmethod
    def _readable(cls, version: int) -> int:
        if not READABLE_SINCE <= version <= SCHEMA_VERSION:
            raise ValueError(
                f"kept at schema version {version}, and this Runa resumes "
                f"{READABLE_SINCE} to {SCHEMA_VERSION}"
            )
        return version


@dataclass
class Pause:
    """A paused root run: its record, and the tree every run in it shares.

    `session` is the very `Session` object the run was given, held only in the process that
    paused it; elsewhere the record's `session_id` is resolved again. `run_id` is the key it is
    kept under in `runa.db`, set by `keep` or `load`.
    """

    record: RunRecord
    tree: RunTree
    run_id: str = ""
    session: Session | None = field(default=None, repr=False)

    @classmethod
    async def load(cls, root: Agent, run_id: str) -> Pause | None:
        """The pause `root`'s run is kept under as `run_id`; `None` if there is none.

        Reads the row without claiming it, so loading spends nothing. Raises `UserError` for a
        row this Runa can no longer rebuild.
        """
        from runa import db

        state_json = await db.paused_runs().load(root.name, run_id)
        if state_json is None:
            return None
        pause = cls.from_json(root, state_json)
        pause.run_id = run_id
        return pause

    async def keep(self) -> None:
        """Save this pause as its starting agent's, under the new `run_id` it is resumed by."""
        from runa import db

        self.run_id = await db.paused_runs().save(self.record.start.name, self.to_json())

    async def claim(self) -> bool:
        """Take this pause off the store to resume it; `False` if it was already resumed."""
        from runa import db

        return await db.paused_runs().claim(self.record.start.name, self.run_id)

    def to_json(self) -> dict[str, Any]:
        """The row this pause is kept as: everything the run carries, agents by name.

        Not included: the guardrail audit trail and the trace's spans, both exported as spans.
        """
        schema = PauseSchema(schema_version=SCHEMA_VERSION, tree=self.tree, run=self.record)
        return schema.model_dump(mode="json")

    @classmethod
    def from_json(cls, root: Agent, state_json: dict[str, Any]) -> Pause:
        """Rebuild the pause `to_json` wrote, its agents found again from `root` by name."""
        try:
            schema = PauseSchema.model_validate(state_json, context={"agents": Agents(root)})
        except ValidationError as exc:
            raise UserError(f"invalid paused run: {exc}") from exc
        return cls(schema.run, schema.tree)


def paused(record: RunRecord, tree: RunTree, session: Session | None) -> RunState:
    """The `RunState` a root run hands back when it pauses, not yet kept (see `keep`)."""
    from runa.run_state import RunState

    return RunState(Pause(record, tree, session=session))


async def keep(state: RunState) -> None:
    """Save `state`'s pause as its starting agent's, under the new `run_id` it is resumed by."""
    await state._pause.keep()


def resuming(state: RunState) -> Pause:
    """The pause a resumed run continues from and claims: `state`'s, under the same `run_id`.

    Its record and audit trail are copies, so `state` still reads as it paused.
    """
    pause = state._pause
    return replace(pause, record=pause.record.resumed(), tree=pause.tree.snapshot())


def resume_session(
    state: RunState, passed: Session | str | None, *, context: Any, user_id: str | None
) -> Session | None:
    """The session to resume `state` in, once what was passed beside it is checked.

    What may be passed beside a `RunState`: see RUNA.md, Human Approval. With no session
    passed, the session is the run's own: the `Session` object it was given when resumed in the
    process that paused, otherwise its id resolved through `runa.db.session`.
    """
    for name, value in (("context", context), ("user_id", user_id)):
        if value is not None:
            raise UserError(
                f"a RunState already carries the {name} of the run it paused, so `{name}=` "
                f"would be ignored when resuming it: pass `{name}=` to the first run instead."
            )
    pause = state._pause
    session_id = pause.record.session_id
    if passed is None:
        if pause.session is not None or session_id is None:
            return pause.session
        from runa import db

        return db.session(session_id, pause.record.start.name)
    if isinstance(passed, str) or passed.session_id != session_id:
        given = passed if isinstance(passed, str) else passed.session_id
        raise UserError(
            f"a RunState already carries the session of the run it paused ({session_id!r}), "
            f"so `session={given!r}` would save its turn somewhere else: resume it without "
            "`session=`, or with the `Session` object of that same session."
        )
    return passed


__all__ = [
    "READABLE_SINCE",
    "SCHEMA_VERSION",
    "Answered",
    "CallStanding",
    "Interruption",
    "Pause",
    "PauseSchema",
    "RunRecord",
    "RunTree",
    "RunView",
    "Suspended",
    "Waiting",
    "keep",
    "paused",
    "resuming",
    "resume_session",
]
