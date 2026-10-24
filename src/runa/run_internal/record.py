"""record.py: `RunRecord`, what a run is as it would be resumed, and `Pause`, one kept in `runa.db`.

A run is one `RunRecord`, the facts about it that outlive the call running it, held by the turn
loop as it goes and serialized as is. What every run in one tree shares, the root's and each
delegate's alike, is one `RunTree`. A fact a run starts carrying is a field on one of the two, and
nowhere else: it reaches a paused state, its stored row and a resumed run with no further edit.

A pause is kept in `runa.db.paused_runs()` the moment it happens, under a `run_id`, and resuming
it claims that row (see `PausedRunStore.claim`). The blob a row holds is this module's own
format, never handed out, so no copy of a pause exists that could be resumed a second time.

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
)

from runa._items import ConversationItem
from runa._types import RunContext
from runa.approval import ApprovalLedger
from runa.exceptions import UserError
from runa.guardrail import GuardrailResults
from runa.session import Session
from runa.tool import ToolCall
from runa.tracing.traces import Trace
from runa.transcript import Transcript

if TYPE_CHECKING:
    from runa.agent import Agent
    from runa.run_state import RunState

SCHEMA_VERSION = 9


def _find_agent_by_name(root: Agent, name: str) -> Agent:
    """`root`, or the agent reachable from it, that is named `name`.

    Deserialization carries an agent as a name string, so resuming has to find the instance
    again: a handoff may have switched the current agent before the pause, or the paused call
    may belong to a delegate, so the match isn't necessarily `root` itself. `reachable` is the
    same walk `Agent.graph` draws, imported here because only a load walks the graph.
    """
    from runa.run_internal.agent_shape import reachable

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


class RunRecord(BaseModel):
    """One run, as it would be resumed: everything it is that outlives the call running it.

    The turn loop works on one directly; a `Pause` holds it, and so does a paused delegate's
    entry in its caller's `delegates`. Every field is data or written as data (an agent by its
    name), so all of it is serialized. What only the call running it holds -- its hooks, its
    `Session` object, its stream, its open span -- isn't here, and isn't resumed.

    * `start`: the agent the run began with. The run as a whole is its: the turn, token and time
      ceilings, the context sources, the compaction of history, and the key a pause is kept under.
    * `agent`: the agent running now, the same as `start` until a handoff.
    * `run_context`: what user code is told, its `context`, `user_id` and usage so far.
    * `tool_calls`: every tool call the run answered, across its resumes, in the order they
      finished. A fact about the run, not its conversation: a run that fails still reports it.
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
    tool_calls: list[ToolCall] = Field(default_factory=list)
    session_id: str | None = None
    turns: int = 0
    elapsed: float = 0.0
    pending: list[Interruption] = Field(default_factory=list)
    ready_results: list[ConversationItem] = Field(default_factory=list)
    delegates: dict[str, RunRecord] = Field(default_factory=dict)

    @property
    def paused_message(self) -> ConversationItem:
        """The assistant message whose calls paused the run: the transcript's last, always."""
        return self.transcript.items[-1]

    def resumed(self) -> RunRecord:
        """A copy for a resumed run to work on, so this one still reads as it paused."""
        usage = copy.deepcopy(self.run_context.usage)
        return self.model_copy(
            update={
                "transcript": self.transcript.copy(),
                "tool_calls": list(self.tool_calls),
                "run_context": RunContext(
                    self.run_context.context, self.run_context.user_id, usage
                ),
                "delegates": dict(self.delegates),
            }
        )


class RunTree(BaseModel):
    """What every run in one tree shares by reference: the root run's and each delegate's.

    * `trace`: the one trace every run's spans go in.
    * `approvals`: the ledger a decision is written to, and each run reads its calls' from.
    * `guardrail_results`: the audit trail of every guardrail any of them ran. Not serialized:
      each result is already a span in the trace.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    trace: _TraceRef
    approvals: ApprovalLedger = Field(default_factory=ApprovalLedger)
    guardrail_results: GuardrailResults = Field(default_factory=GuardrailResults, exclude=True)

    def snapshot(self) -> RunTree:
        """The same trace and ledger, with a copy of the audit trail as it stands now."""
        return self.model_copy(update={"guardrail_results": self.guardrail_results.snapshot()})


class PauseSchema(BaseModel):
    """The JSON envelope a `Pause` is kept as: its record and its tree.

    Not serialized: `run_id`, the key the row is kept under beside the starting agent's name, and
    the run's `Session` object, whose id the record carries.
    """

    schema_version: Literal[9]
    tree: RunTree
    run: RunRecord


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
            schema = PauseSchema.model_validate(state_json, context={"root": root})
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

    Every rule about the arguments a `RunState` is run with is here. It carries the `context`,
    `user_id` and session of the run it paused, so any of them passed beside it is a `UserError`
    rather than dropped or the turn saved elsewhere, with one exception: a `Session` object for
    the run's own `session_id`, which names the store a session lives in when `runa.db.session`
    doesn't reach it, an object a pause loaded in another process doesn't have. With none
    passed, the session is the run's own.
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
    "SCHEMA_VERSION",
    "Interruption",
    "Pause",
    "PauseSchema",
    "RunRecord",
    "RunTree",
    "keep",
    "paused",
    "resuming",
    "resume_session",
]
