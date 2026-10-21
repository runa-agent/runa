"""Tests for the one promise every `Run` rests on: errors never raise out of a run.

`Run.status` exists so that no caller has to write a `try` around `agent.run(...)`, and
`runa serve` is written on that basis. What makes the promise worth anything is that it holds for
failures Runa never anticipated, not only for the `RunaError` subclasses it declared: half of what
a run calls is application code reached through a sanctioned extension point -- a `Hooks` method,
a custom `Session`, a `Compactor`, a memory or knowledge store. A locked SQLite file is not an
exotic trigger (it is the default backend, under the concurrency `runa serve` invites), and when
one of these raised, the run both escaped the contract and skipped the close-out that exports its
trace: the one run worth looking at was the one that left nothing behind.

So the boundary decides, not the exception's type (`run_internal/run_loop._guarded`, `Agent.run`).
`tests/test_limits.py` owns the same contract for the three ceilings, which are `RunaError`s
already; the ceilings are not what this file is about.
"""

import asyncio
import logging
from typing import Any

import pytest

from runa import Agent, db
from runa._types import ModelResponse, Usage
from runa.exceptions import OperatorError, UnexpectedError
from runa.lifecycle import Hooks
from runa.run import Run
from runa.session.database import DatabaseSession
from runa.session.store import MESSAGES, SESSIONS
from runa.tracing import Trace, observe


def _final_message(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text, "tool_calls": None}


def _tool_call_message(name: str, arguments: str, call_id: str = "call_1") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


class _ScriptedModel:
    """A `Model` stand-in answering with pre-scripted messages, through the real turn loop."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = list(messages)

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:  # noqa: ANN002, ANN003
        return ModelResponse(
            output=[self._messages.pop(0)],
            usage=Usage(input_tokens=3, output_tokens=4, total_tokens=7, requests=1),
        )


def _agent(**overrides: Any) -> Agent:
    """A one-answer agent, on a scripted model, built fresh so nothing leaks between tests."""

    class Simple(Agent):
        name = "Simple"
        instructions = "Answer."
        model = _ScriptedModel([_final_message("ok")])

    return Simple(**overrides)


class _Exports:
    """A `TraceExporter` that keeps what it was handed, to count a run's traces and read them."""

    def __init__(self) -> None:
        self.traces: list[Trace] = []

    def export(self, trace: Trace) -> None:
        self.traces.append(trace)


def _agent_span(trace: Trace | None) -> Any:
    """The one `"agent"` span of a run's trace, narrowed away from `Run.trace`'s `| None`."""
    assert trace is not None
    (span,) = [s for s in trace.spans if s.type == "agent"]
    return span


@pytest.fixture(autouse=True)
def _in_memory_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every session these tests open in the `memory://` database, never in the cwd's file."""
    monkeypatch.setenv("RUNA_DATABASE_URL", "memory://")


class _StoredSession(DatabaseSession):
    """A session in this process's `memory://` database, for a subclass to break one method of."""

    def __init__(self, session_id: str) -> None:
        super().__init__(session_id, db.database(SESSIONS, MESSAGES))


class _UnreadableSession(_StoredSession):
    """A `Session` whose backend doesn't answer: a locked SQLite file, a dropped connection."""

    async def get_items(self) -> list[Any]:
        raise OSError("database is locked")


class _UnwritableSession(_StoredSession):
    """A `Session` that reads fine and fails on the write, after the turn already answered."""

    async def add_items(self, items: list[Any]) -> None:
        raise OSError("disk I/O error")


def test_a_session_that_cannot_be_read_is_an_error_run() -> None:
    """A `Session` is the application's to implement, and its failure is this run's to report."""
    run = _agent().run_sync("hi", session=_UnreadableSession("s1"))

    assert run.status == "error"
    assert run.error == "OSError: database is locked"
    assert run.output is None


def test_a_session_that_cannot_be_written_is_an_error_run() -> None:
    """Saving the turn happens after the model answered, and is no less part of the run.

    The answer is lost either way -- it was never persisted -- so the caller is told, rather than
    being handed an exception where every other failure is a `Run`.
    """
    run = _agent().run_sync("hi", session=_UnwritableSession("s2"))

    assert run.status == "error"
    assert run.error == "OSError: disk I/O error"


def test_a_hook_with_a_bug_in_it_is_an_error_run() -> None:
    """`Hooks` are application code; a stray `AttributeError` in one is reported, not raised."""

    class _BrokenHooks(Hooks[Any]):
        async def on_agent_start(self, context: Any, agent: Any) -> None:
            raise AttributeError("'NoneType' object has no attribute 'name'")

    run = _agent().run_sync("hi", hooks=_BrokenHooks())

    assert run.status == "error"
    assert run.error is not None and run.error.startswith("AttributeError:")


def test_a_hook_that_fails_after_the_answer_is_an_error_run() -> None:
    """`on_agent_end` fires from `_finish`, which the boundary encloses like everything else."""

    class _BrokenHooks(Hooks[Any]):
        async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
            raise RuntimeError("the metrics sink is down")

    run = _agent().run_sync("hi", hooks=_BrokenHooks())

    assert run.status == "error"
    assert run.error == "RuntimeError: the metrics sink is down"


def test_a_compactor_that_raises_is_an_error_run() -> None:
    """`compact` is a plain function the app supplies, called mid-run and again at the end."""

    def _broken_compactor(items: list[Any], tokens: int) -> list[Any]:
        raise ZeroDivisionError("division by zero")

    run = _agent(compact=_broken_compactor).run_sync("hi")

    assert run.status == "error"
    assert run.error == "ZeroDivisionError: division by zero"


def test_an_unexpected_failure_still_records_usage_and_exports_its_trace() -> None:
    """The run that failed is the one worth inspecting, so it leaves the same marks as any other.

    One exported trace (not two, and not none), its agent span closed and marked with the
    failure, and the tokens the turn did spend still accounted for on the agent.
    """
    exports = _Exports()
    agent = _agent()

    with observe.scoped(exporter=[exports]):
        run = agent.run_sync("hi", session=_UnwritableSession("s3"))

    assert run.status == "error"
    assert len(exports.traces) == 1
    assert exports.traces[0].end_time is not None
    assert _agent_span(run.trace).status == "error"
    assert _agent_span(run.trace).error == "OSError: disk I/O error"
    assert run.usage.total_tokens == 7
    assert agent.last_usage.total_tokens == 7


def test_an_unexpected_failure_logs_its_traceback(caplog: pytest.LogCaptureFixture) -> None:
    """Coming back as a result must not cost the traceback: it goes to the log instead.

    This is the trade the contract makes -- a bug in an app's own hook is a string on `Run.error`
    rather than a traceback at the call site -- so the traceback has to be reachable somewhere.
    """

    class _BrokenHooks(Hooks[Any]):
        async def on_llm_start(self, context: Any, agent: Any, system: Any, items: Any) -> None:
            raise AttributeError("no attribute 'provider'")

    with caplog.at_level(logging.ERROR, logger="runa"):
        run = _agent().run_sync("hi", hooks=_BrokenHooks())

    assert run.status == "error"
    (record,) = [r for r in caplog.records if "unhandled" in r.getMessage()]
    assert "Simple" in record.getMessage()
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], AttributeError)


def test_an_unexpected_failure_keeps_the_original_as_its_cause() -> None:
    """`UnexpectedError` names the exception it replaced and carries it, for code that looks.

    Nothing in Runa branches on it; a caller debugging one wants the type back, and `run.error`
    names it for the same reason.
    """

    class _BrokenHooks(Hooks[Any]):
        async def on_agent_start(self, context: Any, agent: Any) -> None:
            raise KeyError("user_id")

    wrapped = UnexpectedError(KeyError("user_id"))

    assert str(wrapped) == "KeyError: 'user_id'"
    assert str(UnexpectedError(RuntimeError())) == "RuntimeError"
    assert _agent().run_sync("hi", hooks=_BrokenHooks()).error == "KeyError: 'user_id'"


def test_an_operator_error_is_not_reported_as_a_run_failure() -> None:
    """The one exception a run lets through: `OperatorError` is not a run's failure to own.

    It is a sibling of `RunaError` exactly so this stays true (see `runa.exceptions`) -- an
    unparseable `RUNA_DATABASE_URL` is the operator's to fix, and `runa chat`/`runa ui` print it
    as one line rather than a run that mysteriously errored. The trace is still closed out.
    """

    class _MisconfiguredSession(_StoredSession):
        async def get_items(self) -> list[Any]:
            raise OperatorError("could not parse RUNA_DATABASE_URL")

    exports = _Exports()

    with observe.scoped(exporter=[exports]), pytest.raises(OperatorError, match="RUNA_DATABASE"):
        _agent().run_sync("hi", session=_MisconfiguredSession("s4"))

    assert len(exports.traces) == 1
    assert _agent_span(exports.traces[0]).status == "error"


def test_cancelling_a_run_escapes_the_broader_catch() -> None:
    """`CancelledError` is a `BaseException`, so catching `Exception` leaves it alone.

    The broad catch is what makes a hook's bug a result; a caller that cancels a run still gets
    the cancellation (`tests/test_limits.py` owns that promise), and nothing is exported for a
    run that was told to stop -- it has no outcome to record.
    """

    class _SlowModel:
        async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:  # noqa: ANN002, ANN003
            await asyncio.sleep(5)
            return ModelResponse(output=[_final_message("too late")], usage=Usage())

    exports = _Exports()
    agent = _agent(model=_SlowModel())

    async def cancel_mid_run() -> None:
        task = asyncio.create_task(agent.run("hi"))
        await asyncio.sleep(0.01)
        task.cancel()
        await task

    with observe.scoped(exporter=[exports]), pytest.raises(asyncio.CancelledError):
        asyncio.run(cancel_mid_run())

    assert exports.traces == []


def test_a_streamed_run_reports_an_unexpected_failure_on_its_run() -> None:
    """`run_streamed` ends with the same `Run` `run` would have returned, failures included."""

    class _BrokenHooks(Hooks[Any]):
        async def on_agent_start(self, context: Any, agent: Any) -> None:
            raise RuntimeError("the audit log is down")

    async def consume() -> Run | None:
        stream = _agent().run_streamed("hi", hooks=_BrokenHooks())
        async for _event in stream:
            pass
        return stream.run

    run = asyncio.run(consume())

    assert run is not None
    assert run.status == "error"
    assert run.error == "RuntimeError: the audit log is down"


def test_a_streamed_operator_error_is_not_reported_as_a_run_failure() -> None:
    """The stream makes the same exception for the same reason `run` does, not a quieter one."""

    class _MisconfiguredSession(_StoredSession):
        async def get_items(self) -> list[Any]:
            raise OperatorError("could not parse RUNA_DATABASE_URL")

    async def consume() -> None:
        stream = _agent().run_streamed("hi", session=_MisconfiguredSession("s5"))
        async for _event in stream:
            pass

    with pytest.raises(OperatorError, match="RUNA_DATABASE"):
        asyncio.run(consume())


def test_a_resumed_run_reports_an_unexpected_failure_too() -> None:
    """Resuming is the same boundary: a run that paused for approval fails the same way.

    `_resume` builds its own `_Run` and dispatches its own `on_agent_start`, so a contract kept
    only on the fresh-run path would break for exactly the runs that already waited on a human.
    """
    from runa.tool import tool

    @tool(needs_approval=True)
    def issue_refund(amount: float) -> str:
        """Refund an order."""
        return f"refunded {amount}"

    class _BrokenHooks(Hooks[Any]):
        async def on_agent_start(self, context: Any, agent: Any) -> None:
            raise RuntimeError("the approval log is down")

    class Support(Agent):
        name = "Support"
        instructions = "Refund when asked."
        tools = [issue_refund]
        model = _ScriptedModel(
            [_tool_call_message("issue_refund", '{"amount": 75}'), _final_message("refunded")]
        )

    agent = Support()
    paused = agent.run_sync("refund $75")
    state = paused.to_state()
    for item in paused.interruptions:
        state.approve(item)

    resumed = agent.run_sync(state, hooks=_BrokenHooks())

    assert paused.status == "paused"
    assert resumed.status == "error"
    assert resumed.error == "RuntimeError: the approval log is down"
    # A resume continues the paused run's trace, so its agent span is the second one on it.
    assert resumed.trace is not None
    assert [s.status for s in resumed.trace.spans if s.type == "agent"] == ["ok", "error"]
