"""Tests for how a finished trace reaches its exporters: queued for a run, inline for a block.

An exporter does blocking I/O by design (`TraceExporter`), so the question these own is not whether
`export` blocks but who waits for it. A run queues its trace with `export_trace_later` and
returns; `run_sync` and `flush_exports` wait for the queue; a synchronous `tracing.trace` block
calls `export_trace` on the thread it already has.
"""

import asyncio
import threading
from typing import Any

from runa import Agent, Trace, tracing
from runa._types import ModelResponse, Usage
from runa.run_internal.agent_shape import AgentShape
from runa.run_internal.run_config import RunConfig
from runa.run_internal.run_loop import _run_async
from runa.tracing import observe
from runa.tracing.config import flush_exports


class _TextModel:
    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:  # noqa: ANN002, ANN003
        return ModelResponse(
            output=[{"role": "assistant", "content": "ok", "tool_calls": None}],
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, requests=1),
        )


class _Recording:
    """Keeps every trace handed over."""

    def __init__(self) -> None:
        self.traces: list[Trace] = []

    def export(self, trace: Trace) -> None:
        """Keep `trace`."""
        self.traces.append(trace)


def _run() -> Any:
    shape = AgentShape(name="Researcher", instructions="You research topics.", model=_TextModel())
    return _run_async(shape, "hi", run_config=RunConfig(workflow_name="Researcher"))


def test_a_run_returns_without_waiting_for_a_slow_exporter() -> None:
    """A run hands its trace over and returns, so one slow backend is no response's latency.

    The exporter blocks until the test releases it, which the test only does once the run has
    returned: had the run waited for its export, it would never have come back. `flush_exports`
    then finds the trace written, with the `observe.scoped(...)` `ContextVar` carried onto the
    export thread -- the process-wide default here is no exporter at all.
    """
    released = threading.Event()
    exported: list[Trace] = []

    class _Blocking:
        def export(self, trace: Trace) -> None:
            """Wait to be released, then keep `trace`."""
            assert released.wait(timeout=2.0), "the run waited for its own export"
            exported.append(trace)

    with observe.scoped(exporter=[_Blocking()]):
        result = asyncio.run(_run())

    assert result.output == "ok"
    assert exported == []
    released.set()
    flush_exports()
    assert [t.id for t in exported] == [result.trace.id]
    assert exported[0].end_time is not None


def test_run_sync_has_written_its_trace_by_the_time_it_returns() -> None:
    """A script or a `runa chat` turn reads its trace back on the next line, with no flush."""
    recording = _Recording()

    class Researcher(Agent):
        name = "Researcher"
        instructions = "You research topics."
        model = _TextModel()

    with observe.scoped(exporter=[recording]):
        result = Researcher().run_sync("hi")
        assert [t.id for t in recording.traces] == [result.trace.id]


def test_a_manual_trace_block_exports_on_its_own_thread() -> None:
    """`with tracing.trace(...)` is synchronous, so its export is too: nothing to get off."""
    exported: list[tuple[Trace, int]] = []

    class _Recording:
        def export(self, trace: Trace) -> None:
            """Keep every trace, with the thread it arrived on."""
            exported.append((trace, threading.get_ident()))

    with observe.scoped(exporter=[_Recording()]):
        with tracing.trace("nightly-batch") as t, tracing.span("step-1", t):
            pass
        assert len(exported) == 1

    trace, thread = exported[0]
    assert trace.id == t.id
    assert trace.end_time is not None
    assert thread == threading.get_ident()
