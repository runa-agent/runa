"""Tests for how a finished trace reaches its exporters: off the loop for a run, inline for a block.

An exporter does blocking I/O by design (`TraceExporter`), so the question these own is not whether
`export` blocks but which thread it blocks. A run awaits `export_trace_off_loop`; a synchronous
`tracing.trace` block calls `export_trace` on the thread it already has.
"""

import asyncio
import threading
from typing import Any

from runa import Trace, tracing
from runa._types import ModelResponse, Usage
from runa.run import Run
from runa.run_internal.agent_shape import AgentShape
from runa.run_internal.run_config import RunConfig
from runa.run_internal.run_loop import _run_async
from runa.tracing import observe


class _TextModel:
    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:  # noqa: ANN002, ANN003
        return ModelResponse(
            output=[{"role": "assistant", "content": "ok", "tool_calls": None}],
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, requests=1),
        )


def _run() -> Any:
    shape = AgentShape(name="Researcher", instructions="You research topics.", model=_TextModel())
    return _run_async(shape, "hi", run_config=RunConfig(workflow_name="Researcher"))


def test_a_blocking_exporter_does_not_stall_the_event_loop() -> None:
    """A slow export is run off the loop, so everything else on that loop keeps going meanwhile.

    The exporter blocks until a coroutine on the run's own loop releases it, which is only
    possible if that loop is still free to run while the export is in flight. Exported inline --
    which is what a `runa serve` process used to do, once per finished run -- the releasing task
    never gets scheduled, the wait times out, and `blocked` says so.

    This is the whole finding: a Langfuse POST with OTel's retries behind it, or a
    `PostgresTraceStore.save` waiting on a struggling database, is one run's observability
    hiccup, and it must not be every concurrent run's latency.
    """
    exporting = threading.Event()
    released = threading.Event()
    blocked: list[bool] = []

    class _BlockingExporter:
        """Blocks in `export` until something else on the run's loop lets it go."""

        def export(self, trace: Trace) -> None:
            """Wait to be released, recording whether anyone managed to."""
            exporting.set()
            blocked.append(not released.wait(timeout=2.0))

    async def release_once_the_export_starts() -> None:
        while not exporting.is_set():
            await asyncio.sleep(0.005)
        released.set()

    async def exercise() -> Run:
        releaser = asyncio.ensure_future(release_once_the_export_starts())
        try:
            return await _run()
        finally:
            releaser.cancel()

    with observe.scoped(exporter=[_BlockingExporter()]):
        result = asyncio.run(exercise())

    assert blocked == [False], "the loop was stalled for the whole of the trace export"
    assert result.output == "ok"


def test_a_run_has_already_been_exported_by_the_time_it_returns() -> None:
    """The run awaits its own export, rather than leaving it behind as a background task.

    Off the loop is not fire-and-forget: a caller that reads `runa traces` (or a `run_sync`
    script that exits on the next line) finds the run it just made, with no flush to remember.
    That the exporter is this test's own also proves the `observe.scoped(...)` `ContextVar` is
    carried onto the export thread -- the process-wide default here is no exporter at all.
    """
    exported: list[Trace] = []

    class _Recording:
        def export(self, trace: Trace) -> None:
            """Keep every trace handed over."""
            exported.append(trace)

    with observe.scoped(exporter=[_Recording()]):
        result = asyncio.run(_run())
        assert len(exported) == 1

    assert exported[0].id == result.trace.id
    assert exported[0].end_time is not None


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
