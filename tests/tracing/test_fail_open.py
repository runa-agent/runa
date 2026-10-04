"""Tests for tracing's fail-open guarantee: a broken exporter must never break an agent run."""

import asyncio
from types import SimpleNamespace
from typing import Any

from runa import Trace
from runa._types import ModelResponse, ModelSettings, Usage
from runa.run_config import RunConfig
from runa.run_internal.run_loop import _run_async
from runa.tracing import observe


class _BrokenExporter:
    """An exporter that always raises, to prove tracing failures don't propagate."""

    def export(self, trace: Trace) -> None:
        """Raise unconditionally."""
        raise RuntimeError("storage is down")


class _TextModel:
    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:  # noqa: ANN002, ANN003
        return ModelResponse(
            output=[{"role": "assistant", "content": "ok", "tool_calls": None}],
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, requests=1),
        )


def _agent() -> Any:
    return SimpleNamespace(
        name="Researcher",
        instructions="You research topics.",
        model=_TextModel(),
        tools=[],
        handoffs=[],
        input_guardrails=[],
        output_guardrails=[],
        output_type=None,
        model_settings=ModelSettings(),
    )


def test_agent_run_sync_completes_even_when_the_exporter_fails() -> None:
    """A run still returns a completed result when the tracing exporter raises."""
    with observe(exporter=_BrokenExporter()):
        result = asyncio.run(
            _run_async(_agent(), "hi", run_config=RunConfig(workflow_name="Researcher"))
        )

    assert result.output == "ok"
    assert result.trace is not None
    assert result.trace.spans
