"""Tests for tracing's fail-open guarantee: a broken exporter must never break an agent run."""

from typing import Any

from helpers import scripted_agent

from runa import Trace
from runa._types import ModelResponse, Usage
from runa.tracing import add_exporter


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

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN002, ANN003
        """Answers unstreamed only."""
        raise NotImplementedError


def _agent() -> Any:
    return scripted_agent(_TextModel(), name="Researcher")


def test_agent_run_sync_completes_even_when_the_exporter_fails() -> None:
    """A run still returns a completed result when the tracing exporter raises."""
    add_exporter(_BrokenExporter())
    result = _agent().run_sync("hi")

    assert result.output == "ok"
    assert result.trace is not None
    assert result.trace.spans
