"""Tests for `runa.tracing.langfuse.LangfuseExporter`.

No live Langfuse project needed: `_otlp` (the `OTLPSpanExporter`) is swapped for a fake that
just records the `ReadableSpan`s it's handed, so these tests check the `Trace`/`Span` -> OTel
conversion without any network call.
"""

from typing import Any

import pytest
from opentelemetry.trace import StatusCode

from runa.exceptions import UserError
from runa.tracing.langfuse import LangfuseExporter
from runa.tracing.spans import Span
from runa.tracing.traces import Trace


class _FakeOTLPExporter:
    def __init__(self) -> None:
        self.exported: list[Any] = []

    def export(self, spans: Any) -> None:
        self.exported = list(spans)


def _exporter() -> tuple[LangfuseExporter, _FakeOTLPExporter]:
    exporter = LangfuseExporter(public_key="pk-lf-test", secret_key="sk-lf-test")
    fake = _FakeOTLPExporter()
    exporter._otlp = fake  # type: ignore[assignment]  # noqa: SLF001 -- fake network client
    return exporter, fake


def _trace(*spans: Span) -> Trace:
    return Trace(id="a" * 32, name="TestAgent", start_time=0.0, end_time=2.0, spans=list(spans))


def test_root_span_becomes_the_otel_trace_id() -> None:
    """A span with no `parent_id` gets its own id used as the whole OTel trace's id."""
    root = Span(
        id="b" * 32,
        trace_id="a" * 32,
        parent_id=None,
        name="TestAgent",
        type="agent",
        start_time=0.0,
        end_time=2.0,
    )
    exporter, fake = _exporter()

    exporter.export(_trace(root))

    (otel_root,) = fake.exported
    assert otel_root.get_span_context().trace_id == int("a" * 32, 16)
    assert otel_root.get_span_context().span_id == int(("b" * 32)[:16], 16)
    assert otel_root.parent is None
    assert otel_root.attributes["langfuse.observation.type"] == "agent"
    assert otel_root.attributes["langfuse.trace.name"] == "TestAgent"


def test_child_span_shares_trace_id_and_points_at_its_parent() -> None:
    """A span with a `parent_id` stays in the same OTel trace and links to its parent's span id."""
    root = Span(
        id="b" * 32,
        trace_id="a" * 32,
        parent_id=None,
        name="TestAgent",
        type="agent",
        start_time=0.0,
        end_time=2.0,
    )
    child = Span(
        id="c" * 32,
        trace_id="a" * 32,
        parent_id=root.id,
        name="my_tool",
        type="tool",
        start_time=0.5,
        end_time=1.0,
    )
    exporter, fake = _exporter()

    exporter.export(_trace(root, child))

    _, otel_child = fake.exported
    assert otel_child.get_span_context().trace_id == int("a" * 32, 16)
    assert otel_child.parent.span_id == int(("b" * 32)[:16], 16)
    assert otel_child.start_time == 500_000_000
    assert otel_child.end_time == 1_000_000_000


def test_llm_span_carries_model_name_and_usage() -> None:
    """An `llm` span's name becomes the Langfuse generation's model, usage gets forwarded."""
    llm = Span(
        id="c" * 32,
        trace_id="a" * 32,
        parent_id=None,
        name="claude-sonnet-5",
        type="llm",
        start_time=0.0,
        end_time=1.0,
        output={"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
    )
    exporter, fake = _exporter()

    exporter.export(_trace(llm))

    (otel_llm,) = fake.exported
    assert otel_llm.attributes["langfuse.observation.type"] == "generation"
    assert otel_llm.attributes["langfuse.observation.model.name"] == "claude-sonnet-5"
    assert otel_llm.attributes["langfuse.observation.usage_details"] == (
        '{"input": 10, "output": 5, "total": 15}'
    )


def test_error_span_sets_otel_error_status_and_langfuse_level() -> None:
    """A failed span's error message becomes both the OTel status and the Langfuse level."""
    tool_span = Span(
        id="c" * 32,
        trace_id="a" * 32,
        parent_id=None,
        name="my_tool",
        type="tool",
        start_time=0.0,
        end_time=1.0,
        status="error",
        error="boom",
    )
    exporter, fake = _exporter()

    exporter.export(_trace(tool_span))

    (otel_span,) = fake.exported
    assert otel_span.status.status_code == StatusCode.ERROR
    assert otel_span.attributes["langfuse.observation.level"] == "ERROR"
    assert otel_span.attributes["langfuse.observation.status_message"] == "boom"


def test_non_string_input_and_output_are_json_encoded() -> None:
    """Dict/list input and output are JSON-encoded, since OTel attributes can't hold them raw."""
    span = Span(
        id="c" * 32,
        trace_id="a" * 32,
        parent_id=None,
        name="my_tool",
        type="tool",
        start_time=0.0,
        end_time=1.0,
        input={"x": 1},
        output=[1, 2, 3],
    )
    exporter, fake = _exporter()

    exporter.export(_trace(span))

    (otel_span,) = fake.exported
    assert otel_span.attributes["langfuse.observation.input"] == '{"x": 1}'
    assert otel_span.attributes["langfuse.observation.output"] == "[1, 2, 3]"


def test_missing_credentials_raise_a_user_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """No `public_key`/`secret_key` arg and no env var is a clear error, not a later crash."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    with pytest.raises(UserError, match="LANGFUSE_PUBLIC_KEY"):
        LangfuseExporter()


def test_credentials_fall_back_to_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Credentials/host fall back to `LANGFUSE_PUBLIC_KEY`/`_SECRET_KEY`/`_HOST` when unset."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-env")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-env")
    monkeypatch.setenv("LANGFUSE_HOST", "https://self-hosted.example.com")

    exporter = LangfuseExporter()

    assert exporter._otlp._endpoint == "https://self-hosted.example.com/api/public/otel/v1/traces"  # noqa: SLF001


def test_explicit_arguments_win_over_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing `public_key`/`secret_key`/`host` directly overrides whatever the env vars hold."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-env")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-env")

    exporter = LangfuseExporter(
        public_key="pk-lf-arg", secret_key="sk-lf-arg", host="https://explicit.example.com"
    )

    assert exporter._otlp._endpoint == "https://explicit.example.com/api/public/otel/v1/traces"  # noqa: SLF001
