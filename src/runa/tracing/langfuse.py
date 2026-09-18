"""tracing/langfuse.py: `LangfuseExporter`, a `TraceExporter` that forwards to Langfuse.

Optional -- the `runa[langfuse]` extra, not a core dependency; nothing outside this module
imports it. Same shape as `db/redis.py`: a hard top-level import of the optional package, no
try/except fallback, since nothing outside this module needs it importable.

Runa's tracing records a `Trace`'s full span tree internally and hands the finished tree to
`TraceExporter.export` once, at the end of a run (`run_internal/spans.py`), unlike live
instrumentation that opens/closes a span as each operation happens. Langfuse's own Python SDK
(v3) only lets an observation's `end_time` be overridden, not its `start_time` -- every
observation it creates starts "now". Replaying an already-finished `Trace` through it would
therefore put every span's start at export time with an end_time already in the past: a negative
duration. Building plain OpenTelemetry spans with explicit start/end timestamps and posting them
straight to Langfuse's OTLP endpoint sidesteps that; it's also the route Langfuse's own
maintainers point to for backdated/nested traces like this one (their SDK-level `start_span`/
`start_observation` calls have no `start_time` parameter at all):
https://langfuse.com/integrations/native/opentelemetry
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, cast

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.id_generator import IdGenerator
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    SpanKind,
    TraceFlags,
    set_span_in_context,
)
from opentelemetry.trace.status import Status, StatusCode

from runa.exceptions import UserError
from runa.tracing.spans import Span
from runa.tracing.traces import Trace

_AS_TYPE = {
    "agent": "agent",
    "llm": "generation",
    "tool": "tool",
    "retrieval": "retriever",
    "handoff": "span",
    "guardrail": "guardrail",
    "custom": "span",
}

_SAMPLED = TraceFlags(TraceFlags.SAMPLED)
_NANOS_PER_SECOND = 1_000_000_000


def _trace_id(trace_id: str) -> int:
    """Reuse `trace_id`'s 32 hex chars (`uuid4().hex`) as-is: already a valid 128-bit OTel id."""
    return int(trace_id, 16) or 1


def _span_id(span_id: str) -> int:
    """Truncate `span_id`'s 32 hex chars down to the 64 bits an OTel span id needs."""
    return int(span_id[:16], 16) or 1


class _FixedIdGenerator(IdGenerator):
    """Hands a `TracerProvider` one pre-chosen id per call, instead of a random one.

    `Tracer.start_span` takes a `context` for the *parent* span but never lets a caller choose
    the *new* span's own id, or a root span's trace id -- those always come from the provider's
    `IdGenerator`. Setting `next_span_id`/`next_trace_id` right before each `start_span` call is
    how `LangfuseExporter` makes every OTel span's id match its `Span.id` exactly.
    """

    def __init__(self) -> None:
        """Start with no id queued; `LangfuseExporter` sets one before every `start_span` call."""
        self.next_span_id = 0
        self.next_trace_id = 0

    def generate_span_id(self) -> int:
        """Return the id `LangfuseExporter` queued for the span currently being started."""
        return self.next_span_id

    def generate_trace_id(self) -> int:
        """Return the id `LangfuseExporter` queued for the root span currently being started."""
        return self.next_trace_id


def _as_json(value: Any) -> str | None:
    """Render `value` as an OTel-attribute-safe string; OTel attributes can't hold dicts/lists."""
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _usage_details(output: Any) -> str | None:
    usage = output.get("usage") if isinstance(output, dict) else None
    if not isinstance(usage, dict):
        return None
    details = {
        "input": usage.get("input_tokens"),
        "output": usage.get("output_tokens"),
        "total": usage.get("total_tokens"),
    }
    return json.dumps({k: v for k, v in details.items() if v is not None})


class LangfuseExporter:
    """`TraceExporter` that forwards every finished `Trace` to Langfuse, as raw OTLP spans.

    Needs the `runa[langfuse]` extra (`opentelemetry-sdk` and
    `opentelemetry-exporter-otlp-proto-http`), not the `langfuse` package -- see the module
    docstring for why.
    """

    def __init__(
        self,
        *,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
    ) -> None:
        """Configure which Langfuse project to export to.

        `public_key`/`secret_key`/`host` fall back to the `LANGFUSE_PUBLIC_KEY`/
        `LANGFUSE_SECRET_KEY`/`LANGFUSE_HOST` env vars, same convention as `ModelProvider`'s
        `OPENAI_API_KEY` and friends -- pass them directly only to override that. `host` is the
        Langfuse deployment's base URL (a region's cloud endpoint, or a self-hosted instance);
        the OTLP traces path is appended automatically.
        """
        public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        if public_key is None or secret_key is None:
            raise UserError(
                "LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are not set. Set them, or pass "
                "public_key/secret_key directly, to use LangfuseExporter."
            )
        host = host or os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self._id_generator = _FixedIdGenerator()
        self._tracer = TracerProvider(
            resource=Resource.create({"service.name": "runa"}), id_generator=self._id_generator
        ).get_tracer("runa")
        self._otlp = OTLPSpanExporter(
            endpoint=f"{host}/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {auth}"},
        )

    def export(self, trace: Trace) -> None:
        """Convert `trace` and every span in it to OTel spans, and export them to Langfuse."""
        trace_id = _trace_id(trace.id)
        otel_spans = [self._otel_span(trace, span, trace_id) for span in trace.spans]
        self._otlp.export(otel_spans)

    def _otel_span(self, trace: Trace, span: Span, trace_id: int) -> ReadableSpan:
        context: Any = None
        if span.parent_id is not None:
            parent = SpanContext(
                trace_id=trace_id,
                span_id=_span_id(span.parent_id),
                is_remote=True,
                trace_flags=_SAMPLED,
            )
            context = set_span_in_context(NonRecordingSpan(parent))
        else:
            self._id_generator.next_trace_id = trace_id
        self._id_generator.next_span_id = _span_id(span.id)

        attributes: dict[str, Any] = {"langfuse.observation.type": _AS_TYPE.get(span.type, "span")}
        if span.parent_id is None:
            attributes["langfuse.trace.name"] = trace.name
        input_json = _as_json(span.input)
        if input_json is not None:
            attributes["langfuse.observation.input"] = input_json
        output_json = _as_json(span.output)
        if output_json is not None:
            attributes["langfuse.observation.output"] = output_json
        if span.type == "llm":
            attributes["langfuse.observation.model.name"] = span.name
            usage_json = _usage_details(span.output)
            if usage_json is not None:
                attributes["langfuse.observation.usage_details"] = usage_json
        if span.status == "error" and span.error:
            attributes["langfuse.observation.level"] = "ERROR"
            attributes["langfuse.observation.status_message"] = span.error

        otel_span = self._tracer.start_span(
            name=span.name,
            context=context,
            kind=SpanKind.INTERNAL,
            attributes=attributes,
            start_time=round(span.start_time * _NANOS_PER_SECOND),
        )
        if span.status == "error":
            otel_span.set_status(Status(StatusCode.ERROR, span.error))
        end_time = span.end_time if span.end_time is not None else span.start_time
        otel_span.end(end_time=round(end_time * _NANOS_PER_SECOND))
        # `Tracer.start_span`'s return type is the API-level `Span`; the SDK `Tracer` we built
        # this from always returns its own `ReadableSpan`-implementing `Span` in practice.
        return cast(ReadableSpan, otel_span)


__all__ = ["LangfuseExporter"]
