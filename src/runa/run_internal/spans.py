"""spans.py: tracing span helpers shared by the turn loop, guardrails, and tool execution.

A run's spans are also where its model calls are charged: inside a span the run opens, a
`metered` call (the loop's own, a context source's, an embedding request anywhere below) is an
`"llm"` span under it, counted in the run's usage. The innermost open span pays.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from runa._metering import Charge, charging
from runa._types import Usage
from runa.tracing import config as tracing_config
from runa.tracing.spans import Span
from runa.tracing.traces import Trace
from runa.tracing.util import gen_span_id


@dataclass(frozen=True)
class _Spans:
    """Where a span opened right here belongs: which trace, and under which parent.

    One value instead of a `(trace, parent_id)` pair beside every step's own arguments. A step that
    opens spans inside its own (a tool call, around its guardrails) hands on `under(span)` rather
    than remembering to swap one of two parameters.

    `usage` is the run's the spans belong to. With it, a scope is a `Meter`: each `span` block
    charges the `metered` calls inside it to `usage`, under that span. Without it (a scope that
    only opens an agent span), a span charges nothing of its own.
    """

    trace: Trace
    parent_id: str | None = None
    usage: Usage | None = None

    def open(self, name: str, span_type: Any, *, input: Any = None) -> Span:
        """Start a span under this scope's parent, recorded on the trace straight away."""
        span = Span(
            id=gen_span_id(),
            trace_id=self.trace.id,
            parent_id=self.parent_id,
            name=name,
            type=span_type,
            start_time=time.time(),
        )
        policy = tracing_config.policy()
        if input is not None and policy.capture_inputs:
            span.input = tracing_config.apply_policy(input, max_bytes=policy.max_input_bytes)
        self.trace.spans.append(span)
        return span

    @contextmanager
    def span(self, name: str, span_type: Any, *, input: Any = None) -> Iterator[Span]:
        """Open a span for the `with` block and close it on the way out, however the block leaves.

        The block sets `span.output`, or `span.error` for a failure it handles itself; one that
        escapes the block is recorded as the span's error before it propagates. So a step that
        raises can't leave its span open, and the failed span is the one worth inspecting.
        `open`/`_close_span` remain for a span whose output comes from later code: the agent's.

        The `metered` calls the block makes, in this task or one it starts, are charged under the
        span, unless a span opened inside it takes them.
        """
        with self._span(name, span_type, input=input) as span:
            if self.usage is None:
                yield span
            else:
                with charging(self.under(span)):
                    yield span

    @contextmanager
    def call(self, name: str, *, input: Any = None) -> Iterator[Charge]:
        """An `"llm"` span for one call to the model `name`, and what to charge its `Usage` to.

        Changes no charging of its own: the block may be an async generator's, which can't
        hold one across a `yield`.
        """
        usage = self.usage
        assert usage is not None, "only a run's spans are charged to"
        with self._span(name, "llm", input=input) as span:

            def charge(spent: Usage) -> None:
                usage.add(spent)
                span.output = {"usage": spent.__dict__}

            yield charge

    @contextmanager
    def _span(self, name: str, span_type: Any, *, input: Any = None) -> Iterator[Span]:
        span = self.open(name, span_type, input=input)
        failure: str | None = None
        try:
            yield span
        except BaseException as exc:  # a cancelled step is as unfinished as a failed one
            detail = str(exc)
            failure = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
            raise
        finally:
            output, span.output = span.output, None  # the block's raw value, before the policy
            _close_span(span, error=failure or span.error, output=output)

    def under(self, span: Span) -> _Spans:
        """This same trace, with `span` as the parent: the scope inside a span just opened."""
        return _Spans(self.trace, span.id, self.usage)


def _close_span(span: Span, *, error: str | None = None, output: Any = None) -> None:
    span.end_time = time.time()
    policy = tracing_config.policy()
    if output is not None and policy.capture_outputs:
        max_bytes = policy.max_tool_result_bytes if span.type == "tool" else policy.max_output_bytes
        span.output = tracing_config.apply_policy(output, max_bytes=max_bytes)
    if error is not None:
        span.status = "error"
        span.error = error


__all__ = ["_Spans", "_close_span"]
