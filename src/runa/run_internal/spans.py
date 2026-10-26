"""spans.py: tracing span helpers shared by the turn loop, guardrails, and tool execution.

A run's spans are also where its model calls are charged: inside a span the run opens, a
`metered` call (the loop's own, a context source's, an embedding request anywhere below) is an
`"llm"` span under the innermost open span, counted in the run's usage and in each caller's.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from runa._metering import Charge, charging
from runa._types import Usage
from runa.tracing.spans import Span, close_span, open_span
from runa.tracing.traces import Trace


@dataclass(frozen=True)
class _Spans:
    """Where a span opened right here belongs: which trace, and under which parent.

    One value instead of a `(trace, parent_id)` pair beside every step's own arguments. A step that
    opens spans inside its own (a tool call, around its guardrails) hands on `under(span)` rather
    than remembering to swap one of two parameters.

    `payers` are the usages a call is charged to: that of the run the spans belong to, then those
    of the runs it is nested in, out to the root. With them, a scope is a `Meter`: each `span`
    block charges the `metered` calls inside it to every payer, under that span. Without (a scope
    that only opens an agent span), a span charges nothing of its own.
    """

    trace: Trace
    parent_id: str | None = None
    payers: tuple[Usage, ...] = ()

    def open(self, name: str, span_type: Any, *, input: Any = None) -> Span:
        """Start a span under this scope's parent, recorded on the trace straight away."""
        return open_span(self.trace, name, span_type, parent_id=self.parent_id, input=input)

    @contextmanager
    def span(self, name: str, span_type: Any, *, input: Any = None) -> Iterator[Span]:
        """Open a span for the `with` block and close it on the way out, however the block leaves.

        The block sets `span.output`, or `span.error` for a failure it handles itself; one that
        escapes the block is recorded as the span's error before it propagates. So a step that
        raises can't leave its span open, and the failed span is the one worth inspecting.
        `open`/`close_span` remain for a span whose output comes from later code: the agent's.

        The `metered` calls the block makes, in this task or one it starts, are charged under the
        span, unless a span opened inside it takes them.
        """
        with self._span(name, span_type, input=input) as span:
            if not self.payers:
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
        payers = self.payers
        assert payers, "only a run's spans are charged to"
        with self._span(name, "llm", input=input) as span:

            def charge(spent: Usage) -> None:
                for usage in payers:
                    usage.add(spent)
                span.usage = spent

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
            close_span(span, error=failure or span.error, output=output)

    def under(self, span: Span) -> _Spans:
        """This same trace, with `span` as the parent: the scope inside a span just opened."""
        return _Spans(self.trace, span.id, self.payers)


__all__ = ["_Spans"]
