"""guardrails.py: running one `Phase`'s guardrails, recording each verdict, raising on a trip."""

from typing import Any

from runa.exceptions import GuardrailTripwireTriggered
from runa.guardrail import GuardrailResult, Phase
from runa.run_internal.active_run import _Run
from runa.run_internal.spans import _close_span, _Spans
from runa.tool import FunctionTool


async def _run_guardrails(
    run: _Run,
    phase: Phase,
    value: Any,
    *,
    tool: FunctionTool | None = None,
    spans: _Spans | None = None,
) -> None:
    """Run `phase`'s guardrails over `value` in list order, raising on the first one that trips.

    The ordering rule is stated here and nowhere else: the first tripwire stops the entries after
    it, but every entry that already ran is in the run's audit trail, so a tripped run's trail is
    complete up to the stop.

    `phase` is the whole of what four near-identical binders used to be: whose list to read (the
    agent's, or `tool`'s), what each entry is handed -- an agent's `(context, agent, value)`, a
    tool's one `data` -- and which phase the audit trail and a tripwire report. A tool phase also
    passes `spans`, so its guardrails hang under the call they guard rather than under the run.
    """
    entries = (run.shape.guardrails if tool is None else tool.guardrails).get(phase, ())
    if not entries:
        return
    scope = run.spans if spans is None else spans
    results = run.context_wrapper.guardrail_results
    for entry in entries:
        span = scope.open(entry.get_name(), "guardrail")
        verdict = await (
            entry.guardrail_function(value)
            if phase.on_tool
            else entry.guardrail_function(run.context_wrapper, run.current_agent, value)
        )
        _close_span(span, error="tripwire triggered" if verdict.tripped else None)
        result = GuardrailResult(entry, verdict, verdict.tripped, phase)
        results.record(result)
        if verdict.tripped:
            raise GuardrailTripwireTriggered(result)


__all__ = ["_run_guardrails"]
