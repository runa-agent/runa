"""guardrails.py: running one `Phase`'s guardrails, recording each verdict, raising on a trip."""

from typing import Any

from runa.exceptions import GuardrailTripwireTriggered
from runa.guardrail import GuardrailResult, Phase
from runa.run_internal.active_run import _Run
from runa.run_internal.spans import _Spans
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
    agent's, or `tool`'s), and which phase the audit trail and a tripwire report. Every entry is
    called the same way, with the one `value` the caller already has -- what each phase makes of
    that value is stated once, in `guardrail._checked`, beside the `Phase` that decides it. A
    tool phase also passes `spans`, so its guardrails hang under the call they guard rather than
    under the run.
    """
    entries = (run.shape.guardrails if tool is None else tool.guardrails).get(phase, ())
    if not entries:
        return
    scope = run.spans if spans is None else spans
    results = run.governance.guardrail_results
    for entry in entries:
        with scope.span(entry.get_name(), "guardrail") as span:
            verdict = await entry.guardrail_function(value)
            if verdict.tripped:
                span.error = "tripwire triggered"
        result = GuardrailResult(entry, verdict, verdict.tripped, phase)
        results.record(result)
        if verdict.tripped:
            raise GuardrailTripwireTriggered(result)


__all__ = ["_run_guardrails"]
