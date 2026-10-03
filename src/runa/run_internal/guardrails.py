"""guardrails.py: running an agent's or tool's guardrails, raising on a tripwire."""

from typing import Any

from runa._types import RunContextWrapper
from runa.exceptions import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    ToolInputGuardrailTripwireTriggered,
    ToolOutputGuardrailTripwireTriggered,
)
from runa.guardrail import GuardrailResult, ToolInputGuardrailContext, ToolInputGuardrailData
from runa.run_internal.spans import _close_span, _new_span
from runa.tool import FunctionTool
from runa.tracing.traces import Trace


async def _run_input_guardrails(
    agent: Any, context_wrapper: RunContextWrapper, turn_input: Any, trace: Trace, parent_id: str
) -> None:
    for guardrail in agent.input_guardrails:
        span = _new_span(trace, parent_id, guardrail.name or "guardrail", "guardrail")
        result = await guardrail.guardrail_function(context_wrapper, agent, turn_input)
        tripped = result.tripwire_triggered
        _close_span(span, error="tripwire triggered" if tripped else None)
        guardrail_result = GuardrailResult(guardrail, result, tripped)
        context_wrapper.input_guardrail_results.append(guardrail_result)
        if tripped:
            raise InputGuardrailTripwireTriggered(guardrail_result)


async def _run_output_guardrails(
    agent: Any, context_wrapper: RunContextWrapper, output: Any, trace: Trace, parent_id: str
) -> None:
    for guardrail in agent.output_guardrails:
        span = _new_span(trace, parent_id, guardrail.name or "guardrail", "guardrail")
        result = await guardrail.guardrail_function(context_wrapper, agent, output)
        tripped = result.tripwire_triggered
        _close_span(span, error="tripwire triggered" if tripped else None)
        guardrail_result = GuardrailResult(guardrail, result, tripped)
        context_wrapper.output_guardrail_results.append(guardrail_result)
        if tripped:
            raise OutputGuardrailTripwireTriggered(guardrail_result)


async def _run_tool_input_guardrails(
    tool: FunctionTool,
    args_json: str,
    call_id: str,
    context_wrapper: RunContextWrapper,
    trace: Trace,
    parent_id: str,
) -> None:
    for guardrail in tool.tool_input_guardrails or []:
        span = _new_span(trace, parent_id, guardrail.get_name(), "guardrail")
        data = ToolInputGuardrailData(
            context=ToolInputGuardrailContext(tool_arguments=args_json, call_id=call_id)
        )
        result = await guardrail.guardrail_function(data)
        tripped = result.behavior["type"] == "raise_exception"
        _close_span(span, error="tripwire triggered" if tripped else None)
        context_wrapper.tool_input_guardrail_results.append(
            GuardrailResult(guardrail, result, tripped)
        )
        if tripped:
            raise ToolInputGuardrailTripwireTriggered(guardrail, result)


async def _run_tool_output_guardrails(
    tool: FunctionTool,
    args_json: str,
    call_id: str,
    output: Any,
    context_wrapper: RunContextWrapper,
    trace: Trace,
    parent_id: str,
) -> None:
    for guardrail in tool.tool_output_guardrails or []:
        span = _new_span(trace, parent_id, guardrail.get_name(), "guardrail")
        data = ToolInputGuardrailData(
            context=ToolInputGuardrailContext(tool_arguments=args_json, call_id=call_id),
            output=output,
        )
        result = await guardrail.guardrail_function(data)
        tripped = result.behavior["type"] == "raise_exception"
        _close_span(span, error="tripwire triggered" if tripped else None)
        context_wrapper.tool_output_guardrail_results.append(
            GuardrailResult(guardrail, result, tripped)
        )
        if tripped:
            raise ToolOutputGuardrailTripwireTriggered(guardrail, result)


__all__ = [
    "_run_input_guardrails",
    "_run_output_guardrails",
    "_run_tool_input_guardrails",
    "_run_tool_output_guardrails",
]
