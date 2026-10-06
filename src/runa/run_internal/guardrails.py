"""guardrails.py: running an agent's or tool's guardrails, raising on a tripwire."""

from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from runa._types import RunContextWrapper
from runa.exceptions import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    ToolInputGuardrailTripwireTriggered,
    ToolOutputGuardrailTripwireTriggered,
)
from runa.guardrail import GuardrailResult, ToolInputGuardrailContext, ToolInputGuardrailData
from runa.run_internal.agent_shape import AgentShape
from runa.run_internal.spans import _close_span, _Spans
from runa.tool import FunctionTool


async def _run_guardrails(
    entries: Iterable[Any],
    *,
    invoke: Callable[[Any], Awaitable[Any]],
    record_to: list[GuardrailResult],
    raise_as: Callable[[GuardrailResult], Exception],
    spans: _Spans,
) -> None:
    """Run `entries` in list order, span and record each verdict, raise on the first that trips.

    The ordering rule is stated here and nowhere else: the first tripwire stops the entries after
    it, but every entry that did run is already appended to `record_to`, so a tripped run's audit
    trail is complete up to the stop. The four lists differ only in what the caller binds:
    `invoke` adapts the two guardrail call shapes (an agent's `(context, agent, value)`, a tool's
    `(data)`) to one call, and `raise_as` names which tripwire exception this list raises.
    """
    for entry in entries:
        span = spans.open(entry.get_name(), "guardrail")
        result = await invoke(entry)
        _close_span(span, error="tripwire triggered" if result.tripped else None)
        guardrail_result = GuardrailResult(entry, result, result.tripped)
        record_to.append(guardrail_result)
        if result.tripped:
            raise raise_as(guardrail_result)


def _tool_data(
    tool: FunctionTool, args_json: str, call_id: str, output: Any = None
) -> ToolInputGuardrailData:
    """Build the `data` a tool's guardrails see: which tool, the call, its arguments, its output.

    The only place a `ToolInputGuardrailContext` is constructed, so a field it carries can't be
    populated on one side of the call and forgotten on the other.
    """
    return ToolInputGuardrailData(
        context=ToolInputGuardrailContext(
            tool_arguments=args_json, tool_name=tool.name, call_id=call_id
        ),
        output=output,
    )


async def _run_input_guardrails(
    shape: AgentShape, context_wrapper: RunContextWrapper, turn_input: Any, spans: _Spans
) -> None:
    """Run the agent's input guardrails against this turn's input, before the model sees it."""
    await _run_guardrails(
        shape.input_guardrails,
        invoke=lambda entry: entry.guardrail_function(context_wrapper, shape.agent, turn_input),
        record_to=context_wrapper.input_guardrail_results,
        raise_as=InputGuardrailTripwireTriggered,
        spans=spans,
    )


async def _run_output_guardrails(
    shape: AgentShape, context_wrapper: RunContextWrapper, output: Any, spans: _Spans
) -> None:
    """Run the agent's output guardrails against its final output, before the run returns it."""
    await _run_guardrails(
        shape.output_guardrails,
        invoke=lambda entry: entry.guardrail_function(context_wrapper, shape.agent, output),
        record_to=context_wrapper.output_guardrail_results,
        raise_as=OutputGuardrailTripwireTriggered,
        spans=spans,
    )


async def _run_tool_input_guardrails(
    tool: FunctionTool,
    args_json: str,
    call_id: str,
    context_wrapper: RunContextWrapper,
    spans: _Spans,
) -> None:
    """Run the tool's input guardrails against the call's raw arguments, before it runs."""
    data = _tool_data(tool, args_json, call_id)
    await _run_guardrails(
        tool.tool_input_guardrails or [],
        invoke=lambda entry: entry.guardrail_function(data),
        record_to=context_wrapper.tool_input_guardrail_results,
        raise_as=lambda result: ToolInputGuardrailTripwireTriggered(
            result.guardrail, result.output
        ),
        spans=spans,
    )


async def _run_tool_output_guardrails(
    tool: FunctionTool,
    args_json: str,
    call_id: str,
    output: Any,
    context_wrapper: RunContextWrapper,
    spans: _Spans,
) -> None:
    """Run the tool's output guardrails against what it returned, before the model sees it."""
    data = _tool_data(tool, args_json, call_id, output=output)
    await _run_guardrails(
        tool.tool_output_guardrails or [],
        invoke=lambda entry: entry.guardrail_function(data),
        record_to=context_wrapper.tool_output_guardrail_results,
        raise_as=lambda result: ToolOutputGuardrailTripwireTriggered(
            result.guardrail, result.output
        ),
        spans=spans,
    )


__all__ = [
    "_run_input_guardrails",
    "_run_output_guardrails",
    "_run_tool_input_guardrails",
    "_run_tool_output_guardrails",
]
