"""streaming.py: `Runner.run_streamed`'s event loop, translates model deltas into `StreamEvent`s.

The public `RunResultStreaming` wrapper around this loop lives in `runa.result`, alongside
`RunResult`, since both are things `Runner` returns to a caller rather than execution-time detail.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from runa._types import RunContextWrapper, TResponseInputItem, Usage
from runa.exceptions import ApprovalRequiredError, DuplicateToolCallError, MaxTurnsExceeded
from runa.lifecycle import RunHooks
from runa.run_config import RunConfig
from runa.run_internal.agent_runner_helpers import (
    _agent_tools,
    _find_tool,
    _gate_tool_call,
    _model_settings,
    _normalized_handoffs,
    _parse_arguments,
    _resolve_instructions,
    _resolve_model,
)
from runa.stream_events import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
    StreamEvent,
)


async def _stream_async(
    agent: Any,
    items: list[TResponseInputItem],
    context_wrapper: RunContextWrapper,
    run_config: RunConfig,
    hooks: RunHooks[Any],
) -> AsyncIterator[StreamEvent]:
    current_agent = agent

    await hooks.on_agent_start(context_wrapper, current_agent)
    for _turn in range(run_config.max_turns):
        model = _resolve_model(current_agent, run_config.model_provider)
        text_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        usage = Usage()

        system_instructions = await _resolve_instructions(current_agent, context_wrapper)
        turn_tools = await _agent_tools(current_agent)
        handoff_map = _normalized_handoffs(getattr(current_agent, "handoffs", []))
        async for delta in model.stream_response(
            system_instructions,
            items,
            _model_settings(current_agent),
            turn_tools,
            getattr(current_agent, "output_type", None),
            list(handoff_map.values()),
        ):
            yield RawResponsesStreamEvent(data=delta)
            if delta.text:
                text_parts.append(delta.text)
            if delta.tool_call_index is not None:
                entry = tool_calls.setdefault(
                    delta.tool_call_index,
                    {"id": None, "type": "function", "function": {"name": None, "arguments": ""}},
                )
                if delta.tool_call_id:
                    entry["id"] = delta.tool_call_id
                if delta.tool_call_name:
                    entry["function"]["name"] = delta.tool_call_name
                if delta.tool_call_arguments:
                    entry["function"]["arguments"] += delta.tool_call_arguments
            if delta.usage is not None:
                usage = delta.usage

        context_wrapper.usage.add(usage)
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(text_parts) or None,
            "tool_calls": [tool_calls[i] for i in sorted(tool_calls)] or None,
        }
        items.append(message)
        yield RunItemStreamEvent(name="message_output_created", item=message)

        if not message["tool_calls"]:
            await hooks.on_agent_end(context_wrapper, current_agent, message["content"] or "")
            return

        for call in message["tool_calls"]:
            yield RunItemStreamEvent(name="tool_called", item=call)
            name = call["function"]["name"]
            if name in handoff_map:
                current_agent = handoff_map[name].agent
                yield AgentUpdatedStreamEvent(new_agent=current_agent)
                items.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": f"Transferred to {current_agent.name}.",
                    }
                )
                continue
            tool = _find_tool(turn_tools, name)
            if tool is None:
                items.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": f"error: unknown tool {name}",
                    }
                )
                continue
            call_id = call["id"]
            args_json = call["function"]["arguments"] or "{}"
            args = _parse_arguments(args_json)
            if isinstance(args, str):
                tool_result = {"role": "tool", "tool_call_id": call_id, "content": args}
                items.append(tool_result)
                yield RunItemStreamEvent(name="tool_output", item=tool_result)
                continue
            gate = await _gate_tool_call(tool, args, call_id, context_wrapper)
            if gate.action == "interrupt":
                raise ApprovalRequiredError(tool.name, call_id)
            if gate.action == "reject":
                tool_result = {"role": "tool", "tool_call_id": call_id, "content": gate.message}
                items.append(tool_result)
                yield RunItemStreamEvent(name="tool_output", item=tool_result)
                continue
            if call_id in context_wrapper.executed_call_ids:
                raise DuplicateToolCallError(call_id, tool.name)
            context_wrapper.executed_call_ids.add(call_id)
            try:
                result = await tool.on_invoke_tool(context_wrapper, args_json, call_id)
            except Exception as exc:  # noqa: BLE001 -- fed back to the model, not a run-ending error
                result = f"error: {exc}"
            tool_result = {"role": "tool", "tool_call_id": call_id, "content": str(result)}
            items.append(tool_result)
            yield RunItemStreamEvent(name="tool_output", item=tool_result)

    raise MaxTurnsExceeded(f"max turns ({run_config.max_turns}) exceeded")


__all__ = ["_stream_async"]
