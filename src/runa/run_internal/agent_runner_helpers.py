"""agent_runner_helpers.py: low-level per-turn helpers, model/instruction/tool resolution."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Literal

from runa._models import Model, ModelProvider
from runa._types import ModelSettings, RunContextWrapper
from runa.exceptions import UserError
from runa.handoff import Handoff
from runa.tool import FunctionTool


def _normalized_handoffs(handoffs: list[Any]) -> dict[str, Handoff]:
    """Map each handoff's tool name to its `Handoff`, wrapping a bare `Agent` if given one."""
    result: dict[str, Handoff] = {}
    for entry in handoffs:
        handoff = entry if isinstance(entry, Handoff) else Handoff.from_agent(entry)
        result[handoff.tool_name] = handoff
    return result


def _find_agent_by_name(root: Any, name: str) -> Any:
    """BFS `root` and everything reachable via its `.handoffs`, matching on `.name`.

    Needed to resolve a `RunState`/`Interruption`'s current agent from a name string after
    deserialization: a handoff may have switched the current agent before the pause, so the
    match isn't necessarily `root` itself.
    """
    seen: set[int] = set()
    queue: list[Any] = [root]
    while queue:
        candidate = queue.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if getattr(candidate, "name", None) == name:
            return candidate
        queue.extend(
            handoff.agent
            for handoff in _normalized_handoffs(getattr(candidate, "handoffs", [])).values()
        )
    raise UserError(f"no agent named {name!r} reachable from {getattr(root, 'name', root)!r}")


async def _agent_tools(agent: Any) -> list[FunctionTool]:
    """This agent's own `tools`, plus whatever its `mcp_servers` currently list."""
    tools = list(getattr(agent, "tools", []))
    for server in getattr(agent, "mcp_servers", []):
        tools.extend(await server.list_tools())
    return tools


def _find_tool(tools: list[Any], name: str) -> FunctionTool | None:
    for candidate in tools:
        if isinstance(candidate, FunctionTool) and candidate.name == name:
            return candidate
    return None


def _parse_arguments(args_json: str) -> dict[str, Any] | str:
    """A tool call's arguments as a dict, or an error string to feed back to the model."""
    try:
        args = json.loads(args_json or "{}")
    except json.JSONDecodeError as exc:
        return f"error: invalid JSON arguments: {exc}"
    if not isinstance(args, dict):
        return "error: tool arguments must be a JSON object"
    return args


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _needs_approval(
    tool: FunctionTool, context_wrapper: RunContextWrapper, args: dict[str, Any], call_id: str
) -> bool:
    if isinstance(tool.needs_approval, bool):
        return tool.needs_approval
    return bool(await _maybe_await(tool.needs_approval(context_wrapper, args, call_id)))


@dataclass
class _ApprovalGate:
    """What `_gate_tool_call` decided for one tool call."""

    action: Literal["run", "reject", "interrupt"]
    message: str | None = None


async def _gate_tool_call(
    tool: FunctionTool,
    args: dict[str, Any],
    call_id: str,
    context_wrapper: RunContextWrapper,
    approvals: dict[str, bool] | None = None,
    rejection_messages: dict[str, str] | None = None,
) -> _ApprovalGate:
    """Decide whether a tool call should run, be rejected, or pause for approval.

    Consults `context_wrapper.approval_ledger` first -- the sticky "always approve"/"always
    reject" decisions set via `RunState.approve`/`.reject(..., always=True)` -- before falling
    back to `_needs_approval` and the per-call-id `approvals` dict. Shared by `tool_execution.py`
    (turn-based runs) and `streaming.py` (`run_streamed`), so there's exactly one sanctioned
    approval-gating path rather than two that could drift apart.
    """
    sticky = context_wrapper.approval_ledger.get(tool.name)
    if sticky is True:
        return _ApprovalGate("run")
    if sticky is False:
        return _ApprovalGate(
            "reject",
            context_wrapper.approval_ledger_messages.get(tool.name, "rejected by the operator"),
        )
    if not await _needs_approval(tool, context_wrapper, args, call_id):
        return _ApprovalGate("run")
    verdict = (approvals or {}).get(call_id)
    if verdict is None:
        return _ApprovalGate("interrupt")
    if verdict is False:
        return _ApprovalGate(
            "reject", (rejection_messages or {}).get(call_id, "rejected by the operator")
        )
    return _ApprovalGate("run")


def _model_settings(agent: Any) -> ModelSettings:
    settings = getattr(agent, "model_settings", None)
    return settings if isinstance(settings, ModelSettings) else ModelSettings()


def _resolve_model(agent: Any, model_provider: ModelProvider) -> Model:
    return (
        agent.model if not isinstance(agent.model, str) else model_provider.get_model(agent.model)
    )


async def _resolve_instructions(agent: Any, context_wrapper: RunContextWrapper) -> str | None:
    """Resolve `agent.instructions`: a string passes through, a callable is called and awaited."""
    instructions = getattr(agent, "instructions", None)
    if not callable(instructions):
        return instructions
    return await _maybe_await(instructions(context_wrapper, agent))


__all__ = [
    "_ApprovalGate",
    "_agent_tools",
    "_find_agent_by_name",
    "_find_tool",
    "_gate_tool_call",
    "_maybe_await",
    "_model_settings",
    "_needs_approval",
    "_normalized_handoffs",
    "_parse_arguments",
    "_resolve_instructions",
    "_resolve_model",
]
