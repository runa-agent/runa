"""interface.py: the `Model` protocol, `StreamDelta`, and wire-format bits both backends share."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import TypeAdapter

from runa._types import ModelResponse, ModelSettings, TResponseInputItem, Usage


class Model(Protocol):
    """What `runa.run_internal` needs from a model backend: a non-streaming and a streaming call.

    `tools`/`handoffs` are read structurally (see the package docstring); `output_schema` is
    `None` or `str` for plain-text output, or any other type to ask the backend for JSON output.
    """

    async def get_response(
        self,
        system_instructions: str | None,
        input: list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Any],
        output_schema: type | None,
        handoffs: list[Any],
    ) -> ModelResponse:
        """Send one turn to the model and return its full response."""
        ...

    def stream_response(
        self,
        system_instructions: str | None,
        input: list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Any],
        output_schema: type | None,
        handoffs: list[Any],
    ) -> AsyncIterator[StreamDelta]:
        """Send one turn to the model and yield incremental `StreamDelta`s as it responds."""
        ...


@dataclass
class StreamDelta:
    """One incremental fragment of a streamed model response.

    Only the fields relevant to a given fragment are set. `run_internal` accumulates a stream of
    these into a final message: `text` fragments concatenate; a tool-call fragment is keyed by
    `tool_call_index`, with `id`/`name` set once (when the call starts) and `arguments` arriving
    in pieces to be concatenated; `usage` is set once, on whichever fragment carries it (a
    provider's final chunk, in practice).
    """

    text: str | None = None
    tool_call_index: int | None = None
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    tool_call_arguments: str | None = None
    usage: Usage | None = None


def _tool_dict(tool: Any) -> dict[str, Any]:
    """Convert a Runa `FunctionTool`-shaped object to a chat-completions tool definition."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.params_json_schema or {"type": "object", "properties": {}},
        },
    }


def _handoff_dict(handoff: Any) -> dict[str, Any]:
    """Convert a Runa `Handoff`-shaped object to a chat-completions tool definition.

    A handoff takes no structured input from the model: calling it is itself the signal to
    switch agents, so its schema is always an empty object.
    """
    return {
        "type": "function",
        "function": {
            "name": handoff.tool_name,
            "description": handoff.tool_description or "",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _output_json_schema(output_schema: type | None) -> dict[str, Any] | None:
    """The JSON schema a structured `output_schema` asks for; `None` for plain-text output."""
    if output_schema is None or output_schema is str:
        return None
    return _closed(TypeAdapter(output_schema).json_schema())


def _closed(schema: Any) -> Any:
    """Forbid extra keys on every object in `schema`, as providers' JSON output modes require."""
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema.setdefault("additionalProperties", False)
        for value in schema.values():
            _closed(value)
    elif isinstance(schema, list):
        for value in schema:
            _closed(value)
    return schema


__all__ = ["Model", "StreamDelta", "_handoff_dict", "_output_json_schema", "_tool_dict"]
