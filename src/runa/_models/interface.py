"""interface.py: the `Model` protocol, the `ModelRequest` it takes, and what both backends share.

Nothing here is wire-format-specific: a `ToolSchema` is neither provider's shape, and each
backend wraps it in its own envelope exactly once.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import TypeAdapter

from runa._types import ModelResponse, ModelSettings, TResponseInputItem, Usage


@dataclass(frozen=True)
class ModelRequest:
    """One turn's worth of input: everything a backend needs to put a request on the wire.

    `tools`/`handoffs` are read structurally (see the package docstring); `output_schema` is
    `None` or `str` for plain-text output, or any other type to ask the backend for JSON output.
    """

    input: list[TResponseInputItem]
    system_instructions: str | None = None
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    tools: list[Any] = field(default_factory=list)
    output_schema: type | None = None
    handoffs: list[Any] = field(default_factory=list)

    @property
    def messages(self) -> list[TResponseInputItem]:
        """`input`, with `system_instructions` prepended as a system message if there are any."""
        if not self.system_instructions:
            return list(self.input)
        return [{"role": "system", "content": self.system_instructions}, *self.input]


class Model(Protocol):
    """What `runa.run_internal` needs from a model backend: a non-streaming and a streaming call."""

    async def get_response(self, request: ModelRequest) -> ModelResponse:
        """Send one turn to the model and return its full response."""
        ...

    def stream_response(self, request: ModelRequest) -> AsyncIterator[StreamDelta]:
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


@dataclass(frozen=True)
class ToolSchema:
    """One callable the model is offered, in neither provider's wire shape."""

    name: str
    description: str
    parameters: dict[str, Any]


def tool_schemas(request: ModelRequest) -> list[ToolSchema]:
    """Every tool and handoff on `request`, in the order the model is offered them.

    A handoff takes no structured input from the model: calling it is itself the signal to
    switch agents, so its `parameters` is always an empty object.
    """
    return [
        ToolSchema(
            tool.name,
            tool.description or "",
            tool.params_json_schema or {"type": "object", "properties": {}},
        )
        for tool in request.tools
    ] + [
        ToolSchema(
            handoff.tool_name, handoff.tool_description or "", {"type": "object", "properties": {}}
        )
        for handoff in request.handoffs
    ]


def output_json_schema(request: ModelRequest) -> dict[str, Any] | None:
    """The JSON schema `request.output_schema` asks for; `None` for plain-text output."""
    output_schema = request.output_schema
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


__all__ = [
    "Model",
    "ModelRequest",
    "StreamDelta",
    "ToolSchema",
    "output_json_schema",
    "tool_schemas",
]
