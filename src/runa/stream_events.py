"""stream_events.py: the events `Agent.run_streamed` yields."""

from dataclasses import dataclass
from typing import Any, Literal

from runa._items import ConversationItem
from runa._models import StreamDelta


@dataclass
class RawResponsesStreamEvent:
    """A raw, provider-shaped fragment of a streamed response, passed through as-is."""

    data: StreamDelta
    type: Literal["raw_response_event"] = "raw_response_event"


@dataclass
class RunItemStreamEvent:
    """One completed item produced mid-stream: a message, a tool call, a tool's output, ..."""

    name: Literal["message_output_created", "tool_called", "tool_output", "handoff_occured"]
    item: ConversationItem
    type: Literal["run_item_stream_event"] = "run_item_stream_event"


@dataclass
class AgentUpdatedStreamEvent:
    """A handoff switched the agent running this turn."""

    new_agent: Any
    type: Literal["agent_updated_stream_event"] = "agent_updated_stream_event"


StreamEvent = RawResponsesStreamEvent | RunItemStreamEvent | AgentUpdatedStreamEvent


__all__ = [
    "AgentUpdatedStreamEvent",
    "RawResponsesStreamEvent",
    "RunItemStreamEvent",
    "StreamEvent",
]
