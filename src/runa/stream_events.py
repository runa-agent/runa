"""stream_events.py: the events `Agent.run_streamed` yields.

Four kinds, told apart by class (`isinstance`, or `match event: case TextDelta(text=text):`).
Everything else a run produced, the finished message included, is on the `Run` the stream ends
with.
"""

from dataclasses import dataclass

from runa._items import ConversationItem
from runa.agent_like import AgentLike


@dataclass(frozen=True)
class TextDelta:
    """The next piece of the answer's text, as the model produces it."""

    text: str


@dataclass(frozen=True)
class ToolCalled:
    """The model called a tool: `item` is the call, before it runs."""

    item: ConversationItem


@dataclass(frozen=True)
class ToolOutput:
    """A tool call finished: `item` is its result, as the model will read it."""

    item: ConversationItem


@dataclass(frozen=True)
class AgentChanged:
    """A handoff switched the agent running this turn to `agent`."""

    agent: AgentLike


StreamEvent = TextDelta | ToolCalled | ToolOutput | AgentChanged


__all__ = ["AgentChanged", "StreamEvent", "TextDelta", "ToolCalled", "ToolOutput"]
