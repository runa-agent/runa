"""_items.py: the conversation item, and every reader and writer of one.

A `ConversationItem` is one turn of conversation as Runa's runtime passes it around: a plain
dict in OpenAI's chat-completions wire format (`tool_calls` on an assistant message,
`function.arguments` as a JSON string, `{"role": "tool", "tool_call_id": ...}` for a result).
That is a deliberate choice, not a neutral interchange format. There is no neutral format, and
adopting the one two of the three major APIs already speak costs one translating backend
(`runa._models.anthropic`, the only module that ever converts away from this shape) where
inventing a third would cost every backend one. The whole argument, and the typed-item
alternative it rejects.

The item stays a plain dict on purpose: `agent.history`, `session.add_items`, a `Compactor` and
a serialized `RunState` all carry items, so one has to survive a JSON round trip and be readable
by application code without importing a type. What plainness costs is that reading an item is
guesswork -- `content` is a string on one turn and a list of parts on the next, a tool call
carries no text at all -- so every reader in Runa goes through this module rather than guessing
locally. Adding a content kind (reasoning blocks, citations, audio) is then a change to
`content_text` here, not to a parser in each of six modules.

Writing one is the same bet pointed the other way. Every item Runa puts into history is built
here: the user message a turn arrives as, the system block that carries instructions or
retrieved context, the result a tool call comes back as, and the assistant message a backend
assembles a provider's response into, with the tool-call envelopes it carries. So giving any of
them a field (structured content, an `is_error` flag on a failed call) is a change to one
function rather than to every site that wrote the dict out by hand -- and the rule about when an
absent field is `None` is stated once rather than copied per backend. A backend owns translating
its provider's shape into these arguments, which is genuinely its own; it does not own the
envelope, which is this module's.
"""

import json
from typing import Any

ConversationItem = dict[str, Any]
"""One turn of conversation history: a `{"role": ..., "content": ...}` message, a tool call, or a
tool result. Plain JSON, never a provider SDK type, and one shape in both directions: a model's
output item becomes the next turn's input item once appended to history.
"""


def user_message(content: str | list[Any]) -> ConversationItem:
    """This turn's words, as the user message that carries them into a run.

    `content` is a plain string, or the parts list `runa.content` builds for a multimodal turn:
    both ride in the same item, a parts list being how the chat-completions shape spells one.
    """
    return {"role": "user", "content": content}


def assistant_message(
    text: str | None, tool_calls: list[ConversationItem] | None
) -> ConversationItem:
    """The model's turn, as the assistant message a backend assembles its response into.

    Both halves are optional and both spell their absence the same way, as `None` rather than
    `""`/`[]`: a turn is words, or calls, or both. Backends differ in how they arrive at the two
    -- one provider hands over a finished string, another a list of content blocks, a stream a
    pile of deltas -- and that translation is theirs; this envelope is not.
    """
    return {"role": "assistant", "content": text or None, "tool_calls": tool_calls or None}


def tool_call(call_id: str, name: str, arguments: str) -> ConversationItem:
    """One call the model made, as the envelope an assistant message's `tool_calls` carries.

    `arguments` stays the JSON string the model produced, unparsed: it round-trips through
    history and a serialized `RunState` as written, and `parsed_arguments` is how a caller that
    needs the keyword arguments gets them.
    """
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def system_message(content: str) -> ConversationItem:
    """Instructions, or a block of retrieved context, as the system message carrying them.

    Everything a run tells the model that the user didn't say rides in one of these: an agent's
    `instructions`, retrieved memories, matching knowledge.
    """
    return {"role": "system", "content": content}


def tool_result(call_id: str, content: object) -> ConversationItem:
    """One tool call's result, as the item answering the call `call_id` names.

    Every result a turn produces leaves through here -- what a tool returned, a handoff
    acknowledgement, a rejected call's message, an error the model should see and retry from --
    so the shape, and the rule that a result reads as text whatever the tool returned, are
    stated once instead of at each of those sites.
    """
    return {"role": "tool", "tool_call_id": call_id, "content": str(content)}


def content_text(content: Any) -> str:
    """Flatten an item's `content` -- a string, a list of parts, or nothing -- to plain text.

    `""` when there are no words to find: a tool call, an image-only message, a missing
    `content`. Parts carrying no string `text` (an `image_url` part) are skipped, so a
    multimodal message reduces to what was written in it.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def item_text(item: ConversationItem) -> str:
    """One item's text, `""` when it carries none: the one way Runa reads an item's words."""
    return content_text(item.get("content"))


def role_and_text(item: ConversationItem) -> tuple[str, str]:
    """One item as the `(role, text)` a transcript shows (`runa sessions`, the Sessions pages).

    A display view, so neither half is ever empty: an item with no `role` is labeled by its
    `type`, and one with no readable text (a tool call, an image) shows its raw dict instead of
    a blank line.
    """
    role = item.get("role") or item.get("type") or "item"
    return str(role), item_text(item) or str(item)


def latest_text(items: list[ConversationItem]) -> str:
    """The last item's text, `""` for an empty list: what an input guardrail is handed."""
    return item_text(items[-1]) if items else ""


def latest_user_index(items: list[ConversationItem]) -> int | None:
    """Where the most recent plain-text user message in `items` is, if any.

    Plain-text only: a multimodal turn's `content` is a list of parts, and the two callers both
    want a string to work with -- a memory search query, and the point a rolling compaction cuts
    at -- rather than a flattening of one.
    """
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        if item.get("role") == "user" and isinstance(item.get("content"), str):
            return index
    return None


def latest_user_text(items: list[ConversationItem]) -> str | None:
    """The most recent plain-text user message in `items`, Memory's default search query."""
    index = latest_user_index(items)
    return items[index]["content"] if index is not None else None


def parsed_arguments(args_json: str) -> dict[str, Any]:
    """A tool call's `function.arguments` JSON string as the dict of keyword arguments it names.

    Raises `ValueError` when the model produced something that isn't a JSON object; what to do
    about that is the caller's policy, not this module's. The turn loop hands the message back
    to the model as that call's result; a backend translating history has no such recourse and
    lets it raise.
    """
    try:
        args = json.loads(args_json or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON arguments: {exc}") from exc
    if not isinstance(args, dict):
        raise ValueError("tool arguments must be a JSON object")
    return args


__all__ = [
    "ConversationItem",
    "assistant_message",
    "content_text",
    "item_text",
    "latest_text",
    "latest_user_index",
    "latest_user_text",
    "role_and_text",
    "parsed_arguments",
    "system_message",
    "tool_call",
    "tool_result",
    "user_message",
]
