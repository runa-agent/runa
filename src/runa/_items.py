"""_items.py: the conversation item, and every reader and writer of one.

A `ConversationItem` is one turn of conversation as Runa's runtime passes it around: a plain
dict in OpenAI's chat-completions wire format (`tool_calls` on an assistant message,
`function.arguments` as a JSON string, `{"role": "tool", "tool_call_id": ...}` for a result).
`runa._models.anthropic` is the only module that converts away from this shape.

The item stays a plain dict so it survives a JSON round trip through `agent.history`,
`session.add_items`, a `Compactor` and a serialized `RunState`, and application code can read it
without importing a type. Because `content` is a string on one turn and a list of parts on the
next, every reader in Runa goes through this module: `content_text` for words, and
`tool_calls`/`call_id`/`call_name`/`call_arguments`/`answered_call_id` for the tool-call envelope.

A provider content kind that is neither words nor a tool call (Claude's signed thinking blocks)
rides in `provider_blocks`, read only by the backend that wrote it and stripped by `wire_items`
on the way to any other.

`parsed_arguments` and `parsed_json_object` read JSON a model wrote; both raise `ValueError` and
leave the policy for an unreadable reply to the caller. Every item Runa puts into history is
built here too (the user message, the system block, a tool result, the assistant message a
backend assembles), so a backend translates its provider's response into these arguments but
never writes the envelope itself.
"""

import json
import re
from typing import Any

# A trailing comma before a closing bracket: the one malformation `parsed_json_object` repairs.
_TRAILING_COMMA = re.compile(r",\s*([\]}])")

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
    text: str | None,
    tool_calls: list[ConversationItem] | None,
    provider_blocks: list[Any] | None = None,
) -> ConversationItem:
    """The model's turn, as the assistant message a backend assembles its response into.

    Both halves are optional and both spell their absence the same way, as `None` rather than
    `""`/`[]`: a turn is words, or calls, or both. Backends differ in how they arrive at the two
    -- one provider hands over a finished string, another a list of content blocks, a stream a
    pile of deltas -- and that translation is theirs; this envelope is not.

    `provider_blocks` is the third thing a turn can carry and the one two providers have no
    shared place for: Claude's `thinking` and `redacted_thinking` blocks, which hold the
    signature a later request needs to hand the model its own reasoning back. A backend that can
    say a block in `text` or `tool_calls` still says it there; one that can't leaves it here
    verbatim and reads its own back with `provider_blocks` when it rebuilds its turns, so the
    block survives `agent.history`, a session and a serialized `RunState` instead of being
    dropped at the boundary for want of a slot -- a loss nothing would report, since the run
    completes and only the bill says the model reasoned from scratch every turn.

    Absent rather than `None` when there are none, unlike the two halves above: those are what a
    turn means, this is one backend's own baggage, and `wire_items` takes it back off on the way
    out to a provider that never put it there.
    """
    message: ConversationItem = {
        "role": "assistant",
        "content": text or None,
        "tool_calls": tool_calls or None,
    }
    if provider_blocks:
        message["provider_blocks"] = provider_blocks
    return message


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


def tool_calls(message: ConversationItem) -> list[ConversationItem]:
    """The call envelopes an assistant message carries, `[]` when it carries none.

    Both spellings of "no calls" read as the empty list -- an absent `tool_calls`, and the
    explicit `None` `assistant_message` writes -- so asking whether this turn called anything is
    just asking whether the list is empty.
    """
    return message.get("tool_calls") or []


def provider_blocks(message: ConversationItem) -> list[Any]:
    """The provider content blocks an assistant message carries, `[]` when it carries none.

    For the backend that wrote them (see `assistant_message`) and for nobody else: the blocks
    are one provider's shape, so anyone but the backend replaying them is reading a format it has
    no claim on. What Runa itself wants from a turn -- its words, its calls -- reads without them.
    """
    return message.get("provider_blocks") or []


def wire_items(items: list[ConversationItem]) -> list[ConversationItem]:
    """`items` as a chat-completions provider is sent them: without the provider-blocks slot.

    One key off and nothing else, an item being that wire's shape already. It's the one key no
    provider put there (see `assistant_message`), and a strict endpoint rejects a field it
    doesn't name, so the backend that passes items straight into a request body passes them
    through here first.
    """
    return [
        {key: value for key, value in item.items() if key != "provider_blocks"} for item in items
    ]


def call_id(call: ConversationItem) -> str:
    """One call's id: what its result is tied back to, and what an approval decision is keyed by."""
    return call["id"]


def call_name(call: ConversationItem) -> str:
    """The tool a call names, to be found among an agent's tools and handoffs."""
    return call["function"]["name"]


def call_arguments(call: ConversationItem) -> str:
    """One call's arguments, as the JSON string the model produced, `"{}"` when it produced none.

    A tool taking no arguments is called with an empty object by some providers, an empty string
    by others and no field at all by the rest; they are the same call, and that is settled here
    rather than at each site that reads one. Parsing the string is `parsed_arguments`' job:
    arguments travel as written through history, a `RunState` blob and a tool-input guardrail.
    """
    return call["function"].get("arguments") or "{}"


def answered_call_id(result: ConversationItem) -> str:
    """Which call a tool result answers: how a turn's results are matched back to its calls."""
    return result["tool_call_id"]


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


def parsed_json_object(text: str) -> dict[str, Any]:
    """The JSON object a model meant by `text`, however it wrapped it.

    For a reply Runa asked to be JSON but which arrives as prose -- a judge's verdict, memory
    extraction's list of facts -- as against a tool call's `arguments`, which arrive as JSON and
    nothing else and are `parsed_arguments`' job. Such a reply comes wrapped in commentary or a
    markdown fence more often than not, and now and then with a trailing comma before a closing
    `]`/`}`: both are tolerated, here rather than at each site, so that how tolerant Runa is of
    a model's JSON stops being a property of which module the reply happened to land in.

    Raises `ValueError` when even that doesn't parse. What to do about it is the caller's policy,
    as above: an `ERROR` result for a judge, a logged warning for memory extraction.
    """
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"model reply contained no JSON object: {text!r}")
    candidate = text[start:end]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_TRAILING_COMMA.sub(r"\1", candidate))
    except json.JSONDecodeError as exc:
        raise ValueError(f"model reply isn't a JSON object ({exc}): {text!r}") from exc


__all__ = [
    "ConversationItem",
    "answered_call_id",
    "assistant_message",
    "call_arguments",
    "call_id",
    "call_name",
    "content_text",
    "item_text",
    "latest_text",
    "latest_user_index",
    "latest_user_text",
    "role_and_text",
    "parsed_arguments",
    "parsed_json_object",
    "provider_blocks",
    "system_message",
    "tool_call",
    "tool_calls",
    "tool_result",
    "user_message",
    "wire_items",
]
