"""Tests for `runa._items`: the one reader and writer of a chat-completions-shaped item."""

import pytest

from runa._items import (
    answered_call_id,
    assistant_message,
    call_arguments,
    call_id,
    call_name,
    content_text,
    item_text,
    latest_text,
    latest_user_index,
    latest_user_text,
    parsed_arguments,
    parsed_json_object,
    role_and_text,
    tool_call,
    tool_calls,
    tool_result,
    user_message,
)


def test_user_message_carries_this_turn_into_a_run() -> None:
    """A turn's words become the one item shape the loop starts from."""
    assert user_message("hello") == {"role": "user", "content": "hello"}


def test_user_message_keeps_a_multimodal_turn_as_parts() -> None:
    """A parts list is how the wire shape spells a multimodal turn, so it rides unflattened."""
    parts = [{"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {}}]

    assert user_message(parts) == {"role": "user", "content": parts}


def test_tool_result_answers_the_call_it_names() -> None:
    """A result is tied to its call by `tool_call_id`, which is what the model matches on."""
    assert tool_result("c1", "sunny") == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "sunny",
    }


def test_tool_result_reads_as_text_whatever_the_tool_returned() -> None:
    """A tool returning a number or a dict still comes back as words for the model to read."""
    assert tool_result("c1", 42)["content"] == "42"
    assert tool_result("c2", {"ok": True})["content"] == "{'ok': True}"


def test_content_text_passes_a_plain_string_through() -> None:
    """A string `content` is already the item's text."""
    assert content_text("hello") == "hello"


def test_content_text_joins_the_text_parts_of_a_multimodal_message() -> None:
    """A parts list reduces to the words in it; an `image_url` part contributes nothing."""
    content = [
        {"type": "text", "text": "look at "},
        {"type": "image_url", "image_url": {"url": "https://example.test/cat.png"}},
        {"type": "text", "text": "this"},
    ]

    assert content_text(content) == "look at this"


@pytest.mark.parametrize("content", [None, 42, {"text": "not a list"}])
def test_content_text_is_empty_for_anything_without_words(content: object) -> None:
    """No text to find is `""`, never the string `"None"` or a stringified dict."""
    assert content_text(content) == ""


def test_item_text_reads_an_item_with_no_content_as_empty() -> None:
    """A tool call carries no `content` at all, and reads as no text rather than as `"None"`."""
    call = {
        "role": "assistant",
        "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
    }

    assert item_text(call) == ""


def test_role_and_text_labels_a_typed_item_by_its_type() -> None:
    """An item with no `role` is labeled by its `type` for a transcript."""
    assert role_and_text({"type": "reasoning", "content": "thinking"}) == ("reasoning", "thinking")


def test_role_and_text_falls_back_to_the_raw_item_when_there_is_no_text() -> None:
    """A transcript shows the raw dict rather than a blank line, so neither half is empty."""
    item = {"role": "assistant", "tool_calls": [{"id": "c1"}]}

    role, text = role_and_text(item)

    assert role == "assistant"
    assert text == str(item)


def test_role_and_text_labels_an_item_with_neither_role_nor_type() -> None:
    """Some item is still some item: `"item"` is the last fallback."""
    assert role_and_text({"content": "orphan"}) == ("item", "orphan")


def test_latest_text_reads_the_last_item() -> None:
    """An input guardrail checks the turn that just arrived, not the whole transcript."""
    items = [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}]

    assert latest_text(items) == "second"


def test_latest_text_of_no_items_is_empty() -> None:
    """An empty list has no latest text, and says so with `""` rather than raising."""
    assert latest_text([]) == ""


def test_latest_user_text_skips_later_assistant_and_tool_turns() -> None:
    """Memory's default query is the user's own words, however the turn ended."""
    items = [
        {"role": "user", "content": "where is my order"},
        {"role": "assistant", "content": "checking"},
        {"role": "tool", "tool_call_id": "c1", "content": "shipped"},
    ]

    assert latest_user_text(items) == "where is my order"
    assert latest_user_index(items) == 0


def test_latest_user_text_ignores_a_multimodal_turn() -> None:
    """Only a plain-text user turn answers "what did they ask", so a parts list is skipped."""
    items = [
        {"role": "user", "content": "describe it"},
        {"role": "user", "content": [{"type": "text", "text": "and this"}]},
    ]

    assert latest_user_text(items) == "describe it"


def test_latest_user_text_of_a_conversation_with_no_user_turn_is_none() -> None:
    """`None`, not `""`: nothing was asked, which is different from asking nothing."""
    assert latest_user_text([{"role": "system", "content": "be nice"}]) is None
    assert latest_user_index([]) is None


def test_tool_calls_reads_back_what_assistant_message_wrote() -> None:
    """The envelope has a reader here because it has a writer here: one convention, one module."""
    message = assistant_message(None, [tool_call("c1", "get_weather", '{"city": "Paris"}')])

    (call,) = tool_calls(message)

    assert call_id(call) == "c1"
    assert call_name(call) == "get_weather"
    assert call_arguments(call) == '{"city": "Paris"}'


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": "just words", "tool_calls": None},
        {"role": "assistant", "content": "just words"},
    ],
)
def test_tool_calls_reads_both_spellings_of_no_calls_as_empty(message: dict[str, object]) -> None:
    """An explicit `None` and an absent field are the same turn, so the turn loop checks a list."""
    assert tool_calls(message) == []


@pytest.mark.parametrize("arguments", ["", None])
def test_call_arguments_of_a_no_argument_call_is_an_empty_object(arguments: str | None) -> None:
    """Providers spell "no arguments" differently; that is settled here, not at each reader."""
    call = {"id": "c1", "type": "function", "function": {"name": "now", "arguments": arguments}}

    assert call_arguments(call) == "{}"
    assert call_arguments({"id": "c1", "function": {"name": "now"}}) == "{}"


def test_answered_call_id_pairs_a_result_with_its_call() -> None:
    """Matching a turn's results to its calls goes through the module that wrote the pairing."""
    assert answered_call_id(tool_result("c1", "sunny")) == "c1"


def test_parsed_arguments_reads_a_json_object() -> None:
    """A call's `function.arguments` is a JSON string, parsed into keyword arguments."""
    assert parsed_arguments('{"city": "Paris"}') == {"city": "Paris"}


@pytest.mark.parametrize("args_json", ["", "{}"])
def test_parsed_arguments_reads_no_arguments_as_no_keywords(args_json: str) -> None:
    """A tool taking no arguments is called with none, whichever way the model spelled it."""
    assert parsed_arguments(args_json) == {}


def test_parsed_arguments_rejects_malformed_json() -> None:
    """Invalid JSON raises `ValueError`; what to do about it is the caller's policy."""
    with pytest.raises(ValueError, match="invalid JSON arguments"):
        parsed_arguments("{not json")


def test_parsed_arguments_rejects_json_that_is_not_an_object() -> None:
    """Arguments name keywords, so a bare array or number is as wrong as malformed JSON."""
    with pytest.raises(ValueError, match="must be a JSON object"):
        parsed_arguments("[1, 2]")


def test_parsed_json_object_reads_a_clean_object() -> None:
    """A reply that's already valid JSON parses straight through."""
    assert parsed_json_object('{"score": 0.8}') == {"score": 0.8}


def test_parsed_json_object_ignores_surrounding_prose() -> None:
    """A model that wraps its JSON in commentary or markdown fences still parses."""
    reply = 'Sure, here is my verdict:\n```json\n{"verdict": "yes", "reason": "matches"}\n```'

    assert parsed_json_object(reply) == {"verdict": "yes", "reason": "matches"}


def test_parsed_json_object_repairs_a_trailing_comma() -> None:
    """A trailing comma before a closing bracket is stripped and retried, not just rejected."""
    reply = '{"statements": ["a", "b",], "extra": "x",}'

    assert parsed_json_object(reply) == {"statements": ["a", "b"], "extra": "x"}


def test_parsed_json_object_rejects_a_reply_with_no_object_in_it() -> None:
    """A reply with no `{...}` at all raises rather than silently returning nothing."""
    with pytest.raises(ValueError, match="contained no JSON object"):
        parsed_json_object("no json here")


def test_parsed_json_object_rejects_unrepairable_json() -> None:
    """Malformed JSON that isn't just a trailing comma raises too, carrying the reply with it."""
    with pytest.raises(ValueError, match="isn't a JSON object"):
        parsed_json_object('{"score": }')
