"""Tests for `runa._items`: the one reader of a chat-completions-shaped conversation item."""

import pytest

from runa._items import (
    content_text,
    item_text,
    latest_text,
    latest_user_index,
    latest_user_text,
    parsed_arguments,
    role_and_text,
)


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
