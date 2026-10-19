"""Tests for `Transcript`: one run's conversation, and the JSON it carries through a pause."""

from typing import Any

from pydantic import TypeAdapter

from runa.tool import ToolCall
from runa.transcript import Transcript

_HI = {"role": "user", "content": "hi"}
_HELLO = {"role": "assistant", "content": "hello"}
_EARLIER = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "noted"}]


def _transcript() -> Transcript:
    transcript = Transcript.start([_HI])
    transcript.record(_HELLO)
    transcript.record_call(ToolCall(name="now", arguments="{}", output="noon"))
    return transcript


def test_a_json_round_trip_keeps_the_conversation_and_the_tool_calls() -> None:
    """What a paused run's blob carries: every field, so a restored run reports every call."""
    transcript = Transcript.start([_HI], prior=_EARLIER)
    transcript.record(_HELLO)
    transcript.record_call(ToolCall(name="now", arguments="{}", output="noon"))
    adapter = TypeAdapter(Transcript)

    restored = adapter.validate_python(adapter.dump_python(transcript, mode="json"))

    assert restored.history() == transcript.history()
    assert restored.unsaved() == transcript.unsaved()
    assert restored.items == transcript.items
    assert restored.tool_calls == transcript.tool_calls


def test_a_session_less_run_starts_from_its_input_and_saves_nothing_else() -> None:
    """Without a session, `input` is the whole conversation so far: history is it plus output."""
    transcript = Transcript.start([*_EARLIER, _HI])
    transcript.record(_HELLO)

    assert transcript.items == (*_EARLIER, _HI, _HELLO)
    assert transcript.history() == [*_EARLIER, _HI, _HELLO]


def test_a_session_run_sends_the_prior_turns_but_saves_only_its_own() -> None:
    """The session already holds `prior`, so only the new turn and its output are unsaved."""
    transcript = Transcript.start([_HI], prior=_EARLIER)
    transcript.record(_HELLO)

    assert transcript.items == (*_EARLIER, _HI, _HELLO)
    assert transcript.unsaved() == [_HI, _HELLO]
    assert transcript.history() == [*_EARLIER, _HI, _HELLO]


def test_context_goes_before_the_latest_user_turn_and_is_never_history() -> None:
    """Retrieved context is for the model only: neither `history()` nor a session sees it."""
    memory = {"role": "system", "content": "Relevant memories: ..."}
    knowledge = {"role": "system", "content": "Relevant knowledge: ..."}
    transcript = Transcript.start([_HI], prior=_EARLIER)

    transcript.add_context(memory, knowledge)
    transcript.record(_HELLO)

    assert transcript.items == (*_EARLIER, memory, knowledge, _HI, _HELLO)
    assert transcript.unsaved() == [_HI, _HELLO]
    assert transcript.history() == [*_EARLIER, _HI, _HELLO]


def test_latest_user_text_reads_this_runs_input_not_the_prior_turns() -> None:
    """What memory and knowledge are searched for: the turn this run was given."""
    assert Transcript.start([_HI], prior=_EARLIER).latest_user_text() == "hi"
    assert Transcript.start([], prior=_EARLIER).latest_user_text() is None


def _truncate(items: list[Any], usage_tokens: int) -> list[Any]:
    """A compactor that rewrites every item without dropping any."""
    return [{**item, "content": item["content"][:2]} for item in items]


def test_a_compaction_that_rewrites_without_dropping_still_takes_effect() -> None:
    """A changed list is a change, whatever its length: truncating output counts."""
    transcript = Transcript.start([{"role": "user", "content": "a long question"}])

    assert transcript.compact(_truncate, 0) == 0
    assert transcript.items == ({"role": "user", "content": "a "},)


def test_a_compaction_that_returns_none_or_the_same_items_changes_nothing() -> None:
    """Both are a compactor's "not yet", reported as `None`."""
    transcript = Transcript.start([_HI])

    assert transcript.compact(lambda items, usage_tokens: None, 0) is None
    assert transcript.compact(lambda items, usage_tokens: items, 0) is None
    assert transcript.compact(None, 0) is None
    assert transcript.items == (_HI,)


def test_a_compactor_cannot_change_the_transcript_by_mutating_what_it_was_handed() -> None:
    """It gets a copy: what it returns is the only thing that counts."""

    def mutate(items: list[Any], usage_tokens: int) -> None:
        items.clear()

    transcript = Transcript.start([_HI])

    assert transcript.compact(mutate, 0) is None
    assert transcript.items == (_HI,)


def test_compacting_the_history_replaces_it_whole() -> None:
    """After a cut, `history()` is the cut conversation, prior turns and output included."""
    transcript = Transcript.start([_HI], prior=_EARLIER)
    transcript.record(_HELLO)

    dropped = transcript.compact_history(lambda items, usage_tokens: items[2:], 0)

    assert dropped == 2
    assert transcript.history() == [_HI, _HELLO]


def test_compacting_what_the_model_is_sent_leaves_the_history_alone() -> None:
    """Mid-run cuts are for the next model call; the history is cut once, when the run ends."""
    transcript = Transcript.start([_HI], prior=_EARLIER)

    transcript.compact(lambda items, usage_tokens: items[2:], 0)

    assert transcript.items == (_HI,)
    assert transcript.history() == [*_EARLIER, _HI]


def test_a_copy_is_appended_to_without_touching_the_original() -> None:
    """What a resumed run continues from: the paused state's transcript stays as it was paused."""
    transcript = _transcript()
    copy = transcript.copy()

    copy.record({"role": "assistant", "content": "more"})
    copy.record_call(ToolCall(name="later", arguments="{}", output="night"))
    copy.compact(lambda items, usage_tokens: items[1:], 0)

    assert transcript == _transcript()
