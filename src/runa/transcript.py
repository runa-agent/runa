"""transcript.py: `Transcript`, one run's conversation, defined once.

What a run started from, what the model is sent, what it produced and what is left to save: the
turn loop works on a `Transcript`, a paused `RunState` carries one, and the caller's `Run` keeps
one. Every rule about the conversation is kept behind the methods below -- what reaches the
model, what counts as history, what a session stores, how compaction cuts it -- and nothing
outside this module writes to its lists. A fact the conversation starts carrying is a field here,
and nowhere else: a `RunState` serializes every field as it is.
"""

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field

from runa._items import ConversationItem, latest_user_index, latest_user_text
from runa.compact import Compactor
from runa.tool import ToolCall


@dataclass
class Transcript:
    """One run's conversation: where it started, what the model is sent, and what it produced.

    Four lists, written only by the methods below:

    * `prior`: the earlier turns a session already holds. Empty without a session.
    * `input`: what this run was given. Without a session that is the earlier conversation and
      the new turn; with one, the new turn alone.
    * `items`: what the model is sent: `prior` and `input`, then everything the run generated,
      with any retrieved context before the latest user turn, as compaction last cut it.
    * `generated`: what the run produced, in order.

    `history()` is the conversation after the run and `unsaved()` the part of it a session has yet
    to store, both read once the run finishes, whether that's straight away or after resumes.
    """

    _prior: list[ConversationItem] = field(default_factory=list)
    _input: list[ConversationItem] = field(default_factory=list)
    _items: list[ConversationItem] = field(default_factory=list)
    _generated: list[ConversationItem] = field(default_factory=list)
    _tool_calls: list[ToolCall] = field(default_factory=list)
    _compacted_history: list[ConversationItem] | None = None

    @classmethod
    def start(
        cls, input: Sequence[ConversationItem], prior: Sequence[ConversationItem] = ()
    ) -> Transcript:
        """A run's transcript before its first turn: `input`, after the session's `prior` turns."""
        return cls(_prior=list(prior), _input=list(input), _items=[*prior, *input])

    @property
    def items(self) -> tuple[ConversationItem, ...]:
        """Everything the model is sent next; a paused run's ends with the message that paused."""
        return tuple(self._items)

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """Every tool call this run answered, in the order they finished (see `ToolCall`)."""
        return tuple(self._tool_calls)

    def latest_user_text(self) -> str | None:
        """The latest user text this run was given: what memory and knowledge are searched for."""
        return latest_user_text(self._input)

    def record(self, *items: ConversationItem) -> None:
        """Append what the run produced: the model is sent it from now on, and it is history."""
        self._items.extend(items)
        self._generated.extend(items)

    def record_call(self, call: ToolCall) -> None:
        """Note one tool call the run answered."""
        self._tool_calls.append(call)

    def add_context(self, *blocks: ConversationItem) -> None:
        """Send the model `blocks` right before the latest user turn, for this run only.

        Context is never history and never saved: the next run retrieves its own rather than
        inheriting this one's.
        """
        at = latest_user_index(self._items)
        assert at is not None, "context is added for a user turn the transcript holds"
        self._items[at:at] = blocks

    def compact(self, compactor: Compactor | None, tokens: int) -> int | None:
        """Cut what the model is sent with `compactor`.

        Returns how many items that dropped, or `None` if it changed nothing.
        """
        replacement = _cut(compactor, self._items, tokens)
        if replacement is None:
            return None
        dropped = len(self._items) - len(replacement)
        self._items = replacement
        return dropped

    def compact_history(self, compactor: Compactor | None, tokens: int) -> int | None:
        """Cut `history()` with `compactor`, so the next run starts from the same cut.

        Returns how many items that dropped, or `None` if it changed nothing. Once it has changed
        something, `history()` is the cut conversation, and a session must replace what it holds
        with it rather than append `unsaved()`.
        """
        history = self.history()
        replacement = _cut(compactor, history, tokens)
        if replacement is None:
            return None
        self._compacted_history = replacement
        return len(history) - len(replacement)

    def history(self) -> list[ConversationItem]:
        """The conversation after this run: `prior`, `input`, then what the run generated.

        Or what `compact_history` cut that to. What `Agent.run` writes back to `Agent.history` on
        a session-less run, and what a session is rewritten with after a compaction.
        """
        if self._compacted_history is not None:
            return list(self._compacted_history)
        return [*self._prior, *self._input, *self._generated]

    def unsaved(self) -> list[ConversationItem]:
        """What a session has yet to store of this run: its input, then what it generated."""
        return [*self._input, *self._generated]

    def copy(self) -> Transcript:
        """A transcript holding new lists with the same items, for a run to append to.

        A resumed run continues from a copy, so the paused `RunState` it came from still reads
        as it was paused, however far the resumed run gets.
        """
        compacted = self._compacted_history
        return Transcript(
            _prior=list(self._prior),
            _input=list(self._input),
            _items=list(self._items),
            _generated=list(self._generated),
            _tool_calls=list(self._tool_calls),
            _compacted_history=None if compacted is None else list(compacted),
        )


def _cut(
    compactor: Compactor | None, items: list[ConversationItem], tokens: int
) -> list[ConversationItem] | None:
    """`compactor`'s replacement for `items`, or `None` if there is none or it changed nothing.

    The compactor is handed a copy, so what it returns is the only way it changes anything, and a
    rewrite that keeps every item (truncating long tool output, say) counts as a change.
    """
    if compactor is None:
        return None
    replacement = compactor(copy.deepcopy(items), tokens)
    if replacement is None or replacement == items:
        return None
    return list(replacement)


__all__ = ["Transcript"]
