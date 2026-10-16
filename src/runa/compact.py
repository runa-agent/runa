"""compact.py: `Compactor`, the escape hatch behind `Agent(compact=...)`.

`compact=True` uses `default_compactor` below: past `DEFAULT_COMPACTION_TOKENS`, keep only the
most recent exchange -- a rolling window, not a summary, the simplest thing that stops `history`
from growing without bound. `compact=my_fn` swaps in any other strategy instead -- a different
threshold, an LLM summary, dropping by item count rather than tokens -- since a `Compactor` gets
full raw `items`/`usage_tokens` and decides entirely for itself whether/how to trim. The turn
loop calls it mid-run and once more when the run finishes, so the next call's history reflects
the same cut.
"""

from typing import Protocol

from runa._items import ConversationItem, latest_user_index

DEFAULT_COMPACTION_TOKENS = 200_000


class Compactor(Protocol):
    """What `Agent(compact=...)` needs beyond a plain `True`/`False`.

    No inheritance required -- any callable with this signature works. Called after every model
    response in a run with `items` (the conversation sent so far) and `usage_tokens` (the latest
    response's input plus output tokens: how large the conversation is now, not this run's
    cumulative usage, which a long tool loop inflates far past the actual context size).
    """

    def __call__(
        self, items: list[ConversationItem], usage_tokens: int
    ) -> list[ConversationItem] | None:
        """Return replacement items to compact `items` down to, or `None` to leave them as is.

        Whether `usage_tokens` warrants compacting at all, and by how much, is entirely this
        call's judgment -- Runa imposes no threshold of its own once you supply your own
        `Compactor`. Returning `items` unchanged (or `None`) is always a valid "not yet".
        """
        ...


def default_compactor(
    items: list[ConversationItem], usage_tokens: int
) -> list[ConversationItem] | None:
    """`Agent(compact=True)`'s strategy: past `DEFAULT_COMPACTION_TOKENS`, keep the latest exchange.

    Cutting at the most recent user message is always safe mid-turn: everything from there on
    (including the current exchange's own tool-call/tool-result pairs) is kept intact; only
    older, already-finished turns are dropped. Which means there is nothing for this strategy to
    drop until a conversation has a second turn: a first turn's own tool loop, however long, all
    sits after that one user message. Swap in a summarizing `Compactor` to trim that too.

    Where that message is is `_items.latest_user_index`' to say, as it is for a memory query: the
    cut lands on the latest plain-text user turn, so a multimodal one keeps the turn before it
    too rather than being cut into.
    """
    if usage_tokens <= DEFAULT_COMPACTION_TOKENS:
        return None
    cut = latest_user_index(items)
    if cut is None:
        return None
    return items[cut:]


__all__ = ["Compactor", "DEFAULT_COMPACTION_TOKENS", "default_compactor"]
