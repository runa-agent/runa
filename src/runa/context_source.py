"""context_source.py: `ContextSource`, what a run asks of its agent's `memory` and `knowledge`.

A run consults each source twice: before the first turn, for a block of context about the user's
message, and once the run completes, with the exchange it produced. What a source searches, how it
scopes and labels what it found, and what it keeps from an exchange are its own. Tracing each call,
paying for it, and finishing a run without a source that didn't answer belong to the run: a model
call through the `model` it's handed and an `runa.embeddings.embed` request made inside either
method are charged to the run, in its usage and its trace.

`runa.memory.MemoryLike` and `runa.knowledge.KnowledgeLike` are this contract as each setting
names it; a custom `memory=`/`knowledge=` object implements those.
"""

from typing import Any, Protocol

from runa._items import ConversationItem
from runa._models import Model
from runa._types import RunContext


class ContextSource(Protocol):
    """Something a run consults before its first turn and tells about the exchange after its last.

    Raise `runa.exceptions.ServiceUnavailable` to be skipped: the run records it on the source's
    span, logs it and carries on without this source. Any other exception ends the run, since the
    empty block it would leave reads like a corpus with nothing relevant. A call through
    `runa.embeddings.embed` or the `model` a source is handed already raises it for a service
    that doesn't answer; a source with a backend of its own translates that backend's failures.
    """

    async def context(self, query: str, run_context: RunContext[Any]) -> ConversationItem | None:
        """The block to put before the user's message `query`, or `None` when nothing is relevant.

        `run_context.user_id` is who the run is for, for a source scoped by user.
        """
        ...

    async def after_turn(
        self, query: str, output: Any, run_context: RunContext[Any], model: Model
    ) -> list[str]:
        """Keep what's worth keeping from a completed exchange, returning the texts kept.

        `query` is the user's message and `output` the run's final output. `model` is the
        agent's own model, for a source that wants an LLM call of its own; each call through it
        is one of the run's, in its trace, its hooks and its usage. A source that learns nothing
        from conversations returns `[]`.
        """
        ...


__all__ = ["ContextSource"]
