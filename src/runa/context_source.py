"""context_source.py: `ContextSource`, what a run asks of its agent's `memory` and `knowledge`.

A run consults each source twice: before the first turn, for a block of context about the user's
message, and once the run completes, with the exchange it produced. What a source searches, how it
scopes and labels what it found, and what it keeps from an exchange are its own. Tracing each call
and finishing a run without a source that didn't answer belong to the run.

`runa.memory.MemoryLike` and `runa.knowledge.KnowledgeLike` are this contract as each setting
names it; a custom `memory=`/`knowledge=` object implements those.
"""

from typing import Any, Protocol

from runa._items import ConversationItem
from runa._models import Model
from runa._types import RunContext


class ContextSource(Protocol):
    """Something a run consults before its first turn and tells about the exchange after its last.

    A method that raises `UserError`, `AttributeError`, `NameError` or `TypeError` ends the run:
    those are a misconfiguration or a bug, and an empty block would hide them. Any other exception
    is the backend not answering, and the run carries on without this source.
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
        agent's own resolved `Model`, for a source that wants an LLM call of its own. A source
        that learns nothing from conversations returns `[]`.
        """
        ...


__all__ = ["ContextSource"]
