"""`runa.memory`: `Memory`, durable semantic facts, scoped by `user_id`.

`Memory` embeds text and holds one `VectorStore` (`runa.db.vectors`) directly: a fact is a row of
`user_id`, `text` and `metadata` beside its embedding, partitioned by `user_id`. Which store a
bare `Memory()` gets is `runa.db`'s decision, not this module's.

`remember_from_conversation` is what a run calls once it finishes, to turn
the turn's exchange into zero or more remembered facts; `MemoryLike` is the contract (it and
`search`) a wholesale custom `memory=` object needs, as opposed to `Memory(store=...)`'s
narrower escape hatch of swapping just the storage backend.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from runa import db
from runa._items import item_text, parsed_json_object, user_message
from runa._log import logger
from runa._models import ModelRequest
from runa._types import RunContext
from runa.db.vectors import Column, VectorMatch, VectorSpec, VectorStore
from runa.embeddings import DEFAULT_EMBEDDING_MODEL, embed, resolve_dimensions
from runa.tool import FunctionTool, tool

# L2 distance over the ~unit-norm vectors OpenAI's embedding models return: only a near-verbatim
# restatement falls under this, not a merely related fact -- see `Memory.remember`.
_DUPLICATE_DISTANCE = 0.1

# `user_id` first so the generated tables keep the column order they already had on disk.
_COLUMNS = {
    "user_id": Column(),
    "text": Column(required=True),
    "metadata": Column(json=True),
}


def _spec(dimensions: int) -> VectorSpec:
    """The `memory_items`/`memory_vectors` spec, for vectors of `dimensions` floats."""
    return VectorSpec(
        name="memory", dimensions=dimensions, columns=_COLUMNS, partition_by="user_id"
    )


@dataclass
class MemoryMatch:
    """One `Memory.search` result: its id, stored text, metadata, and distance to the query."""

    id: int
    text: str
    metadata: dict[str, Any] | None
    distance: float


def _match(row: VectorMatch) -> MemoryMatch:
    """One stored row as the `MemoryMatch` a `Memory` hands back."""
    return MemoryMatch(
        id=row.id, text=row.payload["text"], metadata=row.payload["metadata"], distance=row.distance
    )


class MemoryLike(Protocol):
    """What `Agent(memory=...)` needs from a custom memory object, beyond `"auto"`/`"llm"`/`None`.

    `Memory` satisfies this already. Implement it yourself to replace Runa's embeddings-based
    retrieval entirely -- a different scoring method, a hosted memory service, keyword search,
    whatever -- rather than just swapping `Memory(store=...)`'s storage backend. No inheritance
    required; `Agent.__init__` accepts any object shaped like this.
    """

    async def search(self, query: str, *, user_id: str | None = None, k: int = 5) -> list[Any]:
        """Return up to `k` items relevant to `query`, most relevant first."""
        ...

    async def remember_from_conversation(
        self, conversation: str, *, user_id: str | None, model: Any
    ) -> list[str]:
        """Extract and store whatever from `conversation` is durably worth remembering.

        Returns the texts stored, empty if none were worth it. `model` is the agent's own
        resolved `Model`, handed back in case extraction wants an LLM call of its own.
        """
        ...


_EXTRACTION_PROMPT = """Below is one exchange between a user and an AI assistant. List any \
durable facts about the user worth remembering for future conversations -- stated preferences, \
recurring context, standing constraints. Not the specific question or answer itself, and not \
anything already generic/obvious. If nothing is durably worth remembering, return an empty list.

{conversation}

Reply with JSON only: {{"memories": ["...", ...]}}"""


class Memory:
    """Durable semantic facts, scoped by `user_id`: `remember`/`search`/`forget`, nothing lower.

    `Memory()` means OpenAI's `text-embedding-3-small` over whichever database `runa.db` resolves:
    embeddings and vector storage are entirely internal, so callers only ever see text in and
    `MemoryMatch`es out. `store=` swaps the storage for a custom `VectorStore`, which is handed
    payloads of `user_id`, `text` and `metadata` (a JSON-able dict or `None`), partitioned by
    `user_id`.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int | None = None,
        store: VectorStore | None = None,
    ) -> None:
        """Configure the embedding model and, unless `store` is given, this deployment's store.

        `dimensions` only needs setting for a model not in Runa's built-in size table.
        """
        self.model = model
        self.dimensions: int = resolve_dimensions(model, dimensions)
        self._store: VectorStore = store or db.vector_store(_spec(self.dimensions))

    async def remember(
        self, text: str, *, user_id: str | None = None, metadata: dict[str, Any] | None = None
    ) -> int:
        """Embed `text` and store it for `user_id`, returning its new memory id.

        Skips storing, returning the existing id instead, when `user_id` already has a
        near-duplicate memory: repeated auto-extraction of the same restated fact across
        conversations doesn't pile up duplicates. `metadata` is dropped, not merged, on a skip.
        """
        (vector,) = await embed([text], model=self.model)
        existing = await self._store.nearest(embedding=vector, k=1, partition=user_id)
        if existing and existing[0].distance <= _DUPLICATE_DISTANCE:
            return existing[0].id
        return await self._store.add(
            payload={"user_id": user_id, "text": text, "metadata": metadata}, embedding=vector
        )

    async def search(
        self, query: str, *, user_id: str | None = None, k: int = 5
    ) -> list[MemoryMatch]:
        """Return `user_id`'s `k` memories closest in meaning to `query`, nearest first.

        Scoped strictly: a memory stored with `user_id=None` is its own scope, not one every user
        can see.
        """
        (vector,) = await embed([query], model=self.model)
        found = await self._store.nearest(embedding=vector, k=k, partition=user_id)
        return [_match(row) for row in found]

    async def forget(self, memory_id: int, *, user_id: str | None = None) -> None:
        """Delete `user_id`'s memory `memory_id`, if it exists; a no-op under any other user."""
        await self._store.delete(item_id=memory_id, partition=user_id)

    def _as_tool(self) -> FunctionTool:
        """A `@tool` that searches this memory, as plain text, in the calling run's `user_id` scope.

        Internal: `Agent(memory="llm")` wires this in on the model's behalf; not meant to be
        built and attached to `tools=[...]` by hand -- see `Agent.__init__`'s `memory=` modes.
        """

        @tool(
            name_override="search_memory",
            description_override="Search stored memory for text relevant to a query.",
        )
        async def search_memory(ctx: RunContext, query: str, k: int = 5) -> str:
            matches = await self.search(query, user_id=ctx.user_id, k=k)
            if not matches:
                return "No relevant memory found."
            return "\n\n".join(match.text for match in matches)

        return search_memory

    async def remember_from_conversation(
        self, conversation: str, *, user_id: str | None, model: Any
    ) -> list[str]:
        """Ask `model` what's durably worth remembering from `conversation`, and store it.

        Called by the run itself once it finishes when `agent.memory` is set, not
        meant to be called directly by app code, but part of `MemoryLike`, the contract a custom
        `memory=` object must implement alongside `search`. Returns the texts it stored, empty if
        none were worth it.

        A reply this can't read JSON out of stores nothing and logs a warning, the way a failed
        retrieval degrades: "nothing worth remembering" and "the model
        answered with prose" are the same empty list to a caller, and an agent on `memory="auto"`
        that has quietly stopped learning is a failure nobody would otherwise report.
        """
        prompt = _EXTRACTION_PROMPT.format(conversation=conversation)
        response = await model.get_response(ModelRequest(input=[user_message(prompt)]))
        reply = item_text(response.output[0])
        try:
            candidates = parsed_json_object(reply).get("memories") or []
        except ValueError as exc:
            logger.warning("memory extraction read nothing from the model's reply: %s", exc)
            return []
        stored = []
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                text = candidate.strip()
                await self.remember(text, user_id=user_id)
                stored.append(text)
        return stored


__all__ = ["Memory", "MemoryLike", "MemoryMatch"]
