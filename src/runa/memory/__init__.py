"""`runa.memory`: `Memory`, durable semantic facts, scoped by `user_id`.

Layered as `Memory` (embeds text, hides vectors) over `MemoryStore` (persists/searches vectors).
Which store a bare `Memory()` gets is `runa.db`'s decision, not this module's: `memory/sqlite.py`
locally, `memory/postgres.py` when `RUNA_DATABASE_URL` points at a shared database.

`remember_from_conversation` is what `run_internal.run_loop._run_async` calls after a run to turn
the turn's exchange into zero or more remembered facts; `MemoryLike` is the contract (it and
`search`) a wholesale custom `memory=` object needs, as opposed to `Memory(store=...)`'s
narrower escape hatch of swapping just the storage backend.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from runa import db
from runa._types import ModelSettings
from runa.embeddings import DEFAULT_EMBEDDING_MODEL, embed, resolve_dimensions
from runa.tool import FunctionTool, tool

# L2 distance over the ~unit-norm vectors OpenAI's embedding models return: only a near-verbatim
# restatement falls under this, not a merely related fact -- see `Memory.remember`.
_DUPLICATE_DISTANCE = 0.1


@dataclass
class MemoryMatch:
    """One `Memory.search` result: its id, stored text, metadata, and distance to the query."""

    id: int
    text: str
    metadata: dict[str, Any] | None
    distance: float


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


class MemoryStore(Protocol):
    """The storage a `Memory` needs: add/search/delete already-embedded text, scoped by user.

    The escape hatch for `Memory(store=...)`: any object with these three async methods works,
    no inheritance required. `SQLiteMemoryStore` and `PostgresMemoryStore` satisfy it by matching
    shape; swapping in a hosted vector DB needs no change to `Memory`, `Agent`, or the lifecycle.
    """

    async def add(
        self,
        *,
        user_id: str | None,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None,
    ) -> int:
        """Store one already-embedded item for `user_id`, returning its new id."""
        ...

    async def search(
        self, *, user_id: str | None, embedding: list[float], k: int
    ) -> list[MemoryMatch]:
        """Return `user_id`'s `k` items closest to `embedding`, nearest first."""
        ...

    async def delete(self, *, user_id: str | None, memory_id: int) -> None:
        """Delete `user_id`'s item `memory_id`, if it exists."""
        ...


_EXTRACTION_PROMPT = """Below is one exchange between a user and an AI assistant. List any \
durable facts about the user worth remembering for future conversations -- stated preferences, \
recurring context, standing constraints. Not the specific question or answer itself, and not \
anything already generic/obvious. If nothing is durably worth remembering, return an empty list.

{conversation}

Reply with JSON only: {{"memories": ["...", ...]}}"""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> dict[str, Any]:
    match = _JSON_OBJECT.search(text)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


class Memory:
    """Durable semantic facts, scoped by `user_id`: `remember`/`search`/`forget`, nothing lower.

    `Memory()` means OpenAI's `text-embedding-3-small` over whichever database `runa.db` resolves:
    embeddings and vector storage are entirely internal, so callers only ever see text in and
    `MemoryMatch`es out. `store=` swaps that for a custom `MemoryStore`, unrelated to embeddings.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int | None = None,
        store: MemoryStore | None = None,
    ) -> None:
        """Configure the embedding model and, unless `store` is given, this deployment's store.

        `dimensions` only needs setting for a model not in Runa's built-in size table.
        """
        self.model = model
        self.dimensions: int = resolve_dimensions(model, dimensions)
        self._store: MemoryStore = store or db.memory_store(dimensions=self.dimensions)

    async def remember(
        self, text: str, *, user_id: str | None = None, metadata: dict[str, Any] | None = None
    ) -> int:
        """Embed `text` and store it for `user_id`, returning its new memory id.

        Skips storing, returning the existing id instead, when `user_id` already has a
        near-duplicate memory: repeated auto-extraction of the same restated fact across
        conversations doesn't pile up duplicates. `metadata` is dropped, not merged, on a skip.
        """
        (vector,) = await embed([text], model=self.model)
        existing = await self._store.search(user_id=user_id, embedding=vector, k=1)
        if existing and existing[0].distance <= _DUPLICATE_DISTANCE:
            return existing[0].id
        return await self._store.add(
            user_id=user_id, text=text, embedding=vector, metadata=metadata
        )

    async def search(
        self, query: str, *, user_id: str | None = None, k: int = 5
    ) -> list[MemoryMatch]:
        """Return `user_id`'s `k` memories closest in meaning to `query`, nearest first."""
        (vector,) = await embed([query], model=self.model)
        return await self._store.search(user_id=user_id, embedding=vector, k=k)

    async def forget(self, memory_id: int, *, user_id: str | None = None) -> None:
        """Delete `user_id`'s memory `memory_id`, if it exists."""
        await self._store.delete(user_id=user_id, memory_id=memory_id)

    def _as_tool(self) -> FunctionTool:
        """A `@tool` that searches this memory, as plain text.

        Internal: `Agent(memory="llm")` wires this in on the model's behalf; not meant to be
        built and attached to `tools=[...]` by hand -- see `Agent.__init__`'s `memory=` modes.
        """

        @tool(
            name_override="search_memory",
            description_override="Search stored memory for text relevant to a query.",
        )
        async def search_memory(query: str, k: int = 5) -> str:
            matches = await self.search(query, k=k)
            if not matches:
                return "No relevant memory found."
            return "\n\n".join(match.text for match in matches)

        return search_memory

    async def remember_from_conversation(
        self, conversation: str, *, user_id: str | None, model: Any
    ) -> list[str]:
        """Ask `model` what's durably worth remembering from `conversation`, and store it.

        Called by `run_internal.run_loop._run_async` after a run when `agent.memory` is set, not
        meant to be called directly by app code, but part of `MemoryLike`, the contract a custom
        `memory=` object must implement alongside `search`. Returns the texts it stored, empty if
        none were worth it.
        """
        prompt = _EXTRACTION_PROMPT.format(conversation=conversation)
        response = await model.get_response(
            None, [{"role": "user", "content": prompt}], ModelSettings(), [], None, []
        )
        reply = response.output[0].get("content") or ""
        candidates = _extract_json_object(reply).get("memories") or []
        stored = []
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                text = candidate.strip()
                await self.remember(text, user_id=user_id)
                stored.append(text)
        return stored


# Below `MemoryMatch`, not above: `memory/sqlite.py` returns them, so the name has to exist
# first. Re-exported because the local adapter is always importable, where
# `PostgresMemoryStore` needs the `postgres` extra.
from runa.memory.sqlite import SQLiteMemoryStore  # noqa: E402

__all__ = ["Memory", "MemoryLike", "MemoryMatch", "MemoryStore", "SQLiteMemoryStore"]
