"""`runa.knowledge`: `Knowledge`, the application's own documents, retrieved by meaning.

Layered like `runa.memory`: `Knowledge` (discovers/chunks/embeds text, hides vectors) over a
`KnowledgeStore` (persists/searches vectors). Which store a bare `Knowledge()` gets is
`runa.db`'s decision: `knowledge/sqlite.py` locally, `knowledge/postgres.py` when
`RUNA_DATABASE_URL` points at a shared database.

Unlike `Memory`, `Knowledge` is application-scoped rather than `user_id`-scoped, and its source
of truth is a directory of files on disk (`app/knowledge/` by default), not calls to `remember`.
Kept deliberately separate: `Memory` is durable facts about a user, learned from conversations;
`Knowledge` is the domain documents whoever built the app put there.

`run_internal.run_loop._run_async` is what makes retrieval automatic during `run`, see its
`knowledge`/`_knowledge_block` handling. `KnowledgeLike` is the contract a wholesale custom
`knowledge=` object needs, as opposed to `Knowledge(store=...)`'s narrower escape hatch of
swapping just the storage backend.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from runa import db
from runa.embeddings import DEFAULT_EMBEDDING_MODEL, embed, resolve_dimensions
from runa.tool import FunctionTool, tool

DEFAULT_KNOWLEDGE_DIR = Path("app/knowledge")

_SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".csv", ".pdf"}
_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 200


@dataclass
class KnowledgeMatch:
    """One `Knowledge.search` result: its id, stored text, source file, and distance."""

    id: int
    text: str
    source: str
    distance: float


class KnowledgeLike(Protocol):
    """What `Agent(knowledge=...)` needs from a custom object, beyond `"auto"`/`"llm"`/`None`.

    `Knowledge` satisfies this already. Implement it yourself to replace Runa's own
    discover-chunk-embed pipeline entirely (a hosted retrieval service, a differently-indexed
    document store, whatever) rather than just swapping `Knowledge(store=...)`'s storage
    backend. No inheritance required.
    """

    async def search(self, query: str, *, k: int = 5) -> list[Any]:
        """Return up to `k` items relevant to `query`, most relevant first."""
        ...


class KnowledgeStore(Protocol):
    """The storage a `Knowledge` needs: add/search already-embedded chunks, and clear them all.

    The escape hatch for `Knowledge(store=...)`: any object with these three async methods works,
    no inheritance required. `SQLiteKnowledgeStore` and `PostgresKnowledgeStore` satisfy it by
    matching shape.
    """

    async def add(self, *, text: str, source: str, embedding: list[float]) -> int:
        """Store one already-embedded chunk, returning its new id."""
        ...

    async def search(self, *, embedding: list[float], k: int) -> list[KnowledgeMatch]:
        """Return the `k` chunks closest to `embedding`, nearest first."""
        ...

    async def clear(self) -> None:
        """Delete every stored chunk, ahead of a fresh `Knowledge.ingest()`."""
        ...


def _discover(directory: Path) -> list[Path]:
    """Every supported file under `directory`, sorted for a deterministic ingest order."""
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in _SUPPORTED_EXTENSIONS
    )


def _extract_text(path: Path) -> str:
    """Read `path`'s text: `pypdf` page text for a PDF, plain read otherwise."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(errors="ignore")


def _chunk(text: str, *, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split `text` into overlapping windows, dropping any that are blank."""
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [chunk for chunk in chunks if chunk]


class Knowledge:
    """Application/domain documents, retrieved by meaning: `search`/`ingest`, nothing lower.

    `Knowledge()` means `app/knowledge/`, OpenAI's `text-embedding-3-small`, and whichever
    database `runa.db` resolves: discovery, chunking, embeddings and vector storage are entirely
    internal, so callers only ever see a source directory in and `KnowledgeMatch`es out. Attach
    an instance to `Agent(knowledge=...)` and `run`/`run_sync` search it automatically before
    every turn; no `tools=[...]` wiring needed.
    """

    def __init__(
        self,
        directory: str | Path = DEFAULT_KNOWLEDGE_DIR,
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int | None = None,
        store: KnowledgeStore | None = None,
    ) -> None:
        """Configure the source directory, embedding model, and, unless `store` is given, storage.

        `dimensions` only needs setting for a model not in Runa's built-in size table.
        """
        self.directory = Path(directory)
        self.model = model
        self.dimensions: int = resolve_dimensions(model, dimensions)
        self._store: KnowledgeStore = store or db.knowledge_store(dimensions=self.dimensions)
        self._ingested = False

    async def ingest(self) -> int:
        """Rebuild the knowledge base from `self.directory`, returning how many chunks it stored.

        A full rebuild every time: discovers supported files, extracts their text, chunks it,
        embeds every chunk, clears whatever was stored before, and stores the fresh set, so
        editing or removing a source file and calling `ingest()` again never leaves stale chunks
        behind, with no file-tracking table to keep in sync. Safe to call when `self.directory`
        doesn't exist or has no supported files: stores nothing rather than raising.
        """
        chunks = [
            (chunk, str(path))
            for path in _discover(self.directory)
            for chunk in _chunk(_extract_text(path))
        ]
        await self._store.clear()
        if chunks:
            vectors = await embed([text for text, _ in chunks], model=self.model)
            for (text, source), vector in zip(chunks, vectors, strict=True):
                await self._store.add(text=text, source=source, embedding=vector)
        self._ingested = True
        return len(chunks)

    async def search(self, query: str, *, k: int = 5) -> list[KnowledgeMatch]:
        """Return the `k` chunks closest in meaning to `query`, nearest first.

        Ingests `self.directory` first if this instance hasn't ingested yet, so the common case
        (`Knowledge()` attached to an `Agent`) needs no manual `.ingest()` call.
        """
        if not self._ingested:
            await self.ingest()
        (vector,) = await embed([query], model=self.model)
        return await self._store.search(embedding=vector, k=k)

    def _as_tool(self) -> FunctionTool:
        """A `@tool` that searches this knowledge base, as plain text.

        Internal: `Agent(knowledge="llm")` wires this in on the model's behalf; not meant to be
        built and attached to `tools=[...]` by hand, see `Agent.__init__`'s `knowledge=` modes.
        """

        @tool(
            name_override="search_knowledge",
            description_override="Search the knowledge base for text relevant to a query.",
        )
        async def search_knowledge(query: str, k: int = 5) -> str:
            matches = await self.search(query, k=k)
            if not matches:
                return "No relevant knowledge found."
            return "\n\n".join(match.text for match in matches)

        return search_knowledge


# Below `KnowledgeMatch`, not above: `knowledge/sqlite.py` returns them, so the name has to
# exist first. Re-exported because the local adapter is always importable, where
# `PostgresKnowledgeStore` needs the `postgres` extra.
from runa.knowledge.sqlite import SQLiteKnowledgeStore  # noqa: E402

__all__ = [
    "DEFAULT_KNOWLEDGE_DIR",
    "Knowledge",
    "KnowledgeLike",
    "KnowledgeMatch",
    "KnowledgeStore",
    "SQLiteKnowledgeStore",
]
