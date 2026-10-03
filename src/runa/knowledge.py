"""knowledge.py: `Knowledge`, application/domain documents retrieved into the run automatically.

Layered like `memory.py`: `Knowledge` (discovers/chunks/embeds text, hides vectors) over a
`KnowledgeStore` (persists/searches vectors) -- `SQLiteKnowledgeStore` is the default, sharing the
same connect-and-create-if-missing `db/runa.db` file `SQLiteSession`/`Memory` use (`db/sqlite.py`).
Unlike `Memory`, `Knowledge` is application-scoped, not `user_id`-scoped, and its source of truth
is a directory of files on disk (`app/knowledge/` by default), not calls to `remember`.

Kept deliberately separate from `Memory`: `Memory` is durable facts about a user/agent, learned
from conversations; `Knowledge` is the application's own domain documents, put there by whoever
built the app. `run_internal.run_loop._run_async` is what makes retrieval automatic during `run`
-- see its `knowledge`/`_knowledge_block` handling, the same shape as its `memory` handling.
`KnowledgeLike` is the contract a wholesale custom `knowledge=` object needs, as opposed to
`Knowledge(store=...)`'s narrower escape hatch of swapping just the storage backend.
"""

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from runa.db.sqlite import DEFAULT_DB_PATH
from runa.db.sqlite import connect as _connect_db
from runa.db.sqlite import pack_vector as _pack
from runa.embeddings import DEFAULT_EMBEDDING_MODEL, embed, resolve_dimensions
from runa.tool import FunctionTool, tool

_ITEMS_TABLE = "knowledge_items"
_VECTORS_TABLE = "knowledge_vectors"

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
    discover-chunk-embed pipeline entirely -- a hosted retrieval service, a differently-indexed
    document store, whatever -- rather than just swapping `Knowledge(store=...)`'s storage
    backend. No inheritance required.
    """

    async def search(self, query: str, *, k: int = 5) -> list[Any]:
        """Return up to `k` items relevant to `query`, most relevant first."""
        ...


class KnowledgeStore(Protocol):
    """The storage a `Knowledge` needs: add/search already-embedded chunks, and clear them all.

    The escape hatch for `Knowledge(store=...)`: any object with these three async methods works,
    no inheritance required. `SQLiteKnowledgeStore` is the default.
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


def _ddl(dimensions: int) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {_ITEMS_TABLE} (
        id INTEGER PRIMARY KEY,
        text TEXT NOT NULL,
        source TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS {_VECTORS_TABLE} USING vec0(
        embedding float[{dimensions}]
    );
    """


class SQLiteKnowledgeStore:
    """The default `KnowledgeStore`: `db/runa.db`'s `knowledge_items`/`knowledge_vectors` tables."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, *, dimensions: int) -> None:
        """Store where this store's chunks/vectors live and the embedding size its table expects."""
        self.db_path = Path(db_path)
        self.dimensions = dimensions

    def _connect(self) -> sqlite3.Connection:
        return _connect_db(self.db_path, _ddl(self.dimensions), load_vec=True)

    async def add(self, *, text: str, source: str, embedding: list[float]) -> int:
        """Store one already-embedded chunk, returning its new id."""
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                f"INSERT INTO {_ITEMS_TABLE} (text, source) VALUES (?, ?)",
                (text, source),
            )
            item_id = cursor.lastrowid
            assert item_id is not None
            conn.execute(
                f"INSERT INTO {_VECTORS_TABLE} (rowid, embedding) VALUES (?, ?)",
                (item_id, _pack(embedding)),
            )
            conn.commit()
        return item_id

    async def search(self, *, embedding: list[float], k: int) -> list[KnowledgeMatch]:
        """Return the `k` chunks closest to `embedding`, nearest first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"""
                SELECT items.id, items.text, items.source, vectors.distance
                FROM {_VECTORS_TABLE} AS vectors
                JOIN {_ITEMS_TABLE} AS items ON items.id = vectors.rowid
                WHERE vectors.embedding MATCH ? AND vectors.k = ?
                ORDER BY vectors.distance
                """,
                (_pack(embedding), k),
            ).fetchall()
        return [
            KnowledgeMatch(id=item_id, text=text, source=source, distance=distance)
            for item_id, text, source, distance in rows
        ]

    async def clear(self) -> None:
        """Delete every stored chunk, ahead of a fresh `Knowledge.ingest()`."""
        with closing(self._connect()) as conn:
            conn.execute(f"DELETE FROM {_VECTORS_TABLE}")
            conn.execute(f"DELETE FROM {_ITEMS_TABLE}")
            conn.commit()


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

    `Knowledge()` means `app/knowledge/`, `db/runa.db`, `sqlite-vec`, OpenAI's
    `text-embedding-3-small` -- discovery, chunking, embeddings, and vector storage are entirely
    internal; callers only ever see a source directory in and `KnowledgeMatch`es out. Attach an
    instance to `Agent(knowledge=...)` and `run`/`run_sync` search it automatically before every
    turn; no `tools=[...]` wiring needed.
    """

    def __init__(
        self,
        directory: str | Path = DEFAULT_KNOWLEDGE_DIR,
        *,
        db_path: str | Path = DEFAULT_DB_PATH,
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
        self._store: KnowledgeStore = store or SQLiteKnowledgeStore(
            db_path, dimensions=self.dimensions
        )
        self._ingested = False

    async def ingest(self) -> int:
        """Rebuild the knowledge base from `self.directory`, returning how many chunks it stored.

        A full rebuild every time: discovers supported files, extracts their text, chunks it,
        embeds every chunk, clears whatever was stored before, and stores the fresh set -- so
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
        built and attached to `tools=[...]` by hand -- see `Agent.__init__`'s `knowledge=` modes.
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


__all__ = ["Knowledge", "KnowledgeLike", "KnowledgeMatch", "KnowledgeStore"]
