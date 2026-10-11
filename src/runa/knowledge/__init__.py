"""`runa.knowledge`: `Knowledge`, the application's own documents, retrieved by meaning.

Layered like `runa.memory`: `Knowledge` (discovers/chunks/embeds text, hides vectors) over a
`KnowledgeStore` (persists/searches vectors, in `knowledge/store.py` and re-exported here). Which
store a bare `Knowledge()` gets is `runa.db`'s decision: `knowledge/sqlite.py` locally,
`knowledge/postgres.py` when `RUNA_DATABASE_URL` points at a shared database.

Unlike `Memory`, `Knowledge` is application-scoped rather than `user_id`-scoped, and its source
of truth is a directory of files on disk (`app/knowledge/` by default), not calls to `remember`.
Kept deliberately separate: `Memory` is durable facts about a user, learned from conversations;
`Knowledge` is the domain documents whoever built the app put there.

Whether the corpus has been ingested is the store's answer, not a flag on the `Knowledge` object:
`Agent(knowledge="auto")` builds a fresh `Knowledge` per agent and `runa.serve` builds an agent
per request, so instance state would have every request re-embed the whole directory into the one
shared store. `KnowledgeStore.reset`/`.version` is where that decision lives instead. The one
piece of process-local state here, `_digests`, answers a different question -- "what would I
write", a pure function of the files on disk -- so an empty cache costs a re-read and never a
re-embed.

`run_internal.run_loop._run_async` is what makes retrieval automatic during `run`, see its
`knowledge`/`_knowledge_block` handling. `KnowledgeLike` is the contract a wholesale custom
`knowledge=` object needs, as opposed to `Knowledge(store=...)`'s narrower escape hatch of
swapping just the storage backend.
"""

import asyncio
import hashlib
from pathlib import Path
from typing import Any, Protocol

from runa import db
from runa.embeddings import DEFAULT_EMBEDDING_MODEL, embed, resolve_dimensions
from runa.knowledge.sqlite import SQLiteKnowledgeStore
from runa.knowledge.store import KnowledgeMatch, KnowledgeStore
from runa.lifecycle import logger
from runa.tool import FunctionTool, tool

DEFAULT_KNOWLEDGE_DIR = Path("app/knowledge")

_SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".csv", ".pdf"}
_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 200

_Stats = tuple[tuple[str, int, int], ...]


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


def _discover(directory: Path) -> list[Path]:
    """Every supported file under `directory`, sorted for a deterministic ingest order."""
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in _SUPPORTED_EXTENSIONS
    )


# Per process, not per instance: a `Knowledge` is built per agent and an agent per request, so an
# instance-scoped cache would never be read twice. Keyed by everything the digest is built from
# besides the files, so two `Knowledge` objects over one directory with different `model=`s don't
# read each other's entry.
_digests: dict[tuple[Path, str], tuple[_Stats, str]] = {}


def _stats(paths: list[Path]) -> _Stats:
    """Each file's path, size and mtime: enough to tell that none of them has been edited."""
    stats = []
    for path in paths:
        info = path.stat()
        stats.append((str(path), info.st_size, info.st_mtime_ns))
    return tuple(stats)


def _digest(directory: Path, header: str) -> str:
    """Hash `header` and every supported file under `directory`, as `Knowledge._fingerprint` does.

    Contents, not `(size, mtime)`: a container build or a fresh checkout rewrites every
    timestamp, so a timestamp-keyed fingerprint would re-embed the whole corpus on each deploy.

    Which is why the timestamps are only a cache key here, never the hash: an unchanged
    `(path, size, mtime)` for every file means the digest this process computed last time still
    stands, and a fresh checkout invalidates the cache, re-reads the corpus once, and arrives at
    the same digest -- so it still doesn't re-embed. Losing this cache costs one hash; it can
    never cost a rebuild, and it can never skip one a changed file calls for.
    """
    paths = _discover(directory)
    stats = _stats(paths)
    cached = _digests.get((directory, header))
    if cached is not None and cached[0] == stats:
        return cached[1]
    digest = hashlib.sha256(header.encode())
    for path in paths:
        # Relative, so the same corpus checked out at another path is the same corpus.
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    fingerprint = digest.hexdigest()
    _digests[(directory, header)] = (stats, fingerprint)
    return fingerprint


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

        A relative `directory` is resolved against the project `runa.db.use_project` named, the
        same way the store's own file is, so an app served from outside its own directory reads
        its corpus and writes its vectors in one place. Absolute paths are left alone, which is
        how app code that means a directory somewhere else says so.
        """
        self.directory = db.project_path(directory)
        self.model = model
        self.dimensions: int = resolve_dimensions(model, dimensions)
        self._store: KnowledgeStore = store or db.knowledge_store(dimensions=self.dimensions)

    async def _fingerprint(self) -> str:
        """A hash of everything the stored chunks would be built from, as `search` compares it.

        Internal: the public surface stays `search`/`ingest`, and a version is the store's
        vocabulary, not an application's.

        The model, its dimensions and the chunk shape are in the hash alongside the files' bytes:
        chunks embedded by another model aren't comparable to this one's query vectors, so
        changing `model=` has to invalidate the corpus exactly the way editing a file does.

        Off the event loop, like a sync `@tool` or `@guardrail`: walking and reading the corpus
        is blocking I/O whose cost grows with it, and `search` runs this before every turn of
        every knowledge-backed run. `_digest` caches it per process, so the usual answer costs
        one `stat` per file rather than a full read.
        """
        header = f"{self.model}\0{self.dimensions}\0{_CHUNK_SIZE}\0{_CHUNK_OVERLAP}"
        return await asyncio.to_thread(_digest, self.directory, header)

    async def ingest(self) -> int:
        """Rebuild the knowledge base from `self.directory`, returning how many chunks it stored.

        A full rebuild every time: discovers supported files, extracts their text, chunks it,
        embeds every chunk, empties whatever was stored before, and stores the fresh set under
        this corpus's fingerprint, so editing or removing a source file never leaves stale chunks
        behind, with no file-tracking table to keep in sync. Safe to call when `self.directory`
        doesn't exist or has no supported files: stores nothing rather than raising, and an empty
        corpus is still a fingerprinted one.

        The fingerprint is taken before the files are read, so a file edited mid-ingest is stored
        under the older fingerprint and the next `search` rebuilds. The safe direction: a corpus
        that looks staler than it is costs one rebuild, where one that looks fresher is wrong.

        An empty corpus is logged, the way `agent._load_prompt` logs a missing prompt file, and
        for the same reason: retrieval is the one step whose failure leaves no mark on the output
        (an empty knowledge block reads exactly like a corpus with nothing relevant in it), so an
        agent that asked for `knowledge=` and found nothing to search has to say so somewhere.
        Once per corpus, not per turn: an empty directory is fingerprinted like any other, so the
        next `search` finds the store already current.
        """
        fingerprint = await self._fingerprint()
        chunks = [
            (chunk, str(path))
            for path in _discover(self.directory)
            for chunk in _chunk(_extract_text(path))
        ]
        await self._store.reset(version=fingerprint)
        if not chunks:
            logger.warning(
                "knowledge base is empty: no %s files under %s",
                "/".join(sorted(_SUPPORTED_EXTENSIONS)),
                self.directory,
            )
            return 0
        vectors = await embed([text for text, _ in chunks], model=self.model)
        for (text, source), vector in zip(chunks, vectors, strict=True):
            await self._store.add(text=text, source=source, embedding=vector)
        return len(chunks)

    async def search(self, query: str, *, k: int = 5) -> list[KnowledgeMatch]:
        """Return the `k` chunks closest in meaning to `query`, nearest first.

        Ingests first when the store doesn't already hold this corpus, so the common case
        (`Knowledge()` attached to an `Agent`) needs no manual `.ingest()` call, and an edited
        source file is picked up on the next search rather than on the next manual rebuild.

        The question asked is "does the store hold what I would write", not "have I ingested
        yet": the answer is the store's, so the second `Knowledge` over an already-ingested
        corpus searches it instead of rebuilding it -- which is what a per-request `Agent`
        (`runa.serve`) and every replica after the first are.
        """
        if await self._store.version() != await self._fingerprint():
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


# `SQLiteKnowledgeStore` is re-exported because the local adapter is always importable, where
# `PostgresKnowledgeStore` needs the `postgres` extra.
__all__ = [
    "DEFAULT_KNOWLEDGE_DIR",
    "Knowledge",
    "KnowledgeLike",
    "KnowledgeMatch",
    "KnowledgeStore",
    "SQLiteKnowledgeStore",
]
