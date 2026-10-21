"""`runa.knowledge`: `Knowledge`, the application's own documents, retrieved by meaning.

Built like `runa.memory`: `Knowledge` discovers, chunks and embeds text, and holds one
`VersionedVectorStore` (`runa.db.vectors`) directly, a chunk being a row of `text` and `source`
beside its embedding. Which store a bare `Knowledge()` gets is `runa.db`'s decision.

Unlike `Memory`, `Knowledge` is application-scoped rather than `user_id`-scoped, and its source
of truth is a directory of files on disk (`app/knowledge/` by default), not calls to `remember`.

Whether the corpus has been ingested is the store's answer (`reset`/`stamp`/`version`), never
state on the `Knowledge` object: a fresh one is built per agent, and `runa serve` builds an agent
per request. `ingest` is idempotent; `search` calls it, and so does `serve.create_app` at startup.
Two pieces of process-local state remain: `_digests`, a cache of what the files on disk would
write (an empty one costs a re-read, never a re-embed), and `_ingest_locks`, which keeps one
process from embedding one corpus twice at once.

A run consults an agent's `knowledge` through `retrieve` before its first turn, so retrieval is
automatic. `KnowledgeLike` is the contract a wholesale custom
`knowledge=` object needs, as opposed to `Knowledge(store=...)`'s narrower escape hatch of
swapping just the storage backend.
"""

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from runa import _app, db
from runa._items import ConversationItem, system_message
from runa._log import logger
from runa._loop import LoopCache
from runa._models import Model
from runa._types import RunContext
from runa.context_source import ContextSource
from runa.db.vectors import Column, VectorMatch, VectorSpec, VersionedVectorStore
from runa.embeddings import DEFAULT_EMBEDDING_MODEL, embed, resolve_dimensions
from runa.tool import FunctionTool, tool

DEFAULT_KNOWLEDGE_DIR = Path("app/knowledge")

_SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".csv", ".pdf"}
_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 200

_Stats = tuple[tuple[str, int, int], ...]

# `text` first so the generated table keeps the column order it already had on disk.
_COLUMNS = {"text": Column(required=True), "source": Column(required=True)}


def _spec(dimensions: int) -> VectorSpec:
    """The `knowledge_items`/`knowledge_vectors` spec: unpartitioned, being application-scoped."""
    return VectorSpec(name="knowledge", dimensions=dimensions, columns=_COLUMNS)


@dataclass
class KnowledgeMatch:
    """One `Knowledge.search` result: its id, stored text, source file, and distance."""

    id: int
    text: str
    source: str
    distance: float


def _match(row: VectorMatch) -> KnowledgeMatch:
    """One stored row as the `KnowledgeMatch` a `Knowledge` hands back."""
    return KnowledgeMatch(
        id=row.id, text=row.payload["text"], source=row.payload["source"], distance=row.distance
    )


class KnowledgeLike(ContextSource, Protocol):
    """What `Agent(knowledge=...)` needs from a custom object, beyond `"auto"`/`"llm"`/`None`.

    `Knowledge` satisfies this already. Implement it yourself to replace Runa's own
    discover-chunk-embed pipeline entirely (a hosted retrieval service, a differently-indexed
    document store, whatever) rather than just swapping `Knowledge(store=...)`'s storage
    backend: `retrieve` finds and labels what it found, `after_turn` is usually a no-op, and
    `ingest` prepares the corpus. No inheritance required.
    """

    async def ingest(self) -> int:
        """Prepare the corpus before a request needs it, returning how many chunks were written.

        `runa serve` calls it at startup. Idempotent; one with nothing to prepare (a service
        that indexes elsewhere) returns 0.
        """
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

# One ingest of one corpus at a time. `Agent(knowledge="auto")` builds a `Knowledge` per agent and
# `runa.serve` an agent per request, so two concurrent turns arriving at a cold store would
# otherwise both rebuild it: paying for the whole corpus's embeddings twice, and -- since one's
# `reset` can land between the other's last `add` and its `stamp` -- able to leave a version
# standing over rows that were just wiped. Keyed like `_digests`, and per event loop because an
# `asyncio.Lock` belongs to the loop that first awaits it (see `_loop.py`).
_ingest_locks: LoopCache[tuple[Path, str], asyncio.Lock] = LoopCache()


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

    `store=` swaps the storage for a custom `VersionedVectorStore`, which is handed unpartitioned
    payloads of `text` and `source`.
    """

    def __init__(
        self,
        directory: str | Path = DEFAULT_KNOWLEDGE_DIR,
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int | None = None,
        store: VersionedVectorStore | None = None,
    ) -> None:
        """Configure the source directory, embedding model, and, unless `store` is given, storage.

        `dimensions` only needs setting for a model not in Runa's built-in size table.

        A relative `directory` is resolved against the project `runa.project.use` named, the
        same way the store's own file is, so an app served from outside its own directory reads
        its corpus and writes its vectors in one place. Absolute paths are left alone, which is
        how app code that means a directory somewhere else says so.
        """
        self.directory = _app.project_path(directory)
        self.model = model
        self.dimensions: int = resolve_dimensions(model, dimensions)
        self._store: VersionedVectorStore = store or db.vector_store(_spec(self.dimensions))

    @property
    def _header(self) -> str:
        """What a fingerprint hashes besides the files: the model, its size, and the chunk shape.

        Also this corpus's identity for `_digests` and `_ingest_locks`, which is the same
        question both times: two `Knowledge` objects agree about a store's contents exactly when
        they agree about these and the directory.
        """
        return f"{self.model}\0{self.dimensions}\0{_CHUNK_SIZE}\0{_CHUNK_OVERLAP}"

    def _lock(self) -> asyncio.Lock:
        """This corpus's ingest lock on the running loop, built on first use there."""
        return _ingest_locks.get((self.directory, self._header), asyncio.Lock)

    async def _fingerprint(self) -> str:
        """A hash of everything the chunks would be built from, compared by `ingest`.

        The model, its dimensions and the chunk shape are hashed alongside the files' bytes, so
        changing `model=` invalidates the corpus the way editing a file does. Computed off the
        event loop, since `search` runs this before every turn; `_digest` caches it per process,
        so the usual answer costs one `stat` per file.
        """
        return await asyncio.to_thread(_digest, self.directory, self._header)

    async def ingest(self) -> int:
        """Bring the store up to date with `self.directory`, returning how many chunks it wrote.

        Zero means there was nothing to write: the store already held this corpus, or the corpus
        is empty. Idempotent: the question is "does the store hold what I would write", answered
        by the store, so another process or replica that already ingested is never repeated.
        `search` calls this before a query and `runa serve` at startup.

        A rebuild is a full one: every chunk is re-embedded and the previous set replaced, so an
        edited or removed file never leaves stale chunks behind. A missing or empty directory
        stores an empty, fingerprinted corpus rather than raising. The version is checked again
        under the lock, so concurrent turns against a cold store rebuild it once.
        """
        fingerprint = await self._fingerprint()
        if await self._store.version() == fingerprint:
            return 0
        async with self._lock():
            if await self._store.version() == fingerprint:
                return 0
            return await self._rebuild()

    async def _rebuild(self) -> int:
        """`ingest`'s work, with its lock already held: empty the store, refill it, then stamp it.

        The fingerprint is taken before the files are read, so a file edited mid-ingest is stored
        under the older fingerprint and the next `search` rebuilds. The safe direction: a corpus
        that looks staler than it is costs one rebuild, where one that looks fresher is wrong.

        Which is also why the store is stamped at the end, after the last `add`, rather than by
        the `reset` that empties it at the start. A version is a claim that the store *holds* this
        corpus, and that only becomes true once the rows are there; stamped up front it would be
        false for as long as the embedding round trip and the inserts behind it take. Every other
        reader of this storage -- the next request's `Knowledge`, a second replica -- would
        believe the claim, skip its own ingest, and search a corpus that was still arriving, which
        is the one failure that leaves no mark on an answer. Between the `reset` and the `stamp`
        the store is unversioned instead: the state that costs a rebuild rather than a quietly
        thin answer.

        An empty corpus is logged, the way `Agent` logs a missing prompt file, and
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
        await self._store.reset()
        if chunks:
            vectors = await embed([text for text, _ in chunks], model=self.model)
            for (text, source), vector in zip(chunks, vectors, strict=True):
                await self._store.add(payload={"text": text, "source": source}, embedding=vector)
        else:
            logger.warning(
                "knowledge base is empty: no %s files under %s",
                "/".join(sorted(_SUPPORTED_EXTENSIONS)),
                self.directory,
            )
        await self._store.stamp(fingerprint)
        return len(chunks)

    async def search(self, query: str, *, k: int = 5) -> list[KnowledgeMatch]:
        """Return the `k` chunks closest in meaning to `query`, nearest first.

        Ingests first when the store doesn't already hold this corpus, so the common case
        (`Knowledge()` attached to an `Agent`) needs no manual `.ingest()` call, and an edited
        source file is picked up on the next search rather than on the next manual rebuild. That
        call is the safety net rather than the plan: a served deployment has already ingested at
        startup (`serve.create_app`), so what a turn usually pays here is one `stat` per file and
        one version read.
        """
        await self.ingest()
        (vector,) = await embed([query], model=self.model)
        return [_match(row) for row in await self._store.nearest(embedding=vector, k=k)]

    async def retrieve(self, query: str, run_context: RunContext[Any]) -> ConversationItem | None:
        """The chunks relevant to `query`, as a labeled block, or `None` if none are.

        Called by the run before its first turn when `agent.knowledge` is set; see
        `KnowledgeLike`. Application-scoped, so `run_context` doesn't narrow it.
        """
        matches = await self.search(query)
        if not matches:
            return None
        lines = "\n".join(f"- {match.text}" for match in matches)
        return system_message(f"Relevant knowledge:\n{lines}")

    async def after_turn(
        self, query: str, output: Any, run_context: RunContext[Any], model: Model
    ) -> list[str]:
        """Keep nothing: the corpus is the files under `directory`, not what a run said."""
        return []

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


__all__ = ["DEFAULT_KNOWLEDGE_DIR", "Knowledge", "KnowledgeLike", "KnowledgeMatch"]
