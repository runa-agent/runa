"""Tests for `runa.knowledge.Knowledge`, everything above the `VersionedVectorStore` it holds.

Discovery, chunking, PDF extraction, the payload it stores, lazy ingest, and the search tool it
hands a model. The storage itself is held to `tests/contracts/vector.py`, driven in
`test_vectors.py` and `test_postgres.py`.
"""

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest
from helpers import run as run_awaitable

from runa import knowledge as knowledge_module
from runa._types import RunContext
from runa.db.vectors import VectorMatch, VersionedVectorStore
from runa.db.vectors.sqlite import SQLiteVectorStore
from runa.exceptions import UserError
from runa.knowledge import DEFAULT_KNOWLEDGE_DIR, Knowledge, KnowledgeMatch
from runa.project import use

_DIMENSIONS = 4


def _sqlite_store(tmp_path: Path) -> SQLiteVectorStore:
    """The local store a bare `Knowledge()` would get, kept in this test's own `tmp_path`."""
    return SQLiteVectorStore(knowledge_module._spec(_DIMENSIONS), tmp_path / "runa.db")


_embedded: list[str] = []


@pytest.fixture(autouse=True)
def _fake_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real OpenAI call with a lookup table, so tests never hit the network.

    Each text maps to a fixed 4-dimensional vector; unmapped text embeds to the origin, which is
    never asserted on directly, just used to keep the call total. Every text asked for is recorded
    in `_embedded`, which is what the tests about re-ingesting assert on: the cost an avoided
    rebuild avoids is the embedding of every chunk.
    """
    vectors = {
        "cats are great pets": [1.0, 0.0, 0.0, 0.0],
        "dogs are loyal companions": [0.9, 0.1, 0.0, 0.0],
        "the stock market fell today": [0.0, 0.0, 0.0, 1.0],
        "query about pets": [1.0, 0.0, 0.0, 0.0],
    }
    _embedded.clear()

    async def fake_embed(texts: list[str], *, model: str = "") -> list[list[float]]:
        _embedded.extend(texts)
        return [vectors.get(text, [0.0, 0.0, 0.0, 0.0]) for text in texts]

    monkeypatch.setattr(knowledge_module, "embed", fake_embed)


@pytest.fixture(autouse=True)
def _fresh_digest_cache() -> None:
    """Start each test with no cached corpus digests, so none of them depends on an earlier one."""
    knowledge_module._digests.clear()


def _knowledge(tmp_path: Path, directory: Path | None = None) -> Knowledge:
    return Knowledge(
        directory if directory is not None else tmp_path / "knowledge",
        dimensions=_DIMENSIONS,
        store=_sqlite_store(tmp_path),
    )


def test_default_configuration_needs_no_arguments() -> None:
    """`Knowledge()` means `app/knowledge/` + `db/runa.db` + `sqlite-vec` + the default model."""
    know = Knowledge()

    assert know.directory == DEFAULT_KNOWLEDGE_DIR
    assert know.model == "text-embedding-3-small"
    assert know.dimensions == 1536
    assert isinstance(know._store, SQLiteVectorStore)
    assert str(know._store.db_path) == str(Path("db/runa.db"))


def test_explicit_directory_overrides_the_default() -> None:
    """`Knowledge(path)` uses `path` as the source directory instead of `app/knowledge/`."""
    know = Knowledge("some/other/path")

    assert know.directory == Path("some/other/path")


def test_a_project_moves_the_source_directory(tmp_path: Path) -> None:
    """A relative `directory` follows `project.use`, the way the store's own file does.

    `Knowledge` is built inside `Agent.__init__`, which has no root to hand it, so an app served
    from outside its own directory (`runa serve`, `create_app(root)`) read its corpus from the
    cwd while writing its vectors under the root: an empty corpus, and silent, since an empty
    knowledge block reads exactly like a corpus with nothing relevant in it.
    """
    corpus = tmp_path / "app" / "knowledge"
    corpus.mkdir(parents=True)

    use(tmp_path)

    assert Knowledge().directory == corpus


def test_an_absolute_directory_ignores_the_project(tmp_path: Path) -> None:
    """An absolute path is already an answer, and is how app code names a corpus elsewhere."""
    use(tmp_path)

    assert Knowledge("/srv/shared-corpus").directory == Path("/srv/shared-corpus")


def test_unknown_model_without_dimensions_raises() -> None:
    """A model Runa doesn't know the size of needs an explicit `dimensions=`."""
    with pytest.raises(UserError, match="dimensions"):
        Knowledge(model="some-custom-model")


def test_ingest_with_missing_directory_stores_nothing(tmp_path: Path) -> None:
    """A `directory` that doesn't exist ingests to zero chunks instead of raising."""
    know = _knowledge(tmp_path, tmp_path / "does-not-exist")

    assert asyncio.run(know.ingest()) == 0


def test_an_empty_corpus_logs_the_directory_it_looked_in(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The one retrieval failure nothing downstream can show: say it out loud, once."""
    missing = tmp_path / "does-not-exist"
    know = _knowledge(tmp_path, missing)

    with caplog.at_level("WARNING", logger="runa"):
        asyncio.run(know.ingest())

    assert str(missing) in caplog.text


def test_ingest_discovers_and_chunks_supported_files(tmp_path: Path) -> None:
    """`.md`/`.txt`/`.csv` files under `directory` are all discovered, read, and stored."""
    source_dir = tmp_path / "knowledge"
    source_dir.mkdir()
    (source_dir / "pets.md").write_text("cats are great pets")
    (source_dir / "companions.txt").write_text("dogs are loyal companions")
    (source_dir / "notes.csv").write_text("the stock market fell today")
    (source_dir / "ignored.json").write_text('{"not": "supported"}')
    know = _knowledge(tmp_path, source_dir)

    stored = asyncio.run(know.ingest())

    assert stored == 3


def test_ingest_extracts_pdf_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A `.pdf` file is read via `pypdf`, concatenating each page's extracted text."""

    class _FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _FakeReader:
        def __init__(self, path: str) -> None:
            self.pages = [_FakePage("cats are great pets"), _FakePage("dogs are loyal companions")]

    import pypdf

    monkeypatch.setattr(pypdf, "PdfReader", _FakeReader)

    source_dir = tmp_path / "knowledge"
    source_dir.mkdir()
    (source_dir / "doc.pdf").write_bytes(b"%PDF-fake")
    know = _knowledge(tmp_path, source_dir)

    stored = asyncio.run(know.ingest())

    assert stored == 1


def test_search_returns_the_closest_chunk_first(tmp_path: Path) -> None:
    """A query embeds closest to the semantically related chunk, which search ranks first."""
    source_dir = tmp_path / "knowledge"
    source_dir.mkdir()
    (source_dir / "pets.md").write_text("cats are great pets")
    (source_dir / "companions.txt").write_text("dogs are loyal companions")
    (source_dir / "market.txt").write_text("the stock market fell today")
    know = _knowledge(tmp_path, source_dir)

    matches = asyncio.run(know.search("query about pets", k=2))

    assert [m.text for m in matches] == ["cats are great pets", "dogs are loyal companions"]
    assert matches[0].distance <= matches[1].distance
    assert matches[0].source.endswith("pets.md")


def test_search_lazily_ingests_when_never_ingested(tmp_path: Path) -> None:
    """`search` ingests automatically the first time -- no manual `.ingest()` call needed."""
    source_dir = tmp_path / "knowledge"
    source_dir.mkdir()
    (source_dir / "pets.md").write_text("cats are great pets")
    know = _knowledge(tmp_path, source_dir)

    matches = asyncio.run(know.search("query about pets", k=1))

    assert [m.text for m in matches] == ["cats are great pets"]


def test_search_does_not_reingest_an_unchanged_corpus(tmp_path: Path) -> None:
    """An unchanged directory is ingested once, however many searches run over it.

    Counted in embedding calls rather than in `ingest()` calls, because the cost this is about is
    the embedding bill: one call for the corpus, then one per query.
    """
    source_dir = tmp_path / "knowledge"
    source_dir.mkdir()
    (source_dir / "pets.md").write_text("cats are great pets")
    know = _knowledge(tmp_path, source_dir)

    asyncio.run(know.search("query about pets", k=1))
    before = list(_embedded)
    asyncio.run(know.search("query about pets", k=1))

    assert _embedded == before + ["query about pets"]


def test_a_second_instance_searches_what_the_first_ingested(tmp_path: Path) -> None:
    """A fresh `Knowledge` over an already-ingested corpus searches it instead of rebuilding.

    The shape `runa.serve` has: `Agent(knowledge="auto")` builds its own `Knowledge` per agent and
    an agent is built per request, so a per-instance "have I ingested" flag would re-embed the
    whole directory on every call.
    """
    source_dir = tmp_path / "knowledge"
    source_dir.mkdir()
    (source_dir / "pets.md").write_text("cats are great pets")
    asyncio.run(_knowledge(tmp_path, source_dir).search("query about pets", k=1))
    _embedded.clear()

    matches = asyncio.run(_knowledge(tmp_path, source_dir).search("query about pets", k=1))

    assert [m.text for m in matches] == ["cats are great pets"]
    assert _embedded == ["query about pets"]


class _WatchedStore:
    """A `VersionedVectorStore` wrapper that records what an ingest did, and in what order.

    Wraps a real store rather than faking one, so the `version()` each step sees is the one a
    reader in another process would get from the same storage.
    """

    def __init__(self, inner: VersionedVectorStore) -> None:
        self._inner = inner
        #: One entry per call, each the operation's name and the version visible as it ran.
        self.steps: list[tuple[str, str | None]] = []

    async def _note(self, operation: str) -> None:
        self.steps.append((operation, await self._inner.version()))

    async def add(self, *, payload: dict[str, Any], embedding: list[float]) -> int:
        await self._note("add")
        return await self._inner.add(payload=payload, embedding=embedding)

    async def nearest(
        self, *, embedding: list[float], k: int, partition: str | None = None
    ) -> list[VectorMatch]:
        await self._note("nearest")
        return await self._inner.nearest(embedding=embedding, k=k, partition=partition)

    async def delete(self, *, item_id: int, partition: str | None = None) -> None:
        await self._inner.delete(item_id=item_id, partition=partition)

    async def reset(self) -> None:
        await self._inner.reset()
        await self._note("reset")

    async def stamp(self, version: str) -> None:
        await self._inner.stamp(version)
        await self._note("stamp")

    async def version(self) -> str | None:
        return await self._inner.version()


def _watched(tmp_path: Path, source_dir: Path) -> tuple[Knowledge, _WatchedStore]:
    store = _WatchedStore(_sqlite_store(tmp_path))
    know = Knowledge(source_dir, dimensions=_DIMENSIONS, store=store)
    return know, store


def test_no_version_is_published_until_every_chunk_has_landed(tmp_path: Path) -> None:
    """The store reports no corpus for the whole of an ingest, and this one only at the end.

    The window this closes is the one a `runa.serve` worker hits on every cold start: a version
    stamped before the embedding round trip and the inserts behind it tells every other reader of
    the same storage that the corpus is there, so each of them skips its own ingest and searches
    whatever has arrived so far -- and an empty knowledge block reads exactly like a corpus with
    nothing relevant in it, so nothing downstream can tell.
    """
    source_dir = tmp_path / "knowledge"
    source_dir.mkdir()
    (source_dir / "pets.md").write_text("cats are great pets")
    (source_dir / "dogs.md").write_text("dogs are loyal companions")
    know, store = _watched(tmp_path, source_dir)

    asyncio.run(know.ingest())

    assert [operation for operation, _ in store.steps] == ["reset", "add", "add", "stamp"]
    # Every step before the last saw an unversioned store: nothing claimed this corpus early.
    assert [version for _, version in store.steps[:-1]] == [None, None, None]
    assert store.steps[-1][1] == asyncio.run(know._fingerprint())


def test_an_empty_corpus_is_stamped_too(tmp_path: Path) -> None:
    """An empty directory still ends versioned, so it is fingerprinted once and not per turn."""
    know, store = _watched(tmp_path, tmp_path / "knowledge")

    assert asyncio.run(know.ingest()) == 0
    assert [operation for operation, _ in store.steps] == ["reset", "stamp"]
    assert asyncio.run(store.version()) == asyncio.run(know._fingerprint())


def test_an_ingest_that_dies_midway_leaves_nothing_claiming_the_corpus(tmp_path: Path) -> None:
    """A crash between `reset` and `stamp` costs the next search a rebuild, not a thin answer.

    The deliberate trade of stamping last: an interrupted ingest is wasted work, where a version
    left standing over rows that never arrived is a wrong answer with no mark on it.
    """
    source_dir = tmp_path / "knowledge"
    source_dir.mkdir()
    (source_dir / "pets.md").write_text("cats are great pets")
    store = _sqlite_store(tmp_path)
    know = Knowledge(source_dir, dimensions=_DIMENSIONS, store=store)

    async def _die_after_reset() -> None:
        async def boom(*, payload: dict[str, Any], embedding: list[float]) -> int:
            raise RuntimeError("process died mid-ingest")

        original = store.add
        store.add = boom  # type: ignore[method-assign]
        try:
            with pytest.raises(RuntimeError):
                await know.ingest()
        finally:
            store.add = original  # type: ignore[method-assign]

    asyncio.run(_die_after_reset())

    assert asyncio.run(store.version()) is None
    # A fresh instance, as the next process would be: it rebuilds rather than trusting the store.
    assert [
        m.text
        for m in asyncio.run(_knowledge(tmp_path, source_dir).search("query about pets", k=1))
    ] == ["cats are great pets"]


def test_concurrent_searches_on_a_cold_store_ingest_once(tmp_path: Path) -> None:
    """Two turns arriving together embed the corpus once between them, not once each.

    The `runa.serve` shape: `Agent(knowledge="auto")` builds a `Knowledge` per agent and an agent
    per request, so concurrent requests against a cold store are the common case, not an edge one.
    Counted in embedding calls, because that is the bill a duplicate rebuild runs up.
    """
    source_dir = tmp_path / "knowledge"
    source_dir.mkdir()
    (source_dir / "pets.md").write_text("cats are great pets")

    async def _both() -> list[list[KnowledgeMatch]]:
        first = _knowledge(tmp_path, source_dir)
        second = _knowledge(tmp_path, source_dir)
        return list(
            await asyncio.gather(
                first.search("query about pets", k=1), second.search("query about pets", k=1)
            )
        )

    results = asyncio.run(_both())

    assert [[m.text for m in found] for found in results] == [["cats are great pets"]] * 2
    assert _embedded.count("cats are great pets") == 1


def test_search_reingests_after_a_source_file_changes(tmp_path: Path) -> None:
    """Editing a file is picked up by the next `search`, with no manual `.ingest()` call."""
    source_dir = tmp_path / "knowledge"
    source_dir.mkdir()
    (source_dir / "pets.md").write_text("cats are great pets")
    know = _knowledge(tmp_path, source_dir)
    asyncio.run(know.search("query about pets", k=1))

    (source_dir / "pets.md").write_text("the stock market fell today")
    matches = asyncio.run(know.search("query about pets", k=5))

    assert [m.text for m in matches] == ["the stock market fell today"]


def test_search_reingests_after_the_embedding_model_changes(tmp_path: Path) -> None:
    """Chunks embedded by another model aren't comparable, so the model is part of the corpus."""
    source_dir = tmp_path / "knowledge"
    source_dir.mkdir()
    (source_dir / "pets.md").write_text("cats are great pets")
    store = _sqlite_store(tmp_path)
    asyncio.run(
        Knowledge(source_dir, dimensions=_DIMENSIONS, store=store).search("query about pets", k=1)
    )

    _embedded.clear()
    other = Knowledge(source_dir, model="some-other-model", dimensions=_DIMENSIONS, store=store)
    asyncio.run(other.search("query about pets", k=1))

    assert "cats are great pets" in _embedded


def test_fingerprint_rereads_the_corpus_only_when_a_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unchanged corpus is answered from its `(path, size, mtime)`, not read again.

    `search` fingerprints before every turn of every knowledge-backed run, so this is counted in
    file reads: the corpus is read once, and not again until a file actually changes.
    """
    source_dir = tmp_path / "knowledge"
    source_dir.mkdir()
    (source_dir / "pets.md").write_text("cats are great pets")
    know = _knowledge(tmp_path, source_dir)
    reads: list[str] = []
    read_bytes = Path.read_bytes

    def recording(self: Path) -> bytes:
        reads.append(self.name)
        return read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", recording)

    first = asyncio.run(know._fingerprint())
    second = asyncio.run(know._fingerprint())

    assert second == first
    assert reads == ["pets.md"]


def test_fingerprint_changes_after_a_same_sized_edit(tmp_path: Path) -> None:
    """A same-length edit still invalidates the cached digest: mtime is part of the check."""
    source_dir = tmp_path / "knowledge"
    source_dir.mkdir()
    (source_dir / "pets.md").write_text("cats are great pets")
    know = _knowledge(tmp_path, source_dir)
    before = asyncio.run(know._fingerprint())

    (source_dir / "pets.md").write_text("dogs are great pets")

    assert asyncio.run(know._fingerprint()) != before


def test_fingerprint_runs_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hashing the corpus is blocking I/O, so it runs in a thread, like a sync `@tool` does."""
    source_dir = tmp_path / "knowledge"
    source_dir.mkdir()
    (source_dir / "pets.md").write_text("cats are great pets")
    know = _knowledge(tmp_path, source_dir)
    hashed_on: list[int] = []
    discover = knowledge_module._discover

    def recording(directory: Path) -> list[Path]:
        hashed_on.append(threading.get_ident())
        return discover(directory)

    monkeypatch.setattr(knowledge_module, "_discover", recording)

    async def _run() -> int:
        await know._fingerprint()
        return threading.get_ident()

    loop_thread = asyncio.run(_run())

    assert hashed_on and loop_thread not in hashed_on


def test_ingest_twice_embeds_an_unchanged_corpus_once(tmp_path: Path) -> None:
    """`ingest()` asks the store what it holds, so calling it again is free, not a second bill.

    The property that lets `serve.create_app` ingest at startup and `search` ingest lazily be the
    same call: every replica's boot, and every agent sharing one corpus, would otherwise re-embed
    the whole directory under the name of warming it up.
    """
    source_dir = tmp_path / "knowledge"
    source_dir.mkdir()
    (source_dir / "pets.md").write_text("cats are great pets")
    know = _knowledge(tmp_path, source_dir)

    assert asyncio.run(know.ingest()) == 1
    before = list(_embedded)

    assert asyncio.run(_knowledge(tmp_path, source_dir).ingest()) == 0
    assert _embedded == before


def test_ingest_again_reflects_edited_and_removed_files(tmp_path: Path) -> None:
    """An edited corpus is rebuilt in full -- edits and removals aren't left as stale chunks."""
    source_dir = tmp_path / "knowledge"
    source_dir.mkdir()
    pets_file = source_dir / "pets.md"
    pets_file.write_text("cats are great pets")
    know = _knowledge(tmp_path, source_dir)
    asyncio.run(know.ingest())

    pets_file.write_text("the stock market fell today")
    asyncio.run(know.ingest())
    matches = asyncio.run(know.search("query about pets", k=5))

    assert [m.text for m in matches] == ["the stock market fell today"]


def test_as_tool_formats_matches_as_plain_text(tmp_path: Path) -> None:
    """`_as_tool` (used internally by `Agent(knowledge="llm")`) wraps `search` as joined text."""
    source_dir = tmp_path / "knowledge"
    source_dir.mkdir()
    (source_dir / "pets.md").write_text("cats are great pets")
    know = _knowledge(tmp_path, source_dir)
    search_tool = know._as_tool()

    args = '{"query": "query about pets", "k": 1}'
    result = run_awaitable(search_tool.on_invoke_tool(RunContext(), args, "call_1"))

    assert result == "cats are great pets"


def test_context_is_a_labeled_block_of_matching_chunks(tmp_path: Path) -> None:
    """`context` searches the corpus and labels what it found, or answers `None` for nothing."""
    source_dir = tmp_path / "knowledge"
    source_dir.mkdir()
    (source_dir / "pets.md").write_text("cats are great pets")
    know = _knowledge(tmp_path, source_dir)
    (tmp_path / "empty").mkdir()
    empty = _knowledge(tmp_path / "empty", tmp_path / "does-not-exist")

    block = asyncio.run(know.retrieve("query about pets", RunContext()))

    assert block == {"role": "system", "content": "Relevant knowledge:\n- cats are great pets"}
    assert asyncio.run(empty.retrieve("query about pets", RunContext())) is None


def test_after_turn_keeps_nothing(tmp_path: Path) -> None:
    """A run's exchange never becomes knowledge: the corpus is the directory."""
    know = _knowledge(tmp_path, tmp_path / "does-not-exist")
    model: Any = None  # never called

    assert asyncio.run(know.after_turn("hi", "hello", RunContext(), model)) == []


def test_as_tool_with_no_matches_says_so(tmp_path: Path) -> None:
    """An empty knowledge base's tool call reports no results instead of an empty string."""
    know = _knowledge(tmp_path, tmp_path / "does-not-exist")
    search_tool = know._as_tool()

    result = run_awaitable(
        search_tool.on_invoke_tool(RunContext(), '{"query": "anything"}', "call_1")
    )

    assert result == "No relevant knowledge found."


def test_custom_store_can_be_injected() -> None:
    """`Knowledge(store=...)` uses the given `VersionedVectorStore`, no inheritance required."""

    class _FakeStore:
        def __init__(self) -> None:
            self.added: list[dict[str, Any]] = []
            self.recorded: str | None = None

        async def add(self, *, payload: dict[str, Any], embedding: list[float]) -> int:
            self.added.append(payload)
            return len(self.added)

        async def nearest(
            self, *, embedding: list[float], k: int, partition: str | None = None
        ) -> list[VectorMatch]:
            return [
                VectorMatch(id=idx, payload=item, distance=0.0)
                for idx, item in enumerate(self.added, start=1)
            ]

        async def delete(self, *, item_id: int, partition: str | None = None) -> None:
            del self.added[item_id - 1]

        async def reset(self) -> None:
            self.added = []
            self.recorded = None

        async def stamp(self, version: str) -> None:
            self.recorded = version

        async def version(self) -> str | None:
            return self.recorded

    store = _FakeStore()
    know = Knowledge(store=store)

    async def _run() -> list[KnowledgeMatch]:
        # Stamping the store with this corpus's fingerprint is what tells `search` not to
        # rebuild, the same way an earlier process's `ingest()` would have.
        await store.add(
            payload={"text": "cats are great pets", "source": "manual"}, embedding=[0.0]
        )
        await store.stamp(await know._fingerprint())
        return await know.search("anything")

    matches = asyncio.run(_run())

    assert [m.text for m in matches] == ["cats are great pets"]
