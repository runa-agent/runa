"""Tests for `runa.knowledge.Knowledge`, over the `KnowledgeStore` contract every backend answers.

The store contract lives in `tests/contracts/knowledge.py` and is driven here over the backends
`runa.db` can resolve without a live Postgres; `tests/test_postgres.py` drives the same checks
against `PostgresKnowledgeStore`. Everything below that is `Knowledge`'s own: discovery, chunking,
PDF extraction, lazy ingest, and the search tool it hands a model.
"""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from contracts.knowledge import CONTRACT, DIMENSIONS, Check
from helpers import run as run_awaitable

from runa import db
from runa import knowledge as knowledge_module
from runa._types import RunContextWrapper
from runa.knowledge import (
    DEFAULT_KNOWLEDGE_DIR,
    Knowledge,
    KnowledgeMatch,
    KnowledgeStore,
    SQLiteKnowledgeStore,
)

_DIMENSIONS = DIMENSIONS


@pytest.fixture(params=["sqlite", "ephemeral"])
def store(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> KnowledgeStore:
    """A `KnowledgeStore`, resolved by `runa.db` the way a bare `Knowledge()` gets one."""
    if request.param == "ephemeral":
        monkeypatch.setenv(db.DATABASE_URL_ENV, "memory://")
    else:
        monkeypatch.setenv(db.DATABASE_URL_ENV, f"sqlite:///{tmp_path / 'runa.db'}")
    return db.knowledge_store(dimensions=DIMENSIONS)


@pytest.mark.parametrize("check", CONTRACT, ids=lambda check: check.__name__)
def test_knowledge_store_contract(store: KnowledgeStore, check: Check) -> None:
    """Every local backend answers the `KnowledgeStore` contract the same way."""
    asyncio.run(check(store, "doc"))


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


def _knowledge(tmp_path: Path, directory: Path | None = None) -> Knowledge:
    return Knowledge(
        directory if directory is not None else tmp_path / "knowledge",
        dimensions=_DIMENSIONS,
        store=SQLiteKnowledgeStore(tmp_path / "runa.db", dimensions=_DIMENSIONS),
    )


def test_default_configuration_needs_no_arguments() -> None:
    """`Knowledge()` means `app/knowledge/` + `db/runa.db` + `sqlite-vec` + the default model."""
    know = Knowledge()

    assert know.directory == DEFAULT_KNOWLEDGE_DIR
    assert know.model == "text-embedding-3-small"
    assert know.dimensions == 1536
    assert isinstance(know._store, SQLiteKnowledgeStore)
    assert str(know._store.db_path) == str(Path("db/runa.db"))


def test_explicit_directory_overrides_the_default() -> None:
    """`Knowledge(path)` uses `path` as the source directory instead of `app/knowledge/`."""
    know = Knowledge("some/other/path")

    assert know.directory == Path("some/other/path")


def test_unknown_model_without_dimensions_raises() -> None:
    """A model Runa doesn't know the size of needs an explicit `dimensions=`."""
    with pytest.raises(ValueError, match="dimensions"):
        Knowledge(model="some-custom-model")


def test_ingest_with_missing_directory_stores_nothing(tmp_path: Path) -> None:
    """A `directory` that doesn't exist ingests to zero chunks instead of raising."""
    know = _knowledge(tmp_path, tmp_path / "does-not-exist")

    assert asyncio.run(know.ingest()) == 0


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
    store = SQLiteKnowledgeStore(tmp_path / "runa.db", dimensions=_DIMENSIONS)
    asyncio.run(
        Knowledge(source_dir, dimensions=_DIMENSIONS, store=store).search("query about pets", k=1)
    )

    _embedded.clear()
    other = Knowledge(source_dir, model="some-other-model", dimensions=_DIMENSIONS, store=store)
    asyncio.run(other.search("query about pets", k=1))

    assert "cats are great pets" in _embedded


def test_ingest_again_reflects_edited_and_removed_files(tmp_path: Path) -> None:
    """Calling `ingest()` again fully rebuilds -- edits and removals aren't left as stale chunks."""
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
    result = run_awaitable(
        search_tool.on_invoke_tool(RunContextWrapper(context=None), args, "call_1")
    )

    assert result == "cats are great pets"


def test_as_tool_with_no_matches_says_so(tmp_path: Path) -> None:
    """An empty knowledge base's tool call reports no results instead of an empty string."""
    know = _knowledge(tmp_path, tmp_path / "does-not-exist")
    search_tool = know._as_tool()

    result = run_awaitable(
        search_tool.on_invoke_tool(
            RunContextWrapper(context=None), '{"query": "anything"}', "call_1"
        )
    )

    assert result == "No relevant knowledge found."


def test_custom_store_can_be_injected() -> None:
    """`Knowledge(store=...)` uses the given `KnowledgeStore` instead of the default SQLite one."""

    class _FakeStore:
        def __init__(self) -> None:
            self.added: list[dict[str, Any]] = []
            self.recorded: str | None = None

        async def add(self, *, text: str, source: str, embedding: list[float]) -> int:
            self.added.append({"text": text, "source": source})
            return len(self.added)

        async def search(self, *, embedding: list[float], k: int) -> list[KnowledgeMatch]:
            return [
                KnowledgeMatch(id=idx, text=item["text"], source=item["source"], distance=0.0)
                for idx, item in enumerate(self.added, start=1)
            ]

        async def reset(self, *, version: str | None = None) -> None:
            self.added = []
            self.recorded = version

        async def version(self) -> str | None:
            return self.recorded

    store = _FakeStore()
    know = Knowledge(store=store)

    async def _run() -> list[KnowledgeMatch]:
        # Stamping the store with this corpus's fingerprint is what tells `search` not to
        # rebuild, the same way an earlier process's `ingest()` would have.
        await store.reset(version=know._fingerprint())
        await store.add(text="cats are great pets", source="manual", embedding=[0.0])
        return await know.search("anything")

    matches = asyncio.run(_run())

    assert [m.text for m in matches] == ["cats are great pets"]
