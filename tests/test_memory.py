"""Tests for `runa.memory.Memory`."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from helpers import run as run_awaitable

from runa import memory as memory_module
from runa._types import RunContextWrapper
from runa.memory import Memory, MemoryMatch, MemoryStore, SQLiteMemoryStore

_DIMENSIONS = 4


@pytest.fixture(autouse=True)
def _fake_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real OpenAI call with a lookup table, so tests never hit the network.

    Each text maps to a fixed 4-dimensional vector; unmapped text embeds to the origin, which is
    never asserted on directly, just used to keep the call total.
    """
    vectors = {
        "cats are great pets": [1.0, 0.0, 0.0, 0.0],
        "dogs are loyal companions": [0.9, 0.1, 0.0, 0.0],
        "the stock market fell today": [0.0, 0.0, 0.0, 1.0],
        "query about pets": [1.0, 0.0, 0.0, 0.0],
        "language preference": [1.0, 0.0, 0.0, 0.0],
        "User prefers Japanese.": [1.0, 0.0, 0.0, 0.0],
    }

    async def fake_embed(texts: list[str], *, model: str = "") -> list[list[float]]:
        return [vectors.get(text, [0.0, 0.0, 0.0, 0.0]) for text in texts]

    monkeypatch.setattr(memory_module, "embed", fake_embed)


def _memory(tmp_path: Path) -> Memory:
    return Memory(tmp_path / "runa.db", dimensions=_DIMENSIONS)


def test_default_configuration_needs_no_arguments() -> None:
    """`Memory()` means `db/runa.db` + `sqlite-vec` + `text-embedding-3-small`."""
    mem = Memory()

    assert mem.model == "text-embedding-3-small"
    assert mem.dimensions == 1536
    assert isinstance(mem._store, SQLiteMemoryStore)
    assert str(mem._store.db_path) == str(Path("db/runa.db"))


def test_unknown_model_without_dimensions_raises() -> None:
    """A model Runa doesn't know the size of needs an explicit `dimensions=`."""
    with pytest.raises(ValueError, match="dimensions"):
        Memory(model="some-custom-model")


def test_remember_then_search_returns_the_closest_match_first(tmp_path: Path) -> None:
    """A query embeds closest to the semantically related item, which search ranks first."""
    mem = _memory(tmp_path)

    async def _run():
        await mem.remember("cats are great pets")
        await mem.remember("dogs are loyal companions")
        await mem.remember("the stock market fell today")
        return await mem.search("query about pets", k=2)

    matches = asyncio.run(_run())

    assert [m.text for m in matches] == ["cats are great pets", "dogs are loyal companions"]
    assert matches[0].distance <= matches[1].distance


def test_remember_skips_a_near_duplicate_and_returns_the_existing_id(tmp_path: Path) -> None:
    """Remembering the same fact again doesn't add a second row -- it returns the first id."""
    mem = _memory(tmp_path)

    async def _run():
        first_id = await mem.remember("User prefers Japanese.", user_id="u1")
        second_id = await mem.remember("User prefers Japanese.", user_id="u1")
        matches = await mem.search("language preference", user_id="u1", k=5)
        return first_id, second_id, matches

    first_id, second_id, matches = asyncio.run(_run())

    assert first_id == second_id
    assert len(matches) == 1


def test_remember_stores_metadata_and_search_returns_it(tmp_path: Path) -> None:
    """Metadata passed to `remember` round-trips through `search` as a dict."""
    mem = _memory(tmp_path)

    async def _run():
        await mem.remember("cats are great pets", metadata={"source": "notes"})
        return await mem.search("query about pets", k=1)

    matches = asyncio.run(_run())

    assert matches[0].metadata == {"source": "notes"}


def test_search_with_no_metadata_returns_none(tmp_path: Path) -> None:
    """An item remembered without metadata comes back with `metadata=None`, not `{}`."""
    mem = _memory(tmp_path)

    async def _run():
        await mem.remember("cats are great pets")
        return await mem.search("query about pets", k=1)

    matches = asyncio.run(_run())

    assert matches[0].metadata is None


def test_user_isolation_search_never_returns_another_users_memory(tmp_path: Path) -> None:
    """A memory remembered for one `user_id` never surfaces in another user's search."""
    mem = _memory(tmp_path)

    async def _run():
        await mem.remember("User prefers Japanese.", user_id="u1")
        return await mem.search("language preference", user_id="u2")

    matches = asyncio.run(_run())

    assert matches == []


def test_user_isolation_search_returns_the_matching_users_memory(tmp_path: Path) -> None:
    """The same query, scoped to the right `user_id`, does find the memory."""
    mem = _memory(tmp_path)

    async def _run():
        await mem.remember("User prefers Japanese.", user_id="u1")
        return await mem.search("language preference", user_id="u1")

    matches = asyncio.run(_run())

    assert [m.text for m in matches] == ["User prefers Japanese."]


def test_no_user_id_is_its_own_scope_not_shared_across_everyone(tmp_path: Path) -> None:
    """Memories with no `user_id` don't leak into a search scoped to a real user, or vice versa."""
    mem = _memory(tmp_path)

    async def _run():
        await mem.remember("cats are great pets")  # no user_id
        return await mem.search("query about pets", user_id="u1")

    matches = asyncio.run(_run())

    assert matches == []


def test_forget_removes_the_memory(tmp_path: Path) -> None:
    """A forgotten memory no longer comes back from `search`."""
    mem = _memory(tmp_path)

    async def _run():
        memory_id = await mem.remember("cats are great pets", user_id="u1")
        await mem.forget(memory_id, user_id="u1")
        return await mem.search("query about pets", user_id="u1")

    matches = asyncio.run(_run())

    assert matches == []


def test_forget_with_the_wrong_user_id_does_not_remove_it(tmp_path: Path) -> None:
    """`forget` is scoped by `user_id`, same as `remember`/`search`."""
    mem = _memory(tmp_path)

    async def _run():
        memory_id = await mem.remember("cats are great pets", user_id="u1")
        await mem.forget(memory_id, user_id="u2")
        return await mem.search("query about pets", user_id="u1")

    matches = asyncio.run(_run())

    assert [m.text for m in matches] == ["cats are great pets"]


def test_as_tool_formats_matches_as_plain_text(tmp_path: Path) -> None:
    """`_as_tool` (used internally by `Agent(memory="llm")`) wraps `search` as joined text."""
    mem = _memory(tmp_path)
    search_tool = mem._as_tool()

    async def _run():
        await mem.remember("cats are great pets")
        await mem.remember("dogs are loyal companions")
        args = '{"query": "query about pets", "k": 1}'
        return await search_tool.on_invoke_tool(RunContextWrapper(context=None), args, "call_1")

    result = asyncio.run(_run())

    assert result == "cats are great pets"


def test_as_tool_with_no_matches_says_so(tmp_path: Path) -> None:
    """An empty memory's tool call reports no results instead of an empty string."""
    mem = _memory(tmp_path)
    search_tool = mem._as_tool()

    result = run_awaitable(
        search_tool.on_invoke_tool(
            RunContextWrapper(context=None), '{"query": "anything"}', "call_1"
        )
    )

    assert result == "No relevant memory found."


class _FakeModel:
    """A `Model` stand-in returning one scripted JSON reply, for extraction tests."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[Any] = []

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        from types import SimpleNamespace

        self.calls.append(args)
        return SimpleNamespace(output=[{"role": "assistant", "content": self.content}])


def test_remember_from_conversation_stores_extracted_candidates(tmp_path: Path) -> None:
    """The internal extraction hook stores whatever the model's JSON reply lists."""
    mem = _memory(tmp_path)
    model = _FakeModel('{"memories": ["User prefers Japanese."]}')

    async def _run():
        stored = await mem.remember_from_conversation(
            "User: hi\nAssistant: hello", user_id="u1", model=model
        )
        matches = await mem.search("language preference", user_id="u1")
        return stored, matches

    stored, matches = asyncio.run(_run())

    assert stored == ["User prefers Japanese."]
    assert [m.text for m in matches] == ["User prefers Japanese."]


def test_remember_from_conversation_stores_nothing_when_the_model_finds_nothing(
    tmp_path: Path,
) -> None:
    """An empty `memories` list stores nothing, rather than an empty-string memory."""
    mem = _memory(tmp_path)
    model = _FakeModel('{"memories": []}')

    async def _run():
        return await mem.remember_from_conversation(
            "User: hi\nAssistant: hello", user_id="u1", model=model
        )

    assert asyncio.run(_run()) == []


def test_custom_store_can_be_injected() -> None:
    """`Memory(store=...)` uses the given `MemoryStore` instead of the default SQLite one."""

    class _FakeStore:
        def __init__(self) -> None:
            self.added: list[dict[str, Any]] = []

        async def add(
            self,
            *,
            user_id: str | None,
            text: str,
            embedding: list[float],
            metadata: dict[str, Any] | None,
        ) -> int:
            self.added.append({"user_id": user_id, "text": text, "metadata": metadata})
            return len(self.added)

        async def search(
            self, *, user_id: str | None, embedding: list[float], k: int
        ) -> list[MemoryMatch]:
            return [
                MemoryMatch(id=idx, text=item["text"], metadata=item["metadata"], distance=0.0)
                for idx, item in enumerate(self.added, start=1)
            ]

        async def delete(self, *, user_id: str | None, memory_id: int) -> None:
            del self.added[memory_id - 1]

    store: MemoryStore = _FakeStore()
    mem = Memory(store=store)

    async def _run():
        await mem.remember("cats are great pets", user_id="u1")
        return await mem.search("anything", user_id="u1")

    matches = asyncio.run(_run())

    assert isinstance(store, _FakeStore)
    assert store.added == [{"user_id": "u1", "text": "cats are great pets", "metadata": None}]
    assert [m.text for m in matches] == ["cats are great pets"]
