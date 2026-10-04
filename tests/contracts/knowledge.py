"""contracts/knowledge.py: the one `KnowledgeStore` contract, so every backend is held to it.

`memory.py`'s counterpart for the application's own documents. Chunks are application-scoped, not
per-user, so there is no scope to isolate a check in: each one starts by clearing the store, which
is also what an `ingest()` does. The tag it gets names the sources it writes, so a failure says
which check wrote the chunk it found.

Vectors go in directly, `DIMENSIONS` floats of them: chunking and embedding are `Knowledge`'s
business, not the store's.
"""

from collections.abc import Callable, Coroutine
from typing import Any

from runa.knowledge.store import KnowledgeStore

DIMENSIONS = 4

Check = Callable[[KnowledgeStore, str], Coroutine[Any, Any, None]]

_NEAR = [1.0, 0.0, 0.0, 0.0]
_FAR = [0.0, 0.0, 0.0, 1.0]


async def check_add_then_search_returns_the_closest_chunk_first(
    store: KnowledgeStore, tag: str
) -> None:
    """A search vector closest to one stored chunk ranks it first."""
    await store.clear()
    await store.add(text="chunk a", source=f"{tag}-a.md", embedding=_NEAR)
    await store.add(text="chunk b", source=f"{tag}-b.md", embedding=_FAR)

    matches = await store.search(embedding=_NEAR, k=2)

    assert [match.text for match in matches] == ["chunk a", "chunk b"]
    assert matches[0].distance <= matches[1].distance


async def check_search_says_which_file_each_chunk_came_from(
    store: KnowledgeStore, tag: str
) -> None:
    """A match carries its `source`, which is what a citation in an answer is built from."""
    await store.clear()
    await store.add(text="chunk a", source=f"{tag}-a.md", embedding=_NEAR)

    assert (await store.search(embedding=_NEAR, k=1))[0].source == f"{tag}-a.md"


async def check_add_returns_the_new_chunks_id(store: KnowledgeStore, tag: str) -> None:
    """`add` hands back an id, and two chunks never share one."""
    await store.clear()

    first = await store.add(text="chunk a", source=f"{tag}-a.md", embedding=_NEAR)
    second = await store.add(text="chunk b", source=f"{tag}-b.md", embedding=_FAR)

    assert first != second
    assert {match.id for match in await store.search(embedding=_NEAR, k=2)} == {first, second}


async def check_search_respects_k(store: KnowledgeStore, tag: str) -> None:
    """`k` caps how many chunks come back."""
    await store.clear()
    await store.add(text="chunk a", source=f"{tag}-a.md", embedding=_NEAR)
    await store.add(text="chunk b", source=f"{tag}-b.md", embedding=_FAR)

    assert len(await store.search(embedding=_NEAR, k=1)) == 1


async def check_clear_removes_every_chunk(store: KnowledgeStore, tag: str) -> None:
    """`clear` empties the whole store, not just the chunks from one source.

    An ingest re-reads the directory, so a file deleted since the last one has to stop being
    retrievable, and that is the only way a chunk is ever removed.
    """
    await store.add(text="chunk a", source=f"{tag}-a.md", embedding=_NEAR)
    await store.add(text="chunk b", source=f"{tag}-b.md", embedding=_FAR)

    await store.clear()

    assert await store.search(embedding=_NEAR, k=5) == []


async def check_search_on_an_empty_store_is_empty(store: KnowledgeStore, tag: str) -> None:
    """Searching a store with nothing ingested returns an empty list, not an error."""
    await store.clear()

    assert await store.search(embedding=_NEAR, k=5) == []


CONTRACT: list[Check] = [
    check_add_then_search_returns_the_closest_chunk_first,
    check_search_says_which_file_each_chunk_came_from,
    check_add_returns_the_new_chunks_id,
    check_search_respects_k,
    check_clear_removes_every_chunk,
    check_search_on_an_empty_store_is_empty,
]
