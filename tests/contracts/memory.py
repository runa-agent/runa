"""contracts/memory.py: the one `MemoryStore` contract, so every backend is held to it.

What `Memory` needs of whatever it was given: vectors go in, the nearest come back, and one user's
facts are never another's. The embedding model is `Memory`'s business, not the store's, so these
checks hand in vectors directly -- `DIMENSIONS` floats of them, which is what a store under test
has to have been built for.

Each check takes the store and one `user_id` of its own, which is also what keeps them honest
against a live Postgres: scoping is the contract, so a check that leaked would see another's items.
"""

from collections.abc import Callable, Coroutine
from typing import Any

from runa.memory.store import MemoryStore

DIMENSIONS = 4

Check = Callable[[MemoryStore, str], Coroutine[Any, Any, None]]

_CATS = [1.0, 0.0, 0.0, 0.0]
_FINANCE = [0.0, 0.0, 0.0, 1.0]


async def check_add_then_search_returns_the_closest_match_first(
    store: MemoryStore, user_id: str
) -> None:
    """A search vector closest to one stored item ranks it first, with its distance."""
    await store.add(user_id=user_id, text="cats", embedding=_CATS, metadata=None)
    await store.add(user_id=user_id, text="finance", embedding=_FINANCE, metadata=None)

    matches = await store.search(user_id=user_id, embedding=_CATS, k=2)

    assert [match.text for match in matches] == ["cats", "finance"]
    assert matches[0].distance <= matches[1].distance


async def check_add_returns_the_new_items_id(store: MemoryStore, user_id: str) -> None:
    """`add` hands back an id, which is what `Memory.remember` returns to its caller."""
    first = await store.add(user_id=user_id, text="cats", embedding=_CATS, metadata=None)
    second = await store.add(user_id=user_id, text="finance", embedding=_FINANCE, metadata=None)

    assert first != second
    assert {match.id for match in await store.search(user_id=user_id, embedding=_CATS, k=2)} == {
        first,
        second,
    }


async def check_metadata_round_trips(store: MemoryStore, user_id: str) -> None:
    """Metadata passed to `add` comes back through `search` as the same dict."""
    await store.add(user_id=user_id, text="cats", embedding=_CATS, metadata={"source": "notes"})

    matches = await store.search(user_id=user_id, embedding=_CATS, k=1)

    assert matches[0].metadata == {"source": "notes"}


async def check_an_item_stored_without_metadata_has_none(store: MemoryStore, user_id: str) -> None:
    """`metadata=None` comes back as `None`, not as an empty dict."""
    await store.add(user_id=user_id, text="cats", embedding=_CATS, metadata=None)

    matches = await store.search(user_id=user_id, embedding=_CATS, k=1)

    assert matches[0].metadata is None


async def check_search_respects_k(store: MemoryStore, user_id: str) -> None:
    """`k` caps how many items come back."""
    await store.add(user_id=user_id, text="cats", embedding=_CATS, metadata=None)
    await store.add(user_id=user_id, text="finance", embedding=_FINANCE, metadata=None)

    assert len(await store.search(user_id=user_id, embedding=_CATS, k=1)) == 1


async def check_search_for_a_user_with_no_items_is_empty(store: MemoryStore, user_id: str) -> None:
    """A user who has never had a fact stored gets an empty list, not an error."""
    assert await store.search(user_id=f"{user_id}-nobody", embedding=_CATS, k=5) == []


async def check_search_is_scoped_to_one_user(store: MemoryStore, user_id: str) -> None:
    """A search scoped to one `user_id` never returns another user's item."""
    await store.add(user_id=user_id, text="mine", embedding=_CATS, metadata=None)

    assert await store.search(user_id=f"{user_id}-other", embedding=_CATS, k=5) == []


async def check_a_none_user_id_is_its_own_scope(store: MemoryStore, user_id: str) -> None:
    """Items with `user_id=None` don't leak into a real user's search, or vice versa."""
    item_id = await store.add(user_id=None, text=user_id, embedding=_CATS, metadata=None)
    try:
        scoped = await store.search(user_id=user_id, embedding=_CATS, k=5)
        unscoped = await store.search(user_id=None, embedding=_CATS, k=5)
    finally:
        # The one scope a unique `user_id` cannot isolate, so this check cleans up after itself
        # rather than leaving a row in a shared database for the next run to search over.
        await store.delete(user_id=None, memory_id=item_id)

    assert scoped == []
    assert user_id in [match.text for match in unscoped]


async def check_delete_removes_the_item(store: MemoryStore, user_id: str) -> None:
    """A deleted item no longer comes back from `search`."""
    item_id = await store.add(user_id=user_id, text="cats", embedding=_CATS, metadata=None)

    await store.delete(user_id=user_id, memory_id=item_id)

    assert await store.search(user_id=user_id, embedding=_CATS, k=5) == []


async def check_delete_under_another_user_id_does_not_remove_it(
    store: MemoryStore, user_id: str
) -> None:
    """Deleting is scoped too: one user cannot forget another's fact by guessing its id."""
    item_id = await store.add(user_id=user_id, text="cats", embedding=_CATS, metadata=None)

    await store.delete(user_id=f"{user_id}-other", memory_id=item_id)

    assert [match.text for match in await store.search(user_id=user_id, embedding=_CATS, k=5)] == [
        "cats"
    ]


async def check_delete_of_a_missing_item_does_not_raise(store: MemoryStore, user_id: str) -> None:
    """Deleting an item that was never stored is a no-op, not an error."""
    await store.delete(user_id=user_id, memory_id=2**40)


CONTRACT: list[Check] = [
    check_add_then_search_returns_the_closest_match_first,
    check_add_returns_the_new_items_id,
    check_metadata_round_trips,
    check_an_item_stored_without_metadata_has_none,
    check_search_respects_k,
    check_search_for_a_user_with_no_items_is_empty,
    check_search_is_scoped_to_one_user,
    check_a_none_user_id_is_its_own_scope,
    check_delete_removes_the_item,
    check_delete_under_another_user_id_does_not_remove_it,
    check_delete_of_a_missing_item_does_not_raise,
]
