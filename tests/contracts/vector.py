"""contracts/vector.py: the one `VectorStore` contract, so every backend is held to it.

What `Memory` and `Knowledge` need of whatever storage `runa.db` resolved: embeddings go in, the
nearest come back with their payloads, one partition's rows are never another's, and a payload
survives the round trip unchanged. Embedding is each concern's business, not the store's, so these
checks hand in vectors directly, `DIMENSIONS` floats of them.

Each check is handed a `build` rather than a store, because two of the contract's promises are
about the spec itself: a partitioned store scopes search, and an unpartitioned one does not. The
`tag` it gets is the partition (or the source value) it writes under, which is what keeps checks
honest against a live Postgres, where the table outlives the test.
"""

from collections.abc import Callable, Coroutine
from typing import Any

from runa.db.vectors import Column, VectorSpec, VectorStore

DIMENSIONS = 4

Build = Callable[[VectorSpec], VectorStore]
Check = Callable[[Build, str], Coroutine[Any, Any, None]]

_NEAR = [1.0, 0.0, 0.0, 0.0]
_FAR = [0.0, 0.0, 0.0, 1.0]


def partitioned(name: str = "contract_scoped") -> VectorSpec:
    """A spec scoped by `owner`, the shape `Memory` uses."""
    return VectorSpec(
        name=name,
        dimensions=DIMENSIONS,
        columns={
            "owner": Column(),
            "text": Column(required=True),
            "extra": Column(json=True),
        },
        partition_by="owner",
    )


def flat(name: str = "contract_flat") -> VectorSpec:
    """A spec with no partition, the shape `Knowledge` uses."""
    return VectorSpec(
        name=name,
        dimensions=DIMENSIONS,
        columns={"text": Column(required=True), "source": Column(required=True)},
    )


async def check_nearest_returns_the_closest_payload_first(build: Build, tag: str) -> None:
    """A search vector closest to one stored row ranks it first, with its distance."""
    store = build(partitioned())
    await store.add(payload={"owner": tag, "text": "near", "extra": None}, embedding=_NEAR)
    await store.add(payload={"owner": tag, "text": "far", "extra": None}, embedding=_FAR)

    matches = await store.nearest(embedding=_NEAR, k=2, partition=tag)

    assert [match.payload["text"] for match in matches] == ["near", "far"]
    assert matches[0].distance <= matches[1].distance


async def check_add_returns_a_new_id_each_time(build: Build, tag: str) -> None:
    """`add` hands back an id, and two rows never share one."""
    store = build(partitioned())

    first = await store.add(payload={"owner": tag, "text": "a", "extra": None}, embedding=_NEAR)
    second = await store.add(payload={"owner": tag, "text": "b", "extra": None}, embedding=_FAR)

    assert first != second
    matches = await store.nearest(embedding=_NEAR, k=2, partition=tag)
    assert {match.id for match in matches} == {first, second}


async def check_an_id_is_not_reused_after_a_delete(build: Build, tag: str) -> None:
    """An id is never handed out twice, even once the row that had it is gone.

    A store numbering rows by how many it holds hands the next `add` an id a surviving row is
    still using, so a later `delete` of the new row takes the old one with it.
    """
    store = build(partitioned())
    first = await store.add(payload={"owner": tag, "text": "a", "extra": None}, embedding=_NEAR)
    second = await store.add(payload={"owner": tag, "text": "b", "extra": None}, embedding=_FAR)

    await store.delete(item_id=first, partition=tag)
    third = await store.add(payload={"owner": tag, "text": "c", "extra": None}, embedding=_NEAR)

    assert third not in {first, second}


async def check_a_json_payload_round_trips(build: Build, tag: str) -> None:
    """A JSON column's value comes back as the same structure it went in as."""
    store = build(partitioned())
    extra = {"source": "notes", "tags": ["a", "b"], "depth": {"nested": 1}}
    await store.add(payload={"owner": tag, "text": "a", "extra": extra}, embedding=_NEAR)

    matches = await store.nearest(embedding=_NEAR, k=1, partition=tag)

    assert matches[0].payload["extra"] == extra


async def check_a_json_payload_round_trips_through_json_in_every_backend(
    build: Build, tag: str
) -> None:
    """A value JSON cannot represent exactly comes back the same way in every backend.

    A tuple is the cheapest example: every backend stores a JSON column encoded, so it comes back
    a list. The in-process adapter kept payloads as live objects once, and handed this one back a
    tuple, which made `memory://` a stand-in a test could pass against and a deployment could not.
    """
    store = build(partitioned())
    await store.add(payload={"owner": tag, "text": "a", "extra": {"pair": (1, 2)}}, embedding=_NEAR)

    matches = await store.nearest(embedding=_NEAR, k=1, partition=tag)

    assert matches[0].payload["extra"] == {"pair": [1, 2]}


async def check_a_json_column_left_unset_comes_back_as_none(build: Build, tag: str) -> None:
    """`None` in a JSON column comes back as `None`, not as `"null"` or an empty dict."""
    store = build(partitioned())
    await store.add(payload={"owner": tag, "text": "a", "extra": None}, embedding=_NEAR)

    matches = await store.nearest(embedding=_NEAR, k=1, partition=tag)

    assert matches[0].payload["extra"] is None


async def check_nearest_respects_k(build: Build, tag: str) -> None:
    """`k` caps how many rows come back."""
    store = build(partitioned())
    await store.add(payload={"owner": tag, "text": "a", "extra": None}, embedding=_NEAR)
    await store.add(payload={"owner": tag, "text": "b", "extra": None}, embedding=_FAR)

    assert len(await store.nearest(embedding=_NEAR, k=1, partition=tag)) == 1


async def check_nearest_in_an_empty_partition_is_empty(build: Build, tag: str) -> None:
    """A partition nothing was ever stored under gets an empty list, not an error."""
    store = build(partitioned())

    assert await store.nearest(embedding=_NEAR, k=5, partition=f"{tag}-nobody") == []


async def check_nearest_is_scoped_to_one_partition(build: Build, tag: str) -> None:
    """A search scoped to one partition never returns another's row."""
    store = build(partitioned())
    await store.add(payload={"owner": tag, "text": "mine", "extra": None}, embedding=_NEAR)

    assert await store.nearest(embedding=_NEAR, k=5, partition=f"{tag}-other") == []


async def check_a_none_partition_is_its_own_scope(build: Build, tag: str) -> None:
    """Rows stored with no partition don't leak into a real one's search, or vice versa."""
    store = build(partitioned())
    item_id = await store.add(payload={"owner": None, "text": tag, "extra": None}, embedding=_NEAR)
    try:
        scoped = await store.nearest(embedding=_NEAR, k=5, partition=tag)
        unscoped = await store.nearest(embedding=_NEAR, k=5, partition=None)
    finally:
        # The one scope a unique `tag` cannot isolate, so this check cleans up after itself rather
        # than leaving a row in a shared database for the next run to search over.
        await store.delete(item_id=item_id, partition=None)

    assert scoped == []
    assert tag in [match.payload["text"] for match in unscoped]


async def check_delete_removes_the_row(build: Build, tag: str) -> None:
    """A deleted row no longer comes back from `nearest`."""
    store = build(partitioned())
    item_id = await store.add(payload={"owner": tag, "text": "a", "extra": None}, embedding=_NEAR)

    await store.delete(item_id=item_id, partition=tag)

    assert await store.nearest(embedding=_NEAR, k=5, partition=tag) == []


async def check_delete_in_another_partition_does_not_remove_it(build: Build, tag: str) -> None:
    """Deleting is scoped too: one partition cannot remove another's row by guessing its id."""
    store = build(partitioned())
    item_id = await store.add(payload={"owner": tag, "text": "a", "extra": None}, embedding=_NEAR)

    await store.delete(item_id=item_id, partition=f"{tag}-other")

    matches = await store.nearest(embedding=_NEAR, k=5, partition=tag)
    assert [match.payload["text"] for match in matches] == ["a"]


async def check_delete_of_a_missing_row_does_not_raise(build: Build, tag: str) -> None:
    """Deleting a row that was never stored is a no-op, not an error."""
    store = build(partitioned())

    await store.delete(item_id=2**40, partition=tag)


async def check_an_unpartitioned_store_searches_everything(build: Build, tag: str) -> None:
    """With no partition declared, every row is in scope and `partition` goes unread."""
    store = build(flat())
    await store.reset()
    await store.add(payload={"text": "near", "source": f"{tag}-a.md"}, embedding=_NEAR)
    await store.add(payload={"text": "far", "source": f"{tag}-b.md"}, embedding=_FAR)

    matches = await store.nearest(embedding=_NEAR, k=5)

    assert [match.payload["text"] for match in matches] == ["near", "far"]
    assert matches[0].payload["source"] == f"{tag}-a.md"


async def check_reset_removes_every_row(build: Build, tag: str) -> None:
    """`reset` empties the whole store, which is what a fresh `Knowledge.ingest()` needs."""
    store = build(flat())
    await store.add(payload={"text": "a", "source": f"{tag}-a.md"}, embedding=_NEAR)
    await store.add(payload={"text": "b", "source": f"{tag}-b.md"}, embedding=_FAR)

    await store.reset()

    assert await store.nearest(embedding=_NEAR, k=5) == []


async def check_nearest_on_a_reset_store_is_empty(build: Build, tag: str) -> None:
    """Searching a store with nothing in it returns an empty list, not an error."""
    store = build(flat())
    await store.reset()

    assert await store.nearest(embedding=_NEAR, k=5) == []


async def check_version_round_trips_through_reset(build: Build, tag: str) -> None:
    """`reset(version=...)` is what a later `version()` reports, so an ingest is detectable."""
    store = build(flat())

    await store.reset(version=f"{tag}-v1")

    assert await store.version() == f"{tag}-v1"


async def check_version_is_none_until_one_is_recorded(build: Build, tag: str) -> None:
    """A store nothing has stamped reports `None`, which no fingerprint ever equals."""
    store = build(flat())

    await store.reset()

    assert await store.version() is None


async def check_reset_replaces_the_previous_version(build: Build, tag: str) -> None:
    """A second `reset` leaves one version, not two: the store holds one corpus at a time."""
    store = build(flat())

    await store.reset(version=f"{tag}-v1")
    await store.reset(version=f"{tag}-v2")

    assert await store.version() == f"{tag}-v2"


async def check_a_version_survives_a_new_store_over_the_same_rows(build: Build, tag: str) -> None:
    """A second store object on the same storage sees the first one's version.

    The whole point of keeping it here rather than on the caller: a per-request `Agent` builds a
    fresh `Knowledge`, and it has to be able to tell that the corpus is already ingested.
    """
    spec = flat()
    await build(spec).reset(version=f"{tag}-v1")

    assert await build(spec).version() == f"{tag}-v1"


CONTRACT: list[Check] = [
    check_nearest_returns_the_closest_payload_first,
    check_add_returns_a_new_id_each_time,
    check_an_id_is_not_reused_after_a_delete,
    check_a_json_payload_round_trips,
    check_a_json_payload_round_trips_through_json_in_every_backend,
    check_a_json_column_left_unset_comes_back_as_none,
    check_nearest_respects_k,
    check_nearest_in_an_empty_partition_is_empty,
    check_nearest_is_scoped_to_one_partition,
    check_a_none_partition_is_its_own_scope,
    check_delete_removes_the_row,
    check_delete_in_another_partition_does_not_remove_it,
    check_delete_of_a_missing_row_does_not_raise,
    check_an_unpartitioned_store_searches_everything,
    check_reset_removes_every_row,
    check_nearest_on_a_reset_store_is_empty,
    check_version_round_trips_through_reset,
    check_version_is_none_until_one_is_recorded,
    check_reset_replaces_the_previous_version,
    check_a_version_survives_a_new_store_over_the_same_rows,
]
