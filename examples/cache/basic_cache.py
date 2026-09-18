"""`Cache`: a minimal get/set/delete/clear cache, independent of Agent, Memory, and Sessions.

See docs/cache.md.

Cache isn't one of RUNA.md's numbered primitives, and nothing wires it into the run lifecycle
automatically -- use it directly wherever your own code wants to cache something: an expensive
tool call, an external API response, a computed value.

Run it:

    uv run python examples/cache/basic_cache.py
"""

import asyncio

from runa import MemoryCache, SQLiteCache


async def main() -> None:
    """Exercise the same four async methods against both a `MemoryCache` and a `SQLiteCache`."""
    memory_cache = MemoryCache()  # a plain dict, gone when the process exits
    await memory_cache.set("weather:paris", {"temp_c": 18}, ttl=300)
    print(await memory_cache.get("weather:paris"))
    await memory_cache.delete("weather:paris")
    print(await memory_cache.get("weather:paris"))  # None: deleted

    sqlite_cache = SQLiteCache()  # persisted to db/runa.db, the same file Sessions/Memory use
    await sqlite_cache.set("weather:tokyo", {"temp_c": 24}, ttl=300)
    print(await sqlite_cache.get("weather:tokyo"))
    await sqlite_cache.clear()  # every key, not just this one


asyncio.run(main())
