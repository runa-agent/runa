# Memory

Long-term, semantic memory: facts that persist across conversations, not just within one (that
is a [Session](sessions.md)). Opt an agent in with `memory`:

```python
class SupportAgent(Agent):
    name = "support_agent"
    memory = "auto"
```

## `"auto"`: the framework remembers for you

Before each run, Runa searches memory for anything relevant to the user's message and adds
it to context. After the run, it asks the model what's durably worth keeping and stores it. No
manual `remember`/`search` calls.

```python
support = SupportAgent()  # memory = "auto"
support.run_sync("I only speak Japanese.", user_id="user-42")

# ...later, a new run:
support.run_sync("What's my order status?", user_id="user-42")  # sees "user speaks Japanese"
```

## Scoping memory to a user

`user_id=` on `run`/`run_sync`/`run_streamed` is who the run acts for, and the scope of every
memory read and write it makes, `"auto"` and `"llm"` alike. It's independent of
[Sessions](sessions.md): a session-less run is scoped the same way. A run with no `user_id`
uses a scope of its own, shared by every other run that passes none, so an agent serving more
than one user should always pass one. A tool or delegate reads it as `ctx.user_id`; a paused
`RunState` carries it, so resuming takes no `user_id=`.

## `"llm"`: the model decides when to look

The model gets a `search_memory` tool and calls it itself. Nothing is written automatically.
Call `memory.remember(...)` from your own tools if you want the model's actions to persist.

```python
class SupportAgent(Agent):
    name = "support_agent"
    memory = "llm"
```

## `Memory` Directly

Reach for a `Memory()` instance when you need a non-default `db_path`/`model`/`store`, or want to
call `remember`/`search`/`forget` yourself, outside the run lifecycle:

```python
from runa import Memory

memory = Memory()
await memory.remember("User prefers email over phone.", user_id="user-42")
matches = await memory.search("how should I contact this user?", user_id="user-42")
await memory.forget(matches[0].id, user_id="user-42")
```

`Memory()` means `db/runa.db` (the same file `SQLiteSession` uses), `sqlite-vec`, and OpenAI's
`text-embedding-3-small`. Nothing to configure for the common case. Every memory is scoped by
`user_id`, so one user's facts never surface in another's search.

Embedding is the one thing memory needs that the agent's own `model` doesn't cover: it goes to
OpenAI whatever that model is, so an agent with `memory` set needs `OPENAI_API_KEY` as well. See
[Where embeddings go](deployment.md#where-embeddings-go) to send it through a gateway instead.

Remembering a near-duplicate of an existing memory skips the write and returns the existing id
instead, so restating the same fact across conversations doesn't pile up duplicates.

## A Different Store

A bare `Memory()` stores wherever `runa.db` resolves: the local `db/runa.db`, or the shared
Postgres once `RUNA_DATABASE_URL` is set. Going multi-replica needs no code change.

`store=` is for a backend Runa does not ship, and swaps storage without touching how embeddings
work. It takes a `VectorStore`: three async methods, no inheritance. Each payload is a dict of
`user_id`, `text` and `metadata`, and `partition` is the `user_id` a search or delete is scoped
to:

```python
from runa import Memory
from runa.db.vectors import VectorMatch


class HostedVectors:
    async def add(self, *, payload: dict, embedding: list[float]) -> int: ...

    async def nearest(
        self, *, embedding: list[float], k: int, partition: str | None = None
    ) -> list[VectorMatch]: ...

    async def delete(self, *, item_id: int, partition: str | None = None) -> None: ...


memory = Memory(store=HostedVectors())
```

## Memory vs. Session vs. Knowledge

Three kinds of context are easy to mix up:

```text
Session   = this conversation's history
Memory    = durable facts about a user, learned over many conversations
Knowledge = the application's own documents, not learned from conversations at all
```

For a plain get/set/delete/clear cache, with no embeddings and no `user_id` scoping, see
[Cache](cache.md) instead. For application documents searched by meaning, see
[Knowledge](knowledge.md).

## Example

```python
--8<--"examples/07_memory/auto_memory.py"
```

```python
--8<--"examples/07_memory/llm_memory.py"
```

More in [`examples/07_memory/`](https://github.com/Benybrahim/runa/tree/main/examples/07_memory).
