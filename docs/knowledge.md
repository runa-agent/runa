# Knowledge

Application and domain documents, retrieved by meaning before every turn. Unlike
[Memory](memory.md), Knowledge is not learned from conversations and is not scoped to a user.
Its source of truth is a directory of files on disk, put there by whoever built the app. Opt an
agent in with `knowledge`:

```python
class SupportAgent(Agent):
    name = "support_agent"
    knowledge = "auto"
```

## `"auto"`: retrieved automatically

Runa searches `app/knowledge/` before every turn and injects matches as a labeled block. No
manual `.search` calls, and no `tools=[...]` wiring needed.

```python
support = SupportAgent()  # knowledge = "auto"
support.run_sync("What's your refund policy?")  # sees matching chunks from app/knowledge/
```

## `"llm"`: the model decides when to look

The model gets a `search_knowledge` tool and calls it itself.

```python
class SupportAgent(Agent):
    name = "support_agent"
    knowledge = "llm"
```

## `Knowledge` Directly

Reach for a `Knowledge()` instance for a non-default `directory`/`db_path`/`model`/`store`, or to
call `.search`/`.ingest()` yourself:

```python
from runa import Knowledge

knowledge = Knowledge("app/knowledge")  # the default directory
matches = await knowledge.search("refund policy")
```

`Knowledge()` means `app/knowledge/`, `db/runa.db` (the same file `SQLiteSession`/`Memory` use),
`sqlite-vec`, and OpenAI's `text-embedding-3-small`. Discovery, chunking, embeddings, and vector
storage are entirely internal. Put Markdown, PDF, plain text, or CSV files under
`app/knowledge/`, and `.search` ingests them lazily on first use, so no manual `.ingest()` call
is needed for the common case.

Call `.ingest()` yourself to force a fresh rebuild, for example after editing a source file:

```python
await knowledge.ingest()  # returns how many chunks it stored
```

Ingesting is always a full rebuild: it clears whatever was stored before and re-chunks and
re-embeds every supported file currently in the directory, so an edited or deleted file never
leaves stale chunks behind.

## A Different Store

`store=` swaps the default `SQLiteKnowledgeStore` for a custom one. For a deployment where
multiple processes need to share one store, use `PostgresKnowledgeStore` (the `runa[postgres]`
extra):

```python
from runa import Knowledge
from runa.db.postgres import PostgresKnowledgeStore

knowledge = Knowledge(
    store=PostgresKnowledgeStore(dsn="postgresql://runa:runa@localhost:5432/runa")
)
```

## Example

```python
--8<--"examples/08_knowledge/auto_knowledge.py"
```

More in [`examples/08_knowledge/`](https://github.com/Benybrahim/runa/tree/main/examples/08_knowledge).
