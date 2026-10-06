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

Reach for a `Knowledge()` instance for a non-default `directory`/`model`/`store`, or to
call `.search`/`.ingest()` yourself:

```python
from runa import Knowledge

knowledge = Knowledge("app/knowledge")  # the default directory
matches = await knowledge.search("refund policy")
```

`Knowledge()` means `app/knowledge/`, `db/runa.db` (the same file `SQLiteSession`/`Memory` use),
`sqlite-vec`, and OpenAI's `text-embedding-3-small`. Discovery, chunking, embeddings, and vector
storage are entirely internal. Put Markdown, PDF, plain text, or CSV files under
`app/knowledge/`, and `.search` ingests them for you, so no manual `.ingest()` call is needed:
not on first use, and not after an edit either. Each search compares a fingerprint of the
directory, the embedding model and the chunk shape against the one the store recorded, and
re-ingests only when they differ. Edit a file and the next search picks it up; change nothing and
the corpus is embedded once, however many processes, replicas or requests search it.

```python
await knowledge.ingest()  # returns how many chunks it stored
```

Calling `.ingest()` yourself forces a rebuild, which you need only to re-embed a corpus whose
files did not change. It is always a full rebuild: it empties whatever was stored before and
re-chunks and re-embeds every supported file currently in the directory, so an edited or deleted
file never leaves stale chunks behind.

## A Different Store

A bare `Knowledge()` stores wherever `runa.db` resolves: the local `db/runa.db`, or the shared
Postgres once `RUNA_DATABASE_URL` is set. Going multi-replica needs no code change.

`store=` is for a backend Runa does not ship: four async methods, `add`/`search`/`reset`/`version`,
no inheritance. `reset` records the fingerprint its chunks were built from and `version` reports
it back, which is what lets a second process tell an ingested corpus from one it has to build.
Note `dimensions`, which must match your embedding model:

```python
from runa import Knowledge
from runa.knowledge.postgres import PostgresKnowledgeStore

knowledge = Knowledge(store=PostgresKnowledgeStore("postgresql://...", dimensions=1536))
```

## Example

```python
--8<--"examples/08_knowledge/auto_knowledge.py"
```

More in [`examples/08_knowledge/`](https://github.com/Benybrahim/runa/tree/main/examples/08_knowledge).
