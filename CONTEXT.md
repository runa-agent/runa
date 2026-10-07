# CONTEXT.md

Runa's domain language. The 14 primitives an application names are defined in
[RUNA.md](RUNA.md) and are not repeated here. What this file holds is the
internal vocabulary: terms that appear in module names, docstrings and commit
messages but that an application never writes.

Use these words as defined. Where a term says "not", that synonym is one
previous code or docs drifted to, so it is worth avoiding deliberately.

## Storage

**Backend.** One of the three answers `RUNA_DATABASE_URL` can give about where
state lives: a local SQLite file, a shared Postgres database, or nothing beyond
this process (`memory://`). Chosen once, in `runa.db`, so nothing above it names
one. _Not_: driver, database (a backend is the choice, not the server).

**Adapter.** The concrete module implementing one concern on one backend, for
example `session/postgres.py`. _Not_: provider, which is the model-side word.

**Concern.** One kind of state `runa.db` resolves: sessions, memory, knowledge,
the cache, traces, eval history. A concern has one interface and three backends
behind it. Four of the six give each backend its own named adapter; `memory` and
`knowledge` name only their SQLite and Postgres ones, because their adapters are
a spec over the shared `VectorStore` and the in-process pairing is one line
`runa.db` holds inline. _Not_: service, store (a store is the object, not the
kind).

**Vector store.** The storage `Memory` and `Knowledge` share: embeddings, the
payload row beside each one, and nearest-neighbor search over them. It is
plumbing, not a [primitive](RUNA.md), and it is app-private: an application
holds a `Memory` or a `Knowledge`, never a `VectorStore`, and `Memory(store=...)`
still takes a `MemoryStore`. See [ADR-0001](docs/adr/0001-vector-store-is-app-private.md).

**Spec.** What one concern's vector storage is called, holds, and is scoped by
(`VectorSpec`). It is the whole of what `Memory` and `Knowledge` differ by, once
the storage is shared: a name, its payload columns, and its partition.

**Payload.** The columns stored beside an embedding, and what comes back from a
search as a `dict` before a concern maps it to its own match type. A memory's
payload is its user, text and metadata; a knowledge chunk's is its text and
source. _Not_: metadata, which is one specific memory payload column.

**Partition.** A payload column that scopes vector search rather than filtering
it: a partitioned search returns that partition's `k` nearest rows, never a
global top-k another partition's rows could crowd out. `Memory` partitions by
`user_id`; `Knowledge` is application-scoped and declares none. `None` is a
partition of its own, not "any". _Not_: tenant, filter, scope key.

**Corpus.** The whole set of chunks one `Knowledge` would store: every supported
file under its directory, chunked, plus the embedding model that gives those
chunks meaning. A corpus is the unit an ingest replaces -- there is no partial
ingest -- and the unit a version identifies. _Not_: index, which names the
storage rather than the contents.

**Fingerprint.** The hash `Knowledge` takes of its corpus's inputs -- each
file's relative path and bytes, the model, its dimensions, the chunk shape -- to
decide whether the store already holds what an ingest would write. Contents, not
timestamps, so a fresh checkout or a container build is the same corpus. _Not_:
checksum, etag.

**Version.** A fingerprint as the store holds it: one opaque string per vector
store, written by `reset` in the same transaction as the rows it describes and
read back by `version()`. The store's word, because the store cannot know a
version is a hash of files; `Knowledge` is the only concern that keeps one. See
[ADR-0002](docs/adr/0002-ingest-state-belongs-to-the-store.md). _Not_: schema
version, which is `RunState`'s unrelated `schema_version`.

## Runs

**Tool call.** One executed call as the turn loop recorded it (`ToolCall`): the
tool's name, the arguments string the model produced, and the result string that
went back to it. This is the gradeable record of what an agent did, which a
`"tool"` span is not -- a span's input and output have been through the tracing
privacy policy first. See
[ADR-0003](docs/adr/0003-eval-evidence-is-recorded-not-traced.md). _Not_: tool
call record, tool span.
