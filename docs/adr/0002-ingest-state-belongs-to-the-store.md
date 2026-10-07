# ADR-0002: whether a corpus is ingested is the store's state, not the `Knowledge` object's

- Status: accepted
- Date: 2026-10-06

## Context

`Knowledge.search` ingested lazily, guarded by `self._ingested`, a `bool` on the
instance. `Knowledge.ingest` is a full rebuild: it empties the store and
re-embeds every chunk of every file in the directory.

Three decisions, each defensible alone, composed into a real cost:

- `Agent(knowledge="auto")` resolves the setting by constructing a fresh
  `Knowledge()` (`agent.py`'s `_resolve_retrieval_setting`).
- `runa.serve` builds an `Agent` per request, deliberately: a module-level agent
  shared across requests would interleave users' histories.
- Nothing above `Knowledge` memoized anything.

So every HTTP request built a `Knowledge` with `_ingested=False` pointing at the
same shared store, and re-embedded the whole corpus before answering. Measured
on a five-chunk corpus with embeddings stubbed, three requests cost three
rebuilds and 15 embedded texts where one would do.

The scope mismatch is the whole of it. "Has this corpus been ingested" is a fact
about the store, which is deployment-scoped and may be shared across processes
and replicas. It was recorded on the `Knowledge` object, which is request-scoped.
The two lifecycles each document the other's cost away, and an application
following both conventions got neither.

Severity beyond the bill: `ingest` empties the store and then adds chunks one
await at a time, so a concurrent search reads a corpus that is briefly empty.
That surfaces as intermittently missing retrieval rather than as an error, and
under Postgres every replica did it to the one shared table.

## Decision

Move the decision to the layer whose scope matches it. `VectorStore.clear()`
becomes `reset(*, version=None)`, and `VectorStore` gains `version()`;
`KnowledgeStore` mirrors both. `Knowledge.search` ingests when the store's
recorded version differs from a fingerprint of what it would write, and
`_ingested` is gone.

The version is written in the same transaction as the deletion of the rows it
describes. That is the reason this lives in the adapters rather than in a
general-purpose side table -- `runa.cache` would have done -- and it is the
deciding argument: a version that can outlive its chunks fails *closed*. Store
dropped, version row kept, and `search` concludes "already ingested" over an
empty corpus and returns nothing, forever. Re-embedding is a bill; silently
empty retrieval is a wrong answer.

The fingerprint hashes contents, not `(size, mtime)`. A container build or a
fresh checkout rewrites every timestamp, so a timestamp-keyed fingerprint would
re-embed the whole corpus on every deploy, which is the case the fingerprint
exists to prevent. Reading the files is milliseconds against the seconds and the
dollars of embedding them, and `search` pays it immediately before a network
call to embed its query. The model, its dimensions and the chunk shape are in the
hash too: chunks embedded by a different model are not comparable to this one's
query vectors, so changing `model=` has to invalidate the corpus exactly the way
editing a file does.

Rejected alternatives:

- **A module-level guard keyed on `(directory, store identity)`.** Cheaper, and
  it removes the per-request rebuild. Rejected because it is process-scoped state
  standing in for store-scoped state -- the same category error one level up --
  and it leaves staleness undetectable, which is what the manual `.ingest()` call
  in `docs/knowledge.md` existed to work around.
- **A deploy-time `runa knowledge ingest` command, with `search` never
  ingesting.** This is the asset-pipeline shape, and it is correct in production.
  Rejected because the fingerprint subsumes it: precompiling becomes an
  optimization rather than a required step, and the menu is omakase -- a new
  mandatory deploy step is configuration by another name.
- **Keeping the version in `runa.cache`.** See the atomicity argument above.

## Consequences

- A corpus is embedded once per change rather than once per `Knowledge`
  instance: one rebuild across every process, replica and request, where before
  each paid its own.
- Editing a source file is picked up by the next `search`. `docs/knowledge.md` no
  longer has to tell people to call `.ingest()` after an edit; `ingest()` stays
  public as the force path, which is RUNA.md's "close the second path" test
  passing rather than a second shape.
- `KnowledgeStore` is four methods, not three, and `VectorStore` five, not four.
  Both are documented extension points (ADR-0001), so this is a breaking change
  for an out-of-tree store. It is the cost the atomicity argument buys, and it
  lands in one place per backend.
- `reset`/`version` is the one pair only `Knowledge` uses. `Memory` resets with
  no version and never reads one. Pushing it down to the shared `VectorStore`
  rather than keeping it above is what makes the write transactional, which only
  an adapter can do.
- Every backend gets a `{name}_meta` table, including `memory_meta`, which stays
  empty. The DDL is `CREATE TABLE IF NOT EXISTS` applied on every connect, so
  existing `db/runa.db` files and Postgres databases gain it without a migration,
  and an existing corpus reports `version() is None` and is re-ingested once.
- **Not fixed by this ADR:** the empty window during a rebuild. `reset` is
  transactional, but the `add` calls after it are not, so a corpus is still
  briefly partial, and two cold processes seeing the same stale version will both
  rebuild. This drops that window from every request to once per corpus change,
  which is a reduction and not a fix. Closing it means ingesting into a new
  generation and flipping a pointer instead of emptying in place -- a separate
  change to the same two methods.
