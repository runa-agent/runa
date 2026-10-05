# ADR-0001: the vector store is app-private, the two typed stores stay public

- Status: accepted
- Date: 2026-10-05

## Context

`Memory` and `Knowledge` both store embeddings with a payload row and search
them by nearest neighbor. They did it through six adapters, three per concern,
holding about 480 lines between them. Only the two SQLite adapters shared
anything: `db/vectors.py`'s `VectorTable` held the `vec0` pairing. The Postgres
and in-process adapters hand-rolled the same shape twice each, which is how
three implementations of one contract drifted:

- the in-process adapter kept payloads as live objects while the other two
  encoded them as JSON, so a value that could not survive a round trip behaved
  differently under `memory://` than under SQLite or Postgres;
- the in-process adapter numbered rows by how many it held, so an `add` after a
  `delete` could reuse an id a surviving row still had.

`MemoryStore` and `KnowledgeStore` are public: both are in their package's
`__all__`, both are the documented escape hatch for `Memory(store=...)` and
`Knowledge(store=...)`, and `docs/memory.md` and `docs/knowledge.md` show
`PostgresMemoryStore(...)` and `PostgresKnowledgeStore(...)` by name.

## Decision

Introduce one `VectorStore` interface with three adapters under
`runa/db/vectors/`, and keep `MemoryStore` and `KnowledgeStore` exactly as they
are. The six named adapter classes and their import paths stay; each becomes a
thin pairing of its concern's spec with the matching vector adapter.

`VectorStore` is app-private and framework-public: an application never holds
one, and it is not a 15th primitive, but it is a real extension point for
someone adding a backend, with `tests/contracts/vector.py` as its conformance
suite.

The rejected alternative was promoting `VectorStore` to the one public seam and
deprecating the two typed stores. It gives a smaller type surface, and it was
rejected because it breaks a sanctioned override to solve a problem that was
never in the two interfaces: the duplication was entirely in the
implementations. It would also force every custom store to implement a
partition concept that `Knowledge` itself does not use.

## Consequences

- The duplication is gone: one `add`/`nearest`/`delete`/`clear` per backend
  instead of one per backend per concern, and L2 distance has one implementation
  per backend rather than two.
- Both divergences above are now contract checks, so they cannot come back
  silently in a fourth backend.
- No public name, import path or documented example changed, and the generated
  DDL is byte-identical, so existing `db/runa.db` files and Postgres tables are
  unaffected.
- `Memory` and `Knowledge` still have two separate public interfaces for two
  separate concerns. A future review will likely notice the two thin mapping
  classes and propose collapsing them. That is this ADR's subject: the cost of
  collapsing is a breaking change to `store=`, and the benefit is type surface
  only.
- A JSON column is still stored as `TEXT` in Postgres rather than `jsonb`.
  `CREATE TABLE IF NOT EXISTS` leaves an existing table alone, so emitting
  `jsonb` would give new deployments a column type old ones do not have.
  `Column(json=True)` is what makes that migration one place to change.
