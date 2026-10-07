# ADR-0005: `runa.db` repeats its three-way branch once per concern, deliberately

- Status: accepted
- Date: 2026-10-07

## Context

`runa/db/__init__.py` holds eight factories -- `session`, `sessions`, `traces`,
`evals`, `memory_store`, `knowledge_store`, `vector_store`, `cache` -- and each
one opens with the same two lines:

```python
if ephemeral():
    ...
if (url := shared_url()) is not None:
    ...
```

Eight copies of each, followed by the same three-branch shape and three deferred
imports. A design review flagged this as duplicated decision-making in the one
module whose docstring claims to be "the only place that answers" where state
lives, and recommended resolving the backend once into one of three objects
implementing the eight factory methods, leaving each factory a delegation.

The count is accurate. The conclusion does not follow, and this ADR records why,
because the shape is conspicuous enough to be re-proposed.

## Decision

Keep one function per concern, each branching inline. Do not introduce a
per-backend object.

**The two axes are not symmetric, and the refactor trades the cheap one for the
expensive one.** The module is a matrix of concerns against backends. Backends
are fixed at three by design, not by accident: `Backend` in
[CONTEXT.md](../../CONTEXT.md) defines one as "one of the three answers
`RUNA_DATABASE_URL` can give" -- shared, own file, or nothing. That is a closed
question with three answers, not an open list. Concerns are the axis that grows,
once per primitive that needs to persist anything: `cache` and `vector_store`
both arrived after the original resolution, and `runa.db` has been edited along
the concern axis in four commits and along the backend axis in exactly one, the
commit that created it.

Today a new concern is one function in one file. Behind a per-backend object it
is a method on each of three classes plus a delegating factory: four edits in
four files, and a reviewer who adds three of the four gets a `TypeError` from an
incomplete interface rather than a missing function. The refactor optimizes the
frozen axis by pessimizing the moving one.

**A fourth backend has already arrived, and it cost nothing here.**
`RedisCache` (`cache/redis.py`) is a real, documented, tested fourth backend. It
required zero edits to `runa.db`, because it is one concern's backend, reached
by name (`RedisCache("redis://...")`) rather than by asking where state lives.
That is the general case: a backend for one concern is that concern's adapter,
and only a deployment-wide answer to `RUNA_DATABASE_URL` belongs in this module.
The premise that a fourth backend means eight coordinated edits is contradicted
by the fourth backend this repo actually has. A per-backend object would have to
encode "Redis, but only for the cache" as a special case -- the exact asymmetry
that currently costs nothing.

**The cited asymmetries are decisions, not drift.** `memory_store` and
`knowledge_store` pair their in-process backend inline while naming the other
two as modules. That is the recorded outcome of
[ADR-0001](0001-vector-store-is-app-private.md)'s amendment:
`EphemeralMemoryStore` and `EphemeralKnowledgeStore` were deleted because
nothing reached them by name, while the SQLite and Postgres adapters stayed
because `Memory(store=...)` is a sanctioned override with documented import
paths. `Concern` in CONTEXT.md states the same thing. Reading the asymmetry as
evidence that the pattern is not holding together inverts it: the pattern bent
exactly where a recorded decision bent it.

Rejected alternatives:

- **Three backend objects with eight methods each.** The review's
  recommendation. Rejected on the axis argument above. It also cannot reduce the
  24 deferred imports, only relocate them: a backend module that imports its
  eight adapters at module scope makes `db.cache()` pay for sessions, traces,
  evals, memory, knowledge and vectors, so each method must defer its own import
  anyway. The branch count drops from eight to one; the import count does not
  move.
- **A `_resolve(ephemeral=..., postgres=..., sqlite=...)` helper taking three
  thunks.** Removes the repeated branch without new files. Rejected because a
  lambda per backend per concern reads worse than the `if` it replaces, and it
  defeats the property that pays for the duplication: each factory reads top to
  bottom as the whole answer for its concern.
- **Dynamic dispatch on the naming convention** (`runa.<concern>.<backend>`,
  `<Backend><Concern>`). The convention is regular enough to make this work.
  Rejected because the constructors are not uniform -- Postgres takes a URL,
  SQLite a path, in-process neither -- and because it would trade a grep-able,
  type-checked call site for a string, which is the wrong direction for a module
  every storage question routes through.

## Consequences

- Eight copies of a three-line branch stay. The cost is paid in reading, not in
  behavior: the branches cannot drift apart silently, because all eight resolve
  through the same `ephemeral()` and `shared_url()`, which are the two functions
  that actually hold the decision. "Decided once" is true of the *question*, and
  that is what the module docstring claims.
- Adding a concern stays one function in one file, which is the edit this module
  keeps getting.
- Adding a deployment-wide backend -- a fourth answer to "where does state
  live" -- would be eight coordinated edits, and is the case that would reopen
  this ADR. Nothing suggests a fourth answer: the question has three, and a
  backend serving one concern has `RedisCache` as its pattern instead.
- This ADR does not defend duplication in general. It defends an 8x3 dispatch
  table written out longhand, where one axis is closed and the other grows, and
  where the alternative moves the per-item edit from one file to four.
