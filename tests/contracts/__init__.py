"""tests/contracts/: one file per store interface, holding what every backend of it must do.

`runa.db` resolves six concerns -- sessions, memory, knowledge, the cache, traces, eval history --
and each one has three adapters behind one interface. A deployment that sets
`RUNA_DATABASE_URL=postgresql://...` is relying on the new backend behaving like the file it had
before, so the thing worth asserting is the interface's contract, once, rather than each adapter's
own SQL three times over.

Each module here exports a `CONTRACT` list of checks and the `Check` type they share. A check takes
the store under test plus one unique tag it is free to write under, and nothing else: no `tmp_path`,
no fixture, no knowledge of which backend it got. That tag is what lets the same checks run against
a live shared Postgres, where every test in a session shares one set of tables and a fixed id
would collide with the run before it.

Who drives them:

- `tests/test_cache.py`, `tests/test_session_store.py`, `tests/test_memory.py`,
  `tests/test_knowledge.py`, `tests/tracing/test_store.py` and `tests/eval/test_store.py` drive
  every contract over the backends a plain `make test` can reach: the SQLite adapters and the
  `memory://` ones.
- `tests/test_postgres.py` and `tests/test_postgres_observability.py` drive the same checks over
  the Postgres adapters, which need a live server, and are skipped without one.

What stays in those files is what is genuinely one backend's own: SQLite's `trace_id` migration
(only a local file can predate the column), `MemoryCache` not outliving the process, the Postgres
exporter, and `RUNA_DATABASE_URL` routing itself.
"""
