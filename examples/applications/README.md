# Applications

Full apps, combining several primitives the way a real project would.

* **`tour.py`** -- one agent wired with tools, guardrails, approval, subagents, hooks, tracing,
  and eval, all in one run. Start here if you want to see the primitives interact.
* **`api.py`** -- serving an agent over HTTP with FastAPI, `SQLiteSession`, single-worker.
* **`api_postgres.py`** -- the multi-process-safe version: Postgres, `pgvector`, and Redis.
* **`Dockerfile`** / **`docker-compose.yml`** -- deploying `api_postgres.py` with Postgres and
  Redis alongside it.
