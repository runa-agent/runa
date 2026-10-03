r"""Serving an Agent over HTTP across several replicas, backed by Postgres and `pgvector`.

The multi-process-safe counterpart to `api.py`, and the point of this example is how little
differs: not one import, class or call below names a backend. `api.py`'s `SQLiteSession` (and
`sqlite-vec`-backed `Memory`/`Knowledge`) only tolerate one process touching `db/runa.db` at a
time, so `runa.db` reads `RUNA_DATABASE_URL` and hands every concern a Postgres-backed store
instead when it is set:

    export RUNA_DATABASE_URL=postgresql://runa:runa@localhost:5432/runa

With that set, this same app runs behind `uvicorn --workers N`, or as several replicas, all
sharing one database. With it unset, the identical code runs on one SQLite file, which is what
you want on a laptop. The sessions, memories and knowledge chunks all move together, so no
replica can end up reading history another one wrote somewhere else.

`RedisCache` is the one thing still chosen by hand, and deliberately so. A shared deployment
already has a cache (`db.cache()` returns a `PostgresCache` over the same database), so Redis is
worth naming only when you want hot keys off the query path. `Cache` is also plain
application-level caching that `Agent.run` never touches by itself, so `lookup_order_status`
calls it the way any tool's own code would.

Needs the `runa-ai[postgres,redis]` extras alongside `fastapi`/`uvicorn`, none of which are core
dependencies. See docs/deployment.md for a compose file running `postgres` alongside this app.

    uv run uvicorn examples.applications.api_postgres:app --port 8000

Try it:

    curl -X POST localhost:8000/chat \\
        -H "content-type: application/json" \\
        -d '{"session_id": "user-42", "message": "Where is order #4821?"}'
"""

import os

from fastapi import FastAPI
from pydantic import BaseModel

from runa import Agent, Knowledge, Memory, db, tool
from runa.cache.redis import RedisCache

cache = RedisCache(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))


@tool
async def lookup_order_status(order_id: str) -> str:
    """Look up an order's shipping status, caching each id's result in Redis for 5 minutes.

    order_id: the order number to look up, e.g. "4821"
    """
    cache_key = f"order_status:{order_id}"
    cached = await cache.get(cache_key)
    if cached is not None:
        return cached
    status = f"Order #{order_id} shipped and is in transit."  # a real lookup goes here
    await cache.set(cache_key, status, ttl=300)
    return status


class SupportAgent(Agent):
    """A support agent with memory and a knowledge base, both wherever `runa.db` resolves."""

    name = "support_agent"
    instructions = "You are a helpful customer support assistant."
    tools = [lookup_order_status]
    memory = Memory()
    knowledge = Knowledge()


app = FastAPI()
agent = SupportAgent()


class ChatRequest(BaseModel):
    """One turn of a conversation, keyed by `session_id` so history persists across replicas."""

    session_id: str
    message: str


class ChatResponse(BaseModel):
    """What the agent said back, or the error if the run didn't complete."""

    output: str | None
    status: str
    error: str | None


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Run one turn for `request.session_id`, resuming its history from the shared database."""
    session = db.session(request.session_id)
    run = await agent.run(request.message, session=session)
    return ChatResponse(output=run.output, status=run.status, error=run.error)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness/readiness probe target."""
    return {"status": "ok"}
