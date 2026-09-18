r"""Serving an Agent over HTTP, backed by Postgres + `pgvector` and Redis instead of SQLite.

The multi-process-safe counterpart to `api.py`: that example's `SQLiteSession` (and
`sqlite-vec`-backed `Memory`/`Knowledge`) only tolerate one process touching `db/runa.db` at a
time (see `session.py`'s docstring). Swap in `runa.db.postgres`'s `PostgresSession`/
`PostgresMemoryStore`/`PostgresKnowledgeStore` and this same FastAPI app can run behind
`uvicorn --workers N`, or as several replicas, all sharing one database -- no other code changes,
since `Memory(store=...)`/`Knowledge(store=...)`/`session=` are exactly the escape hatches
`memory.py`/`knowledge.py`/`session.py` document for this.

`RedisCache` plays a different role: not something `Agent.run` touches automatically (`Cache` is
plain application-level caching -- see `cache.py`), so `lookup_order_status` below calls it by
hand, the way any tool's own code would.

Needs the `runa[postgres,redis]` extras (`uv add "runa[postgres,redis]"`) alongside `fastapi`/
`uvicorn`, none of which are core `runa` dependencies.

Run it (see docker-compose.yml for `postgres`/`redis` alongside this app):

    uv run uvicorn examples.applications.api_postgres:app --port 8000

Try it:

    curl -X POST localhost:8000/chat \\
        -H "content-type: application/json" \\
        -d '{"session_id": "user-42", "message": "Where is order #4821?"}'
"""

import os

from fastapi import FastAPI
from pydantic import BaseModel

from runa import Agent, Knowledge, Memory, tool
from runa.db.postgres import (
    DEFAULT_POSTGRES_DSN,
    PostgresKnowledgeStore,
    PostgresMemoryStore,
    PostgresSession,
)
from runa.db.redis import DEFAULT_REDIS_URL, RedisCache

_POSTGRES_DSN = os.environ.get("POSTGRES_DSN", DEFAULT_POSTGRES_DSN)
_REDIS_URL = os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)
_EMBEDDING_DIMENSIONS = 1536  # text-embedding-3-small, Memory/Knowledge's own default model

cache = RedisCache(_REDIS_URL)


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
    """A support agent with memory and a knowledge base, both stored in Postgres."""

    name = "support_agent"
    instructions = "You are a helpful customer support assistant."
    tools = [lookup_order_status]
    memory = Memory(store=PostgresMemoryStore(_POSTGRES_DSN, dimensions=_EMBEDDING_DIMENSIONS))
    knowledge = Knowledge(
        store=PostgresKnowledgeStore(_POSTGRES_DSN, dimensions=_EMBEDDING_DIMENSIONS)
    )


app = FastAPI()
agent = SupportAgent()


class ChatRequest(BaseModel):
    """One turn of a conversation, keyed by `session_id` so history persists in Postgres."""

    session_id: str
    message: str


class ChatResponse(BaseModel):
    """What the agent said back, or the error if the run didn't complete."""

    output: str | None
    status: str
    error: str | None


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Run one turn for `request.session_id`, resuming its history from Postgres."""
    session = PostgresSession(request.session_id, _POSTGRES_DSN)
    run = await agent.run(request.message, session=session)
    return ChatResponse(output=run.output, status=run.status, error=run.error)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness/readiness probe target."""
    return {"status": "ok"}
