r"""Serving an Agent over HTTP with FastAPI, deployed as a single-worker process.

FastAPI/uvicorn aren't Runa dependencies -- they belong to the app, added via
`uv add fastapi uvicorn`. One process, one worker: `SQLiteSession`/`runa.db`
skip cross-process locking (see `session.py`), so scale by running more
single-worker containers (each with their own `db/runa.db`), not more
workers in one.

Run it:

    uv run uvicorn examples.applications.api:app --port 8000

Try it:

    curl -X POST localhost:8000/chat \\
        -H "content-type: application/json" \\
        -d '{"session_id": "user-42", "message": "My order has not arrived"}'
"""

from fastapi import FastAPI
from pydantic import BaseModel

from runa import Agent, SQLiteSession


class SupportAgent(Agent):
    """A minimal support agent, standing in for a real `app/agents/` one."""

    name = "support_agent"
    instructions = "You are a helpful customer support assistant."


app = FastAPI()
agent = SupportAgent()


class ChatRequest(BaseModel):
    """One turn of a conversation, keyed by `session_id` so history persists in `runa.db`."""

    session_id: str
    message: str


class ChatResponse(BaseModel):
    """What the agent said back, or the error if the run didn't complete."""

    output: str | None
    status: str
    error: str | None


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Run one turn for `request.session_id`, resuming its history from `runa.db`."""
    session = SQLiteSession(request.session_id)
    run = await agent.run(request.message, session=session)
    return ChatResponse(output=run.output, status=run.status, error=run.error)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness/readiness probe target."""
    return {"status": "ok"}
