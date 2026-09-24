# Deployment

Everything up to here runs an agent on your machine. This page is about running one in front of
users: serving it over HTTP, keeping one deployment's state out of another's, and bounding what a
single run can cost you.

## Serving agents over HTTP

`runa serve` puts every agent under `app/agents/` behind an HTTP API. There is nothing to write:
the agents you already have are the endpoints.

```bash
uv add "runa-ai[serve]"
export RUNA_API_KEY=$(openssl rand -hex 32)
runa serve
```

```
runa serve running at http://127.0.0.1:8000 (bearer auth)
```

| Route | What it does |
| --- | --- |
| `GET /health` | Liveness. The one unauthenticated route, because a load balancer cannot hold a token. |
| `GET /agents` | Every agent this app serves, by declared `name`. |
| `POST /agents/{name}/runs` | Run one turn, return the finished `Run`. |
| `POST /agents/{name}/runs/stream` | The same turn as server-sent events. |

```bash
curl -X POST http://127.0.0.1:8000/agents/support_agent/runs \
  -H "Authorization: Bearer $RUNA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message": "where is my order?", "session_id": "conv-42"}'
```

```json
{
  "agent": "support_agent",
  "status": "completed",
  "output": "Your order shipped on Tuesday.",
  "error": null,
  "trace_id": "2ebeef9a3d224da3bd3bb55eb0af747a",
  "usage": {"input_tokens": 412, "output_tokens": 38, "total_tokens": 450, "requests": 2},
  "interruptions": []
}
```

`status` is the `Run`'s own, so a client tells "answered" from "paused for approval" from
"failed" without parsing prose, and `trace_id` is the handle for `runa traces show` when someone
reports that the answer was wrong.

The streaming route emits one JSON object per `data:` line: `{"type": "token", ...}` as the
answer is produced, then a single `{"type": "run", ...}` carrying the same payload as above, so a
streaming client still gets the trace id and the usage. A failure part-way through arrives as
`{"type": "error", ...}`: the HTTP status went out with the first byte, so an error cannot become
a 500, and silently truncating would leave a client unable to tell a finished answer from a
dropped connection.

### Authentication

`runa serve` requires `RUNA_API_KEY` and refuses to start without it. An agent endpoint spends
money on every call, so "nobody set the variable" is an error rather than a silently open door.

```
$ runa serve
error: RUNA_API_KEY is not set. Set it to the token clients must send as
`Authorization: Bearer <token>`, or pass --no-auth to serve without authentication.
```

Clients send it as `Authorization: Bearer <token>`. A missing token is `401`, a wrong one `403`.
If something in front of the container already authenticates, `runa serve --no-auth` makes that
choice explicit.

## One agent instance, one conversation

An `Agent` instance holds the conversation it is running. Two overlapping runs on the same
instance would read the same `self.history` and race to write it back, losing one conversation
into the other, so Runa refuses rather than letting it happen quietly:

```python
agent = SupportAgent()
await asyncio.gather(agent.run("one"), agent.run("two"))
# UserError: SupportAgent is already running: one Agent instance cannot run concurrently
# without a session, because both runs would share (and overwrite) `self.history`.
```

Two ways out, both cheap:

```python
# A session per run: each conversation's history is its own.
await asyncio.gather(
    agent.run("one", session=SQLiteSession("conv-a")),
    agent.run("two", session=SQLiteSession("conv-b")),
)

# Or an Agent per run.
await asyncio.gather(SupportAgent().run("one"), SupportAgent().run("two"))
```

`runa serve` builds a fresh agent per request and passes a session when the request carries a
`session_id`, so an app served this way satisfies the rule without doing anything. If you embed
Runa in your own web framework, build the agent **inside** the request handler, not at module
scope.

## More than one replica

SQLite is per-process by design. Three replicas with the default settings keep three disjoint
histories, and a dashboard that can only ever show one of them. Set one variable and everything
Runa persists moves to a shared database:

```bash
uv add "runa-ai[postgres]"
export RUNA_POSTGRES_DSN=postgresql://user:password@host:5432/runa
```

That covers sessions, memory, knowledge, **traces** and **eval history**. No code changes:
`list_traces(...)`, `runa traces`, `runa ui` and the exporter all follow the same variable, so a
trace written by one replica is readable from any of them.

Leave it unset and nothing changes: a single process keeps its own `db/runa.db`, which is the
right answer for one machine.

## Bounding a run

Three independent ceilings, each optional, each ending the run as `Run(status="error")` rather
than raising:

```python
class SupportAgent(Agent):
    max_turns = 10       # model calls (the default)
    max_tokens = 50_000  # total tokens this run may spend
    timeout = 30.0       # wall-clock seconds
```

`max_turns` bounds how many times a run calls the model; `max_tokens` bounds what those calls
cost, which a turn limit alone cannot; `timeout` bounds elapsed time, which neither can, and is
the only thing that saves you from a tool or a provider that hangs.

`max_tokens` on the agent is the run's whole budget. It is not
`ModelSettings(max_tokens=...)`, which caps the length of one response.

Cancelling a run (a dropped connection, a worker shutting down) propagates `CancelledError`
rather than being converted into an error result: a caller that cancels wants it to stop, not to
receive a verdict.

### Retries

Both model backends retry connection errors and `408`/`409`/`429`/`5xx` twice with jittered
backoff, honoring `retry-after`. Tune or disable it per agent:

```python
class SupportAgent(Agent):
    model_settings = ModelSettings(max_retries=5)  # 0 disables retrying outright
```

## Logs and user data

The default hooks log at INFO, and at INFO they name what happened and never carry content: an
agent's answer and a tool's result are user data. Content is logged at DEBUG only, and even there
it passes through the same redaction policy as tracing, so this governs your logs too:

```python
from runa.tracing import observe

observe(redact=["email", "card_number"])  # scrub these keys wherever they appear
observe(capture_outputs=False)            # keep outputs out of traces and logs entirely
```

## Retention

Everything Runa persists is append-only: every run adds a trace and its spans, every turn adds
session messages, every eval adds a run. Without a retention pass the database grows until the
disk does not. `runa prune` is that pass:

```bash
runa prune --older-than 30 --dry-run   # what would go
runa prune --older-than 30             # traces, sessions and eval runs older than 30 days
runa prune --older-than 90 --only traces
```

A session ages out by when it was last used, so an active conversation is never pruned out from
under a user. On SQLite the freed pages are reclaimed with `VACUUM`; on Postgres autovacuum
handles it, and no exclusive lock is ever taken on a database other replicas are writing to.

Wire it into whatever already runs on a schedule. There is no background thread doing this for
you, on purpose: a framework should not decide when to delete your data.

## Containers

`runa new` scaffolds a `Dockerfile` that serves the app:

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir uv && uv sync --frozen --extra serve

EXPOSE 8000

CMD ["uv", "run", "runa", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

`--host 0.0.0.0` is what makes the port reachable from outside the container; the default
`127.0.0.1` deliberately is not. `uv sync --frozen` installs exactly what `uv.lock` pins, so
commit the lockfile.

A deployment checklist, in the order things tend to go wrong:

- [ ] `RUNA_API_KEY` set (or `--no-auth`, deliberately)
- [ ] The model provider's key set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...)
- [ ] `RUNA_POSTGRES_DSN` set if more than one replica
- [ ] `/health` wired to the liveness probe
- [ ] `max_tokens` and `timeout` set on agents that face the public
- [ ] `runa prune` on a schedule
- [ ] Log level at INFO, not DEBUG, unless you mean to record user content
