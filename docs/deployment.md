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
| `POST /agents/{name}/runs` | Run one turn -- or finish a paused one -- and return the `Run`. |
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
  "interruptions": [],
  "state": null
}
```

Startup does the work a first request would otherwise do: the server resolves every agent class
as it boots (so a broken `app/agents/` fails the rollout, not a user's turn) and ingests the
[knowledge](knowledge.md) corpus of each agent that declares one, which is a read of every file
and an embedding call for every chunk. Both are costs a deploy should pay, not a user waiting on
an answer. The server accepts no connections until that finishes, so `/health` answering is
already the signal that a replica is warm -- and an agent with `knowledge` set now needs its
embeddings credential at boot rather than at its first retrieval.

`status` is the `Run`'s own, so a client tells "answered" from "paused for approval" from
"failed" without parsing prose, and `trace_id` is the handle for `runa traces show` when someone
reports that the answer was wrong.

The streaming route emits one JSON object per `data:` line: `{"type": "token", ...}` as the
answer is produced, then a single `{"type": "run", ...}` carrying the same payload as above, so a
streaming client still gets the trace id and the usage. A run that failed part-way through is
that same `run` event with `status="error"` -- one wire shape either way, since a run reports
every failure on its `Run`. Should the server itself break while streaming, a bare
`{"type": "error", ...}` arrives instead: the HTTP status went out with the first byte, so an
error cannot become a 500, and silently truncating would leave a client unable to tell a finished
answer from a dropped connection.

### Resuming a paused run

A run that hits a tool needing [human approval](approval.md) comes back paused, with the calls
waiting on a decision and the `state` that resolves them:

```json
{
  "agent": "refund_agent",
  "status": "paused",
  "output": null,
  "interruptions": [
    {"name": "issue_refund", "arguments": "{\"amount\": 75}", "call_id": "call_1",
     "agent": "refund_agent"}
  ],
  "state": {"schema_version": 4, "agent_name": "refund_agent", "...": "..."}
}
```

Post that `state` back where the message would have gone, with one decision per interruption:

```bash
curl -X POST http://127.0.0.1:8000/agents/refund_agent/runs \
  -H "Authorization: Bearer $RUNA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"state": {...}, "decisions": [{"call_id": "call_1", "approve": true}],
       "session_id": "conv-42"}'
```

| Field of a decision | What it does |
| --- | --- |
| `call_id` | The interruption being decided, by the id the paused payload reported. |
| `approve` | `true` runs the paused call, `false` skips it. |
| `always` | Apply the decision to every later call to that tool, not just this one. |
| `rejection_message` | Fed back to the model instead of the default rejection text. |

The response is the same payload as any other run: completed, errored, or paused again if the
agent asks for something else. `state` and `message` are the same argument of `Agent.run`, so a
body carrying both (or neither) is a `422`, and the streaming route takes the same body -- a
client that streamed a run that paused resumes it streaming. Send the `session_id` the run paused
on: a session-backed run persists nothing while paused, and the whole turn is saved once the
resumed run finishes. An interruption you send no decision for pauses the run again rather than
being decided for you.

The `state` blob is a capability, not an opaque receipt: it names the agent, carries the run's
context and its approval ledger, and nothing on the server tracks that a given blob was already
spent -- posting an approved one twice runs the approved call twice, so treat it as single-use
the way you would any other idempotency key. `RUNA_API_KEY` is a server-to-server credential for
exactly this reason -- put `runa serve`
behind your own API and keep the blob between your backend and Runa. Handing it to a browser
means handing out an approval, so sign it or stash it under a key of your own first.

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

### Your own server

`runa serve` is the short way to run that app, not the only way. `runa new` scaffolds an `asgi.py`
holding the app itself, which is the two lines you would otherwise have to go looking for:

```python
from pathlib import Path

from dotenv import load_dotenv
from runa.serve import create_app, resolve_api_key

load_dotenv()

app = create_app(Path(__file__).parent, api_key=resolve_api_key(no_auth=False))
```

Point any ASGI server at it, with whatever process model your platform expects:

```bash
uvicorn asgi:app --host 0.0.0.0 --port 8000 --workers 4
gunicorn asgi:app -k uvicorn.workers.UvicornWorker -w 4
```

Or mount the agents inside an API you already have, under a path of your choosing:

```python
from asgi import app as agents

api.mount("/ai", agents)
```

`runa serve` does not read `asgi.py` -- it calls `create_app` itself, so its `--host`/`--port`/
`--no-auth` flags stay the CLI's contract. Editing `asgi.py` changes the server you run from
`asgi.py`; reach for it when you want middleware, a mount path, or a process manager that
`runa serve` does not give you, and stay on `runa serve` when you don't.

Several worker processes is one of those things `runa serve` does not give you, and the reason is
visible in the line above: a worker is forked from an *import string* a supervisor re-imports per
process, and `create_app(root, api_key=...)` takes arguments, so `asgi.py` -- not a flag -- is
where an app gets a name to be addressed by. `runa serve` runs one process; `uvicorn asgi:app
--workers 4` runs as many as you ask for.

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
    agent.run("one", session="conv-a"),
    agent.run("two", session="conv-b"),
)

# Or an Agent per run.
await asyncio.gather(SupportAgent().run("one"), SupportAgent().run("two"))
```

`runa serve` builds a fresh agent per request and passes a session when the request carries a
`session_id`, so an app served this way satisfies the rule without doing anything. If you embed
Runa in your own web framework, build the agent **inside** the request handler, not at module
scope.

The agent *class* is the other half: it is resolved once, when the server starts. `runa serve`
imports `main.py` and `app/` at boot, never per request, so whatever your app keeps at module
level -- an HTTP client, a connection pool, a cache -- is built once and stays built, and a
module that fails to import stops the server from starting rather than failing one route at a
time. Editing `app/agents/` therefore takes a restart; `uvicorn asgi:app --reload` does that for
you while you work. `runa ui` reads the app the same way, so its Agents page is the app as
imported at startup too.

One process serves one app. Importing an app runs its `main.py` under the names `main` and
`app.*`, and its local state resolves against one directory -- one fact, said once, by the `root`
you hand `create_app`. Runa says it out loud rather than leaving it to chance: mounting a second
Runa app in the same process -- a second `create_app`, or a second `project.use` naming a
different project -- raises a `UserError` instead of quietly pointing both at whichever one was
built last. Serve the second app from a second process.

## More than one replica

SQLite is per-process by design. Three replicas with the default settings keep three disjoint
histories, and a dashboard that can only ever show one of them. Set one variable and everything
Runa persists moves to a shared database:

```bash
uv add "runa-ai[postgres]"
export RUNA_DATABASE_URL=postgresql://user:password@host:5432/runa
```

That covers sessions, memory, knowledge, the cache, **traces** and **eval history**. No code
changes: `runa serve`, `Memory()`, `db.traces()`, `runa sessions`, `runa traces` and `runa ui`
all resolve their backend through `runa.db`, so a session or trace written by one replica is
readable from any of them.

The cache moves with it, into a `cache_entries` table in the same database. No second service to
run. Reach for `RedisCache("redis://...")` explicitly if you want hot keys off the query path:

```python
from runa.cache.redis import RedisCache

cache = RedisCache("redis://localhost:6379/0")
```

Leave the variable unset and nothing changes: a single process keeps its own `db/runa.db`, which
is the right answer for one machine. `sqlite:///data/runa.db` relocates that file if you need it
somewhere specific.

### Testing against the store

`memory://` is the third answer to the same question: every concern resolves to an in-process
store that dies with the process. No file to clean up, no server to run, and the code under test
goes through the same interfaces a deployment does.

```python
import pytest
from runa import db


@pytest.fixture(autouse=True)
def in_memory_store(monkeypatch):
    monkeypatch.setenv("RUNA_DATABASE_URL", "memory://")
    db.reset_ephemeral()  # the stores are process-wide, like a file
```

It is a testing and local-prototyping backend, not a deployment one: nothing is persisted, and
two replicas would share nothing at all.

To run the shared path locally before you ship it, point the variable at containers:

```yaml
# docker-compose.yml
services:
  postgres:
    image: pgvector/pgvector:pg17
    environment: { POSTGRES_USER: runa, POSTGRES_PASSWORD: runa, POSTGRES_DB: runa }
    ports: ["5432:5432"]
  app:
    build: .
    environment:
      RUNA_DATABASE_URL: postgresql://runa:runa@postgres:5432/runa
    ports: ["8000:8000"]
    depends_on: [postgres]
```

The `pgvector` image matters: `Memory` and `Knowledge` need that extension, and Runa creates it
on first connect. In production the variable comes from your platform's environment or secrets,
not from a compose file.

## Where embeddings go

An agent's `model` picks the provider for its *turns*. [Memory](memory.md) and
[Knowledge](knowledge.md) also need text turned into vectors, and that goes to OpenAI's
`text-embedding-3-small` whatever the agent's own model is: a `claude-sonnet-5` agent with
`memory = "auto"` needs `OPENAI_API_KEY` set as well, and says so on the first run if it isn't.
It is the one place Runa asks for a vendor no setting named -- OpenAI is the only provider Runa
routes chat to that also serves embeddings.

A deployment that can't send text there -- Azure, a corporate gateway, something self-hosted --
redirects it the way it redirects the database, with an environment variable and no code change:

```bash
export RUNA_EMBEDDINGS_BASE_URL=https://gateway.example.com/v1/
export RUNA_EMBEDDINGS_API_KEY=...  # only if the token isn't OPENAI_API_KEY's
```

Anything speaking OpenAI's `POST /embeddings` wire format works. `Memory(model=...)`/
`Knowledge(model=...)` still name the model (with `dimensions=` for one Runa doesn't know the
vector size of), because that is the app's decision; the endpoint is the deployment's.

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

## Upgrading Runa with a database already in place

Tables are created on first use, and brought up to date on every connect after that. A release
that adds a column to one of Runa's own tables -- sessions, traces, eval history, the cache --
reaches databases that already exist: each connect diffs what Runa declares against what the
database has and adds what's missing. Nothing to run, no migration directory, and the same
behavior for the local `db/runa.db` and a shared Postgres.

Only additive changes are automatic, and that is deliberate. A column that cannot be filled in
for the rows already there -- a `NOT NULL` column, a new primary key -- has a backfill to decide,
and Runa refuses to guess it:

```text
runa.db.schema.SchemaOutOfDate: this database's eval_cases table has no tier column, and only a
nullable column with no default can be added to a table that already exists. Add
eval_cases.tier by hand (with whatever value its existing rows should get), or delete the
database to recreate it from the current schema.
```

Both ways out are in the message. For a local file, deleting it is usually the honest answer;
for a shared Postgres, run the `ALTER TABLE` through whatever migration tooling you already
have. A column Runa no longer declares is left alone, so rolling a replica back is safe.

Your own tables are your own business: Runa reconciles what it declares and nothing else.

## Retention

Everything Runa persists is append-only: every run adds a trace and its spans, every turn adds
session messages, every eval adds a run. Without a retention pass the database grows until the
disk does not. Runa ships no retention command, on purpose: deciding when to delete your data is
your database's job, and yours. Every child table cascades from its parent, so deleting the
parent rows is the whole pass:

```sql
DELETE FROM traces     WHERE start_time < extract(epoch FROM now() - interval '30 days');
DELETE FROM agent_sessions WHERE updated_at < now() - interval '30 days';
DELETE FROM eval_runs  WHERE created_at < (now() - interval '30 days')::text;
```

Spans, `agent_messages` and `eval_cases` go with their parents via `ON DELETE CASCADE`. A session
ages out by when it was last used, so an active conversation is never deleted out from under a
user. Autovacuum reclaims the space, and no exclusive lock is taken on a database other replicas
are writing to.

Wire that into whatever already runs on a schedule: pg_cron, a Kubernetes CronJob, or your
existing migration tooling. For the local `db/runa.db`, deleting the file is the honest
equivalent.

## Containers

`runa new` scaffolds a `Dockerfile` that serves the app:

```dockerfile
FROM python:3.14-slim

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
- [ ] `OPENAI_API_KEY` set too if any agent uses `memory`/`knowledge` (see
      [Where embeddings go](#where-embeddings-go))
- [ ] `RUNA_DATABASE_URL` set if more than one replica
- [ ] `/health` wired to the liveness probe
- [ ] `max_tokens` and `timeout` set on agents that face the public
- [ ] A retention job on a schedule (see [Retention](#retention))
- [ ] Log level at INFO, not DEBUG, unless you mean to record user content
