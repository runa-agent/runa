# Sessions and Chat

## In-Memory History

By default, an `Agent` instance remembers its own conversation:

```python
agent = SupportAgent()
agent.run_sync("My order hasn't arrived.")
agent.run_sync("It's order #4821.")  # remembers the first message
```

`agent.history` holds this. It only lives as long as the instance does.

## Persisting with a `session`

Pass a `session` to persist history to `runa.db` instead, keyed by a `session_id` you choose:

```python
agent.run_sync("My order hasn't arrived.", session="user-42")
agent.run_sync("It's order #4821.", session="user-42")
```

With a `session`, prior turns are read back from `runa.db` automatically. You only ever pass the
new message, and `agent.history` is left untouched. Reuse the same `session_id` (a user id, a
ticket id) to resume a conversation from anywhere, including a later process. Don't mix the two:
pick session-backed or in-memory per agent instance, not both for the same conversation.

The id is all your application names. Which backend holds it is
[`RUNA_DATABASE_URL`](deployment.md)'s answer, resolved through `db.session(...)`, so the two
calls above persist to a local file on a laptop and to a shared Postgres in production with no
edit here.

## Working with the Session Object

For anything beyond the id, ask `runa.db` for the session itself and pass that:

```python
from runa import db

session = db.session("user-42", user_id="user-42")
agent.run_sync("My order hasn't arrived.", session=session)

await session.get_items()  # this session's items, oldest first
await session.add_items(items)  # append items to this session's history
await session.set_items(items)  # replace this session's whole history
await session.pop_item()  # remove and return the most recent item
await session.clear_session()  # delete the session and all its items
```

`user_id` scopes that agent's automatic [Memory](memory.md) to this user. It is unrelated to
conversation history itself; `run` reads it only to know which user's memories to retrieve and
store.

## Loading an Existing Transcript

`message` is one turn: a string, or a list of text and image parts. It does not take a list of
past messages, and passing one raises `TypeError`. To start from an earlier conversation, seed it
through the seam the mode already uses, before the first run:

```python
transcript = [
    {"role": "user", "content": "My order hasn't arrived."},
    {"role": "assistant", "content": "What is the order number?"},
]

agent.history = transcript  # in-memory
asyncio.run(session.add_items(transcript))  # session-backed
```

`add_items` appends, so seeding a session that already holds turns concatenates two
conversations; `set_items` replaces its history instead. Both are `async`, hence the
`asyncio.run` in synchronous code: inside an `async def`, `await` them directly.

## Writing Your Own Backend

Runa ships one implementation per backend `RUNA_DATABASE_URL` understands, and `db.session`
picks between them. A custom store subclasses `Session`'s four abstract methods (`get_items`,
`add_items`, `pop_item`, `clear_session`), nothing less.

For a deployment where multiple processes share one store, set `RUNA_DATABASE_URL` and change
nothing else. `runa serve`, `runa chat` and `runa ui` all resolve their session store through
`runa.db`, so they follow it automatically:

```bash
uv add "runa-ai[postgres]"
export RUNA_DATABASE_URL=postgresql://runa:runa@localhost:5432/runa
```

Constructing one by hand is the escape hatch for a store that is not this deployment's own:

```python
from runa import SQLiteSession
from runa.session.postgres import PostgresSession

SQLiteSession("user-42", "other/runa.db")  # one specific file
PostgresSession("user-42", "postgresql://runa:runa@localhost:5432/runa")  # another database
```

Both implement the same interface, so either can be passed as `session=`. Naming one is the
exception, not the default: it opts that conversation out of `RUNA_DATABASE_URL`, which is why
`session="user-42"` is the shape to reach for first.

## `runa chat`

`runa chat` is an interactive REPL that talks to any agent under `app/agents/`, by its declared
`name`:

```bash
runa chat support_agent
```

Every chat starts a fresh session by default, so repeated runs don't pile onto the same
conversation. Pick one back up with:

```bash
runa chat support_agent --continue           # most recent session
runa chat support_agent --resume             # choose from past sessions
runa chat support_agent --resume SESSION_ID  # resume a specific one
runa chat support_agent --session SESSION_ID # pin an exact id
```

Inspect what's stored without starting a chat:

```bash
runa chat --list             # every session
runa chat --show SESSION_ID  # replay one session's full history
```

Or from code. `runa.db.sessions()` hands back this deployment's `SessionStore`, pointed at
whichever backend it has, which is the same one those two commands read:

```python
from runa import db

store = db.sessions()

store.listing()                      # every session, most recently updated first
store.listing(agent="support_agent") # one agent's past sessions
store.messages("support_agent-1")    # that session's transcript, oldest first
```

`listing` returns `SessionSummary(id, updated_at)` and `messages` returns
`SessionMessage(created_at, role, text)`, identically on every backend. `messages` raises
`SessionNotFound` for a session id with no history, which is a different answer from a session
with nothing in it.

## Example

```python
--8<--"examples/06_session/sqlite_session.py"
```

```python
--8<--"examples/06_session/custom_session_store.py"
```

More in [`examples/06_session/`](https://github.com/Benybrahim/runa/tree/main/examples/06_session).
