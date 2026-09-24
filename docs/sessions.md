# Sessions and Chat

## In-Memory History

By default, an `Agent` instance remembers its own conversation:

```python
agent = SupportAgent()
agent.run_sync("My order hasn't arrived.")
agent.run_sync("It's order #4821.")  # remembers the first message
```

`agent.history` holds this. It only lives as long as the instance does.

## Persisting with `SQLiteSession`

Pass a `session` to persist history to `runa.db` instead, keyed by a `session_id` you choose:

```python
from runa import SQLiteSession

session = SQLiteSession("user-42")

agent.run_sync("My order hasn't arrived.", session=session)
agent.run_sync("It's order #4821.", session=session)
```

With a `session`, prior turns are read back from `runa.db` automatically. You only ever pass the
new message, and `agent.history` is left untouched. Reuse the same `session_id` (a user id, a
ticket id) to resume a conversation from anywhere, including a later process. Don't mix the two:
pick session-backed or in-memory per agent instance, not both for the same conversation.

`SQLiteSession` also supports:

```python
await session.get_items()  # this session's items, oldest first
await session.add_items(items)  # append items to this session's history
await session.set_items(items)  # replace this session's whole history
await session.pop_item()  # remove and return the most recent item
await session.clear_session()  # delete the session and all its items
```

Pass `user_id="user-42"` to `SQLiteSession` to also scope that agent's automatic
[Memory](memory.md) to this user. `session.user_id` is unrelated to conversation history itself;
`run` reads it only to know which user's memories to retrieve and store.

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

`SQLiteSession` is the only session implementation Runa ships for a single local process. A
custom store subclasses `SessionABC`'s four abstract methods (`get_items`, `add_items`,
`pop_item`, `clear_session`), nothing less.

For a deployment where multiple processes need to share one store, use `PostgresSession` (the
`runa-ai[postgres]` extra):

```python
from runa.db.postgres import PostgresSession

session = PostgresSession("user-42", dsn="postgresql://runa:runa@localhost:5432/runa")
```

It implements the same interface, backed by Postgres instead of SQLite.

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

Inspect what's stored in `runa.db` without starting a chat:

```bash
runa chat --list             # every session
runa chat --show SESSION_ID  # replay one session's full history
```

## Example

```python
--8<--"examples/06_session/sqlite_session.py"
```

```python
--8<--"examples/06_session/custom_session_store.py"
```

More in [`examples/06_session/`](https://github.com/Benybrahim/runa/tree/main/examples/06_session).
