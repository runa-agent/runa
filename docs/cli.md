# CLI Reference

Every subcommand runs from inside a Runa app (created with `runa new`), except `runa new` itself.

## `runa new [NAME]`

Scaffold a new application at `./NAME`, with the conventional layout. See
[Getting Started](getting_started.md). Omit `NAME` to scaffold the current directory in place
instead of creating a subdirectory.

## `runa generate KIND NAME`

Generate scaffolding inside an existing app:

```bash
runa generate agent MyAgent --model gpt-5.4-nano  # app/agents/my_agent.py, evals/my_agent.jsonl
runa generate tool MyTool                         # app/tools/my_tool.py
runa generate guardrail MyCheck                   # app/guardrails/my_check.py
runa generate prompt my_agent                     # app/prompts/my_agent.md
runa generate evaluation my_agent                 # evals/my_agent.jsonl, for a hand-written agent
```

`agent`'s `NAME` must be UpperCamelCase ending in `Agent`, for example `MyAgent`. That is the one
naming convention this command enforces. Both the generated file and the class's `name`
attribute are derived from it via snake_case (`MyAgent` becomes `app/agents/my_agent.py`, with
`name = "my_agent"`). There is no separate `--name` to pass. `--model` is required.
`--instructions`, `--tool`, `--guardrail`, `--memory`, `--knowledge`, and `--compact` stay
optional.

## `runa chat [AGENT_NAME]`

Chat with an agent, or inspect past sessions. See [Sessions and Chat](sessions.md).

```bash
runa chat support_agent                       # start (or resume) a chat
runa chat support_agent --continue            # resume the most recent session
runa chat support_agent --resume [SESSION_ID] # resume a chosen/given session
runa chat support_agent --session SESSION_ID  # pin an exact session id
runa chat --list                              # list every session
runa chat --show SESSION_ID                   # replay one session's history
```

## `runa test`

Run every `test_*` function under `tests/`.

## `runa eval [AGENT_NAME]`

Run every dataset under `evals/` against its agent, or, with `AGENT_NAME` (the Agent's declared
`name`, for example `support_agent`), just that one. See [Evaluation](evaluation.md).

```bash
runa eval                                     # grade every evals/ dataset
runa eval support_agent                       # grade one agent's dataset
runa eval --add TRACE_ID [--expected TEXT]    # add a traced run as a case
```

## `runa traces SUBCOMMAND`

Inspect this app's traces in `runa.db`. See [Tracing and Hooks](tracing.md).

```bash
runa traces list           # most recent traces
runa traces errors         # most recent traces that errored
runa traces show TRACE_ID  # one trace's full span tree
```

## `runa serve`

Serve this app's agents over HTTP. See [Deployment](deployment.md). Needs the `serve` extra
(`uv add "runa-ai[serve]"`), not installed by a plain `runa-ai` install.

```bash
runa serve                              # http://127.0.0.1:8000, bearer auth
runa serve --host 0.0.0.0 --port 8000   # reachable from outside a container
runa serve --no-auth                    # local use, no token required
runa serve --workers 4                  # uvicorn worker processes
```

Requires `RUNA_API_KEY` and refuses to start without it, unless `--no-auth` is passed. Routes:
`GET /health` (unauthenticated), `GET /agents`, `POST /agents/{name}/runs` and
`POST /agents/{name}/runs/stream`.

## `runa prune`

Delete traces, sessions and eval runs this app has outgrown. Everything Runa persists is
append-only, so without this the database grows without bound. See
[Deployment](deployment.md#retention).

```bash
runa prune                                # older than 30 days (the default)
runa prune --older-than 90                # older than 90 days
runa prune --older-than 30 --dry-run      # report what would go, delete nothing
runa prune --older-than 30 --only traces  # one kind; repeatable
```

Prunes the shared Postgres when `RUNA_POSTGRES_DSN` is set, the local `db/runa.db` otherwise.

## `runa ui`

Serve a local, read-only dashboard over `runa.db`: Agents, Sessions, Traces, and Evaluations.
Needs the `ui` extra (`uv add "runa-ai[ui]"`), not installed by a plain `runa` install.

```bash
runa ui                    # http://127.0.0.1:8765
runa ui --host 0.0.0.0 --port 3000
```

## Exit Codes

`runa eval` and `runa test` exit `1` if any case or test failed, `0` otherwise, safe to wire into
CI. Everything else exits `1` only on an operator error (a mistyped id, running outside a Runa
app), printing a clean message instead of a traceback.
