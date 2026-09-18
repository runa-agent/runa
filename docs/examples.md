# Examples

Runnable code for every primitive in [RUNA.md](https://github.com/Benybrahim/runa/blob/main/RUNA.md),
one folder per primitive under
[`examples/`](https://github.com/Benybrahim/runa/tree/main/examples), numbered to match. Every
script runs standalone with `uv run python examples/<folder>/<file>.py`. The same code also
appears inline on each concept page below, right next to the prose that explains it.

## Start Here

* **[Quickstart](https://github.com/Benybrahim/runa/tree/main/examples/00_quickstart)**.
  The smallest useful agent: one `Agent`, one `@tool`, one call.

## Core Concepts

* **[Agent](https://github.com/Benybrahim/runa/tree/main/examples/01_agent)**, see [Agents](agents.md).
* **[Tool](https://github.com/Benybrahim/runa/tree/main/examples/02_tool)**, see [Tools](tools.md).
* **[Guardrail](https://github.com/Benybrahim/runa/tree/main/examples/03_guardrail)**, see [Guardrails](guardrails.md).
* **[Subagent](https://github.com/Benybrahim/runa/tree/main/examples/05_subagent)**, see [Subagents](subagents.md).

## Running Agents

* **[Human Approval](https://github.com/Benybrahim/runa/tree/main/examples/04_approval)**, see [Human Approval](approval.md).
* **[Session](https://github.com/Benybrahim/runa/tree/main/examples/06_session)**, see [Sessions and Chat](sessions.md).
* **[Memory](https://github.com/Benybrahim/runa/tree/main/examples/07_memory)**, see [Memory](memory.md).
* **[Knowledge](https://github.com/Benybrahim/runa/tree/main/examples/08_knowledge)**, see [Knowledge](knowledge.md).
* **[MCP Server](https://github.com/Benybrahim/runa/tree/main/examples/09_mcp)**, see [MCP Servers](mcp.md).

## Testing and Evaluation

* **[Eval (Case/Dataset)](https://github.com/Benybrahim/runa/tree/main/examples/13_eval)**, see [Evaluation](evaluation.md).
* **[Test](https://github.com/Benybrahim/runa/tree/main/examples/12_test)**. A bare `test_*` function with a plain `assert`.

## Observability

* **[Hooks](https://github.com/Benybrahim/runa/tree/main/examples/11_hooks)**, see [Tracing and Hooks](tracing.md#hooks).
* **[Tracing](https://github.com/Benybrahim/runa/tree/main/examples/14_tracing)**, see [Tracing and Hooks](tracing.md).

## Reference

* **[Model](https://github.com/Benybrahim/runa/tree/main/examples/10_model)**, see [Model Providers](models.md).

## Applications

* **[Full apps](https://github.com/Benybrahim/runa/tree/main/examples/applications)**. A guided
  tour combining every primitive in one run, plus FastAPI deployments over SQLite and over
  Postgres/Redis.
