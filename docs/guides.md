# Runa Guides

These guides assume you've read [Getting Started with Runa](getting_started.md). Each one
stands alone. Read whichever covers what you're building today.

## Start Here

* **[Getting Started with Runa](getting_started.md).**
  Install Runa, scaffold an app, and run your first agent.

## Core Concepts

* **[Agents](agents.md).**
  The `Agent` class: attributes, `instructions`, `run`/`run_sync`/`run_streamed`, history,
  usage, and context.

* **[Tools](tools.md).**
  Exposing functions to the model with `@tool`.

* **[Guardrails](guardrails.md).**
  Checking input and output with `@guardrail`.

* **[Subagents](subagents.md).**
  Composing agents with `.handoff` and `.delegate`.

## Running Agents

* **[Sessions and Chat](sessions.md).**
  Persisting conversation history with `session=`, and `runa chat`.

* **[Memory](memory.md).**
  Long-term, semantic memory that persists across conversations.

* **[Knowledge](knowledge.md).**
  Retrieving the application's own documents by meaning, before every turn.

* **[Cache](cache.md).**
  A minimal get/set/delete/clear cache, independent of Agent, Memory, and Sessions.

* **[MCP Servers](mcp.md).**
  Connecting to external tool servers over MCP.

* **[Human Approval](approval.md).**
  Gating sensitive tool calls behind human approval, and how that differs from a guardrail.

## Testing and Evaluation

* **[Evaluation](evaluation.md).**
  Behavioral evals with `runa eval`.

## Observability

* **[Tracing and Hooks](tracing.md).**
  What gets recorded for every run, `runa traces`, and lifecycle hooks.

## Reference

* **[Model Providers](models.md).**
  Which model string picks which provider, and where its API key lives.

* **[CLI Reference](cli.md).**
  Every `runa` command in one place.
