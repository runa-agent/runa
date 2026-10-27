# API Reference

Auto-generated from docstrings. For the concepts and conventions behind each primitive, see
[Core Concepts](agents.md) and [Running Agents](sessions.md); this page is the exhaustive
signature-level reference.

## Agent

::: runa.Agent

## Tool

::: runa.tool.tool

::: runa.tool.Tool

::: runa.tool.FunctionTool

## Guardrail

::: runa.guardrail.guardrail

::: runa.guardrail.Guardrail

::: runa.guardrail.Phase

::: runa.guardrail.GuardrailVerdict

::: runa.guardrail.GuardrailResult

::: runa.guardrail.GuardrailResults

## Human Approval

::: runa.approval.approval

## Session

::: runa.session.Session

## Memory

::: runa.Memory

::: runa.MemoryMatch

## Knowledge

::: runa.Knowledge

::: runa.KnowledgeMatch

## Cache

::: runa.Cache

::: runa.MemoryCache

## MCP Server

::: runa.MCPServer

## Running

::: runa.Run

::: runa.RunStatus

::: runa.RunStream

::: runa.ToolCall

::: runa.ModelSettings

::: runa.Model

::: runa.OpenAICompatibleModel

::: runa.Usage

::: runa.RunState

::: runa.Interruption

## Exceptions

::: runa.exceptions.RunaError

::: runa.exceptions.MaxTurnsExceeded

::: runa.exceptions.MaxTokensExceeded

::: runa.exceptions.RunTimeout

::: runa.exceptions.ModelBehaviorError

::: runa.exceptions.UserError

::: runa.exceptions.UnexpectedError

## Serving

::: runa.serve.create_app

::: runa.serve.resolve_api_key

::: runa.serve.RunRequest

::: runa.serve.Decision

## Hooks

::: runa.Hooks

::: runa.LoggingHooks

## Streaming

::: runa.StreamEvent

::: runa.TextDelta

::: runa.ToolCalled

::: runa.ToolOutput

::: runa.AgentChanged

## Evaluation

::: runa.Case

::: runa.CaseReport

::: runa.Dataset

::: runa.EvaluationResult

::: runa.Report

::: runa.EvalStatus

## Tracing

::: runa.tracing.observe

::: runa.Span

::: runa.tracing.SpanRow

::: runa.Trace

::: runa.TraceExporter

::: runa.ConsoleExporter

::: runa.StoreExporter

::: runa.tracing.TraceStore

## Storage

::: runa.db.at

::: runa.db.Place

::: runa.db.traces

::: runa.db.sessions

::: runa.db.evals

::: runa.db.session

::: runa.db.cache

::: runa.session.store.SessionStore

::: runa.eval.store.EvalStore

## Exceptions

::: runa.exceptions.RunaError

::: runa.exceptions.MaxTurnsExceeded

::: runa.exceptions.ModelBehaviorError

::: runa.exceptions.UserError

::: runa.exceptions.UnexpectedError

::: runa.exceptions.GuardrailTripwireTriggered

::: runa.exceptions.DuplicateToolCallError

::: runa.exceptions.RunAlreadyResumed
