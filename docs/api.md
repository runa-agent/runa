# API Reference

Auto-generated from docstrings. For the concepts and conventions behind each primitive, see
[Core Concepts](agents.md) and [Running Agents](sessions.md); this page is the exhaustive
signature-level reference.

## Agent

::: runa.Agent

::: runa.lifecycle.AgentHooks

::: runa.lifecycle.LoggingAgentHooks

## Tool

::: runa.tool.tool

::: runa.tool.FunctionTool

## Guardrail

::: runa.guardrail.guardrail

::: runa.guardrail.Guardrail

::: runa.guardrail.GuardrailFunctionOutput

::: runa.guardrail.GuardrailResult

::: runa.guardrail.ToolGuardrailFunctionOutput

## Human Approval

::: runa.approval.approval

## Session

::: runa.SQLiteSession

::: runa.session.SessionABC

## Memory

::: runa.Memory

::: runa.MemoryMatch

## Knowledge

::: runa.Knowledge

::: runa.KnowledgeMatch

## Cache

::: runa.Cache

::: runa.MemoryCache

::: runa.SQLiteCache

## MCP Server

::: runa.MCPServer

## Running

::: runa.Run

::: runa.RunStream

::: runa.ModelSettings

::: runa.Reasoning

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

## Serving

::: runa.serve.create_app

::: runa.serve.RunRequest

## Retention

::: runa.db.prune.prune

::: runa.db.prune.Pruned

## Hooks

::: runa.RunHooks

::: runa.LoggingRunHooks

## Streaming

::: runa.StreamEvent

::: runa.AgentUpdatedStreamEvent

::: runa.RawResponsesStreamEvent

::: runa.RunItemStreamEvent

## Evaluation

::: runa.Case

::: runa.CaseReport

::: runa.Dataset

::: runa.EvaluationResult

::: runa.Report

::: runa.Status

## Tracing

::: runa.tracing.observe

::: runa.Span

::: runa.Trace

::: runa.TraceExporter

::: runa.ConsoleExporter

::: runa.SQLiteExporter

## Exceptions

::: runa.exceptions.RunaError

::: runa.exceptions.RunErrorDetails

::: runa.exceptions.MaxTurnsExceeded

::: runa.exceptions.ModelBehaviorError

::: runa.exceptions.UserError

::: runa.exceptions.InputGuardrailTripwireTriggered

::: runa.exceptions.OutputGuardrailTripwireTriggered

::: runa.exceptions.ToolInputGuardrailTripwireTriggered

::: runa.exceptions.ToolOutputGuardrailTripwireTriggered

::: runa.exceptions.DuplicateToolCallError
