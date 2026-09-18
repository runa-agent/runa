"""Runa: an opinionated framework for agentic AI."""

from importlib.metadata import version

from runa.agent import Agent
from runa.approval import approval
from runa.cache import Cache, MemoryCache, SQLiteCache
from runa.eval import (
    DEFAULT_THRESHOLDS,
    Case,
    CaseReport,
    Dataset,
    EvaluationResult,
    Report,
    Status,
)
from runa.guardrail import guardrail
from runa.knowledge import Knowledge, KnowledgeMatch
from runa.lifecycle import AgentHooks, LoggingAgentHooks, LoggingRunHooks, RunHooks
from runa.mcp import MCPServer, MCPServerStdio, MCPServerStreamableHttp
from runa.memory import Memory, MemoryMatch
from runa.result import RunResult, RunResultStreaming
from runa.run import Run
from runa.run_config import RunConfig
from runa.run_state import Interruption, RunState
from runa.runner import Runner
from runa.session import SQLiteSession
from runa.stream_events import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
    StreamEvent,
)
from runa.tool import tool
from runa.tracing import (
    ConsoleExporter,
    Span,
    SQLiteExporter,
    Trace,
    TraceExporter,
    add_exporter,
    observe,
)

__version__ = version("runa-ai")

__all__ = [
    "DEFAULT_THRESHOLDS",
    "Agent",
    "AgentHooks",
    "AgentUpdatedStreamEvent",
    "Cache",
    "Case",
    "CaseReport",
    "ConsoleExporter",
    "Dataset",
    "EvaluationResult",
    "Interruption",
    "Knowledge",
    "KnowledgeMatch",
    "LoggingAgentHooks",
    "LoggingRunHooks",
    "MCPServer",
    "MCPServerStdio",
    "MCPServerStreamableHttp",
    "Memory",
    "MemoryCache",
    "MemoryMatch",
    "RawResponsesStreamEvent",
    "Report",
    "Run",
    "RunConfig",
    "RunHooks",
    "RunItemStreamEvent",
    "RunResult",
    "RunResultStreaming",
    "RunState",
    "Runner",
    "SQLiteCache",
    "SQLiteExporter",
    "SQLiteSession",
    "Span",
    "Status",
    "StreamEvent",
    "Trace",
    "TraceExporter",
    "__version__",
    "add_exporter",
    "approval",
    "guardrail",
    "observe",
    "tool",
]
