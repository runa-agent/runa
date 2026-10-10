"""Runa: an opinionated framework for agentic AI."""

from importlib.metadata import version

from runa import db
from runa._items import ConversationItem
from runa._types import ModelSettings, Reasoning, Usage
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
from runa.lifecycle import Hooks, LoggingHooks
from runa.mcp import MCPServer
from runa.memory import Memory, MemoryMatch
from runa.run import Run, RunStream
from runa.run_state import Interruption, RunState
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
    StoreExporter,
    Trace,
    TraceExporter,
    add_exporter,
    observe,
)

__version__ = version("runa-ai")

__all__ = [
    "DEFAULT_THRESHOLDS",
    "Agent",
    "AgentUpdatedStreamEvent",
    "Cache",
    "Case",
    "CaseReport",
    "ConsoleExporter",
    "ConversationItem",
    "Dataset",
    "EvaluationResult",
    "Hooks",
    "Interruption",
    "Knowledge",
    "KnowledgeMatch",
    "LoggingHooks",
    "MCPServer",
    "Memory",
    "MemoryCache",
    "MemoryMatch",
    "ModelSettings",
    "RawResponsesStreamEvent",
    "Reasoning",
    "Report",
    "Run",
    "RunStream",
    "RunItemStreamEvent",
    "RunState",
    "SQLiteCache",
    "SQLiteExporter",
    "SQLiteSession",
    "Span",
    "Status",
    "StoreExporter",
    "StreamEvent",
    "Trace",
    "Usage",
    "TraceExporter",
    "__version__",
    "add_exporter",
    "approval",
    "db",
    "guardrail",
    "observe",
    "tool",
]
