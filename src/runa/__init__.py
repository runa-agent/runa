"""Runa: an opinionated framework for agentic AI.

The public surface is one flat namespace -- `from runa import Agent, tool` -- but it is resolved
one name at a time. A package's `__init__` runs before any of its submodules, so importing the
whole surface here would make every subsystem's cost unavoidable for every entry point: `runa
generate`, a `@tool` file, a lambda cold start. The Anthropic SDK and the MCP client alone are
most of a four-second import, and neither is needed to scaffold a file or print a version.

That also keeps the layering readable. When `runa.memory` reaches for `runa.db` it names the
submodule, not this facade, so the import graph stays a graph; a facade that pulled every
subsystem would make "does `runa.tool` depend on `runa.eval`?" unanswerable, because everything
would already depend on everything through here.

So: `_EXPORTS` maps each public name to the module that defines it, and PEP 562's `__getattr__`
imports that module on first read, caching the result as a plain global. The names are declared
for real under `TYPE_CHECKING`, so type checkers and editors see the full namespace.

The three exceptions are `tool`, `guardrail` and `approval`, imported eagerly below. Each names both
a submodule and the decorator that submodule defines, and `__getattr__` only runs for an attribute
this module doesn't have: the moment anything imports `runa.tool`, the import system binds the
*module* here, and `from runa import tool` would hand back a module instead of a decorator.
Importing the decorator eagerly, after its module, wins that race. They cost no subsystem -- a
`@tool` file was importing `runa.tool` anyway.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from runa.approval import approval
from runa.guardrail import guardrail
from runa.tool import tool

if TYPE_CHECKING:
    from runa import db
    from runa._items import ConversationItem
    from runa._models import Model, OpenAICompatibleModel
    from runa._types import ModelSettings, RunContext, Usage
    from runa.agent import Agent
    from runa.cache import Cache, MemoryCache, SQLiteCache
    from runa.eval import (
        DEFAULT_THRESHOLDS,
        Case,
        CaseReport,
        Dataset,
        EvalStatus,
        EvaluationResult,
        Report,
    )
    from runa.knowledge import Knowledge, KnowledgeMatch
    from runa.lifecycle import Hooks, LoggingHooks
    from runa.mcp import MCPServer
    from runa.memory import Memory, MemoryMatch
    from runa.run import Run, RunStatus, RunStream
    from runa.run_state import Interruption, RunState
    from runa.session import SQLiteSession
    from runa.stream_events import (
        AgentUpdatedStreamEvent,
        RawResponsesStreamEvent,
        RunItemStreamEvent,
        StreamEvent,
    )
    from runa.tool import ToolCall
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

__version__: str

_EXPORTS = {
    "Agent": "runa.agent",
    "AgentUpdatedStreamEvent": "runa.stream_events",
    "Cache": "runa.cache",
    "Case": "runa.eval",
    "CaseReport": "runa.eval",
    "ConsoleExporter": "runa.tracing",
    "ConversationItem": "runa._items",
    "DEFAULT_THRESHOLDS": "runa.eval",
    "Dataset": "runa.eval",
    "EvalStatus": "runa.eval",
    "EvaluationResult": "runa.eval",
    "Hooks": "runa.lifecycle",
    "Interruption": "runa.run_state",
    "Knowledge": "runa.knowledge",
    "KnowledgeMatch": "runa.knowledge",
    "LoggingHooks": "runa.lifecycle",
    "MCPServer": "runa.mcp",
    "Memory": "runa.memory",
    "MemoryCache": "runa.cache",
    "MemoryMatch": "runa.memory",
    "Model": "runa._models",
    "ModelSettings": "runa._types",
    "OpenAICompatibleModel": "runa._models",
    "RawResponsesStreamEvent": "runa.stream_events",
    "Report": "runa.eval",
    "Run": "runa.run",
    "RunContext": "runa._types",
    "RunItemStreamEvent": "runa.stream_events",
    "RunState": "runa.run_state",
    "RunStatus": "runa.run",
    "RunStream": "runa.run",
    "SQLiteCache": "runa.cache",
    "SQLiteExporter": "runa.tracing",
    "SQLiteSession": "runa.session",
    "Span": "runa.tracing",
    "StoreExporter": "runa.tracing",
    "StreamEvent": "runa.stream_events",
    "Trace": "runa.tracing",
    "ToolCall": "runa.tool",
    "TraceExporter": "runa.tracing",
    "Usage": "runa._types",
    "add_exporter": "runa.tracing",
    "observe": "runa.tracing",
}
"""The lazily resolved public names, mapped to the module that defines each one.

Four names are missing. `db` is a submodule rather than a name inside one, and `__version__` is
read off the installed distribution rather than imported at all; `__getattr__` handles both
itself, along with `Status`, the deprecated spelling of `EvalStatus`. `tool`, `guardrail` and
`approval` are the eager three, for the reason this module's docstring gives -- any new public
name that collides with a submodule's belongs up there with them.
"""

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
    "EvalStatus",
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
    "Model",
    "ModelSettings",
    "OpenAICompatibleModel",
    "RawResponsesStreamEvent",
    "Report",
    "Run",
    "RunContext",
    "RunStream",
    "RunItemStreamEvent",
    "RunState",
    "RunStatus",
    "SQLiteCache",
    "SQLiteExporter",
    "SQLiteSession",
    "Span",
    "StoreExporter",
    "StreamEvent",
    "ToolCall",
    "Trace",
    "TraceExporter",
    "Usage",
    "__version__",
    "add_exporter",
    "approval",
    "db",
    "guardrail",
    "observe",
    "tool",
]


def __getattr__(name: str) -> object:
    """Resolve one public name, importing only the module that defines it.

    The result is written into this module's globals, so the lookup happens once per name and
    every read after it is an ordinary global. A name that isn't public raises `AttributeError`
    here rather than `ImportError` at `import runa` -- at the same line of calling code either
    way.
    """
    if name == "db":
        value = import_module("runa.db")
    elif name == "__version__":
        from importlib.metadata import version

        value = version("runa-ai")
    elif name == "Status":
        import warnings

        from runa.eval import EvalStatus

        warnings.warn(
            "runa.Status is deprecated, use runa.EvalStatus (Run.status is a RunStatus)",
            DeprecationWarning,
            stacklevel=2,
        )
        return EvalStatus
    else:
        try:
            module = _EXPORTS[name]
        except KeyError:
            raise AttributeError(f"module 'runa' has no attribute {name!r}") from None
        value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the public surface, which `__getattr__` keeps out of this module's globals."""
    return sorted(__all__)
