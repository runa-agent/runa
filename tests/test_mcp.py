"""Tests for `runa.mcp`: the bridge from an MCP server's tools to `FunctionTool`."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, cast

import pytest
from helpers import run as await_
from mcp_types import CallToolResult, ListToolsResult, TextContent, Tool

from runa import Agent, guardrail, mcp
from runa._types import ModelResponse, RunContext, Usage
from runa.exceptions import ToolError
from runa.guardrail import Phase
from runa.mcp import (
    MCPServer,
    MCPServerStdio,
    MCPServerStreamableHttp,
    _Connection,
    _MCPServerBase,
)


class _FakeSession:
    """A stand-in for `mcp.ClientSession`: scripted `list_tools`/`call_tool` results."""

    def __init__(self, tools: list[Tool], call_results: dict[str, CallToolResult]) -> None:
        self._tools = tools
        self._call_results = call_results
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> ListToolsResult:
        return ListToolsResult(tools=self._tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        self.calls.append((name, arguments))
        return self._call_results[name]


class _FakeServer(_MCPServerBase):
    """An `_MCPServerBase` opening onto a `_FakeSession` instead of a real transport.

    `_open` is what a transport would supply, so overriding it leaves everything this file is
    testing -- the per-loop caching in `connect`, the tool mapping, `close` -- as written.
    """

    def __init__(self, session: _FakeSession, **policy: Any) -> None:
        super().__init__(name="fake", address="fake://server", **policy)
        self._fake_session = session
        self.opens = 0
        self.closed = 0

    async def _open(self) -> _Connection:
        self.opens += 1
        listed = await self._fake_session.list_tools()
        stack = AsyncExitStack()
        await stack.enter_async_context(_count_close(self))
        return _Connection(
            stack=stack,
            session=cast(Any, self._fake_session),
            tools=[self._as_function_tool(tool) for tool in listed.tools],
        )


@asynccontextmanager
async def _count_close(server: _FakeServer) -> AsyncIterator[None]:
    """Stand in for a transport, recording that its connection was actually shut down."""
    try:
        yield
    finally:
        server.closed += 1


def test_list_tools_maps_mcp_tools_to_function_tools() -> None:
    """Each MCP `Tool` becomes a `FunctionTool`, named under its server and otherwise as listed."""
    tool = Tool(
        name="add",
        description="Add two numbers.",
        input_schema={"type": "object", "properties": {"a": {"type": "integer"}}},
    )
    server = _FakeServer(_FakeSession([tool], {}))

    tools = asyncio.run(server.list_tools())

    assert len(tools) == 1
    assert tools[0].name == "fake__add"
    assert tools[0].description == "Add two numbers."
    assert tools[0].params_json_schema == tool.input_schema


def test_list_tools_caches_after_the_first_call() -> None:
    """A second `list_tools()` call reuses the cached result instead of asking the server again."""
    session = _FakeSession([Tool(name="add", input_schema={"type": "object"})], {})
    server = _FakeServer(session)

    async def twice() -> tuple[list[Any], list[Any]]:
        return await server.list_tools(), await server.list_tools()

    first, second = asyncio.run(twice())

    assert first is second
    assert server.opens == 1


def test_a_second_event_loop_gets_its_own_connection() -> None:
    """A session cannot outlive the loop it was opened on, so a new loop opens a new one.

    `Agent.run_sync` is an `asyncio.run` per call, so this is the shape of every `runa chat`
    turn after the first: reusing turn 1's session here is a hang or "Event loop is closed".
    """
    session = _FakeSession([Tool(name="add", input_schema={"type": "object"})], {})
    server = _FakeServer(session)

    first = asyncio.run(server.list_tools())
    second = asyncio.run(server.list_tools())

    assert server.opens == 2
    assert first is not second


def test_close_shuts_the_connection_down_and_reconnects_on_demand() -> None:
    """`close()` closes the transport and forgets the tools; the next call opens a fresh one."""
    session = _FakeSession([Tool(name="add", input_schema={"type": "object"})], {})
    server = _FakeServer(session)

    async def use_close_and_use_again() -> None:
        await server.list_tools()
        await server.close()
        assert server.closed == 1
        await server.list_tools()

    asyncio.run(use_close_and_use_again())

    assert server.opens == 2


def test_close_is_a_no_op_when_this_loop_never_connected() -> None:
    """Closing a server that was never used on this loop does nothing, rather than failing."""
    server = _FakeServer(_FakeSession([], {}))

    asyncio.run(server.close())

    assert (server.opens, server.closed) == (0, 0)


def test_on_invoke_tool_extracts_text_from_a_successful_call() -> None:
    """A successful `call_tool` result's text content becomes the tool's return value."""
    tool = Tool(name="add", input_schema={"type": "object"})
    result = CallToolResult(content=[TextContent(type="text", text="3")])
    session = _FakeSession([tool], {"add": result})
    server = _FakeServer(session)

    (function_tool,) = asyncio.run(server.list_tools())

    async def call() -> Any:
        return await function_tool.on_invoke_tool(RunContext(), '{"a": 1, "b": 2}', "call_1")

    output = asyncio.run(call())

    assert output == "3"
    assert session.calls == [("add", {"a": 1, "b": 2})]


def test_on_invoke_tool_raises_a_tool_error_for_an_error_result() -> None:
    """A `CallToolResult` with `is_error=True` fails the call as a `ToolError`, not a string."""
    tool = Tool(name="fail", input_schema={"type": "object"})
    result = CallToolResult(content=[TextContent(type="text", text="boom")], is_error=True)
    session = _FakeSession([tool], {"fail": result})
    server = _FakeServer(session)

    (function_tool,) = asyncio.run(server.list_tools())

    async def call() -> Any:
        return await function_tool.on_invoke_tool(RunContext(), "{}", "call_1")

    with pytest.raises(ToolError, match="^boom$"):
        asyncio.run(call())


def test_on_invoke_tool_falls_back_to_structured_content() -> None:
    """A tool call with no text content returns `structured_content` instead."""
    tool = Tool(name="lookup", input_schema={"type": "object"})
    result = CallToolResult(content=[], structured_content={"id": 1})
    session = _FakeSession([tool], {"lookup": result})
    server = _FakeServer(session)

    (function_tool,) = asyncio.run(server.list_tools())

    async def call() -> Any:
        return await function_tool.on_invoke_tool(RunContext(), "{}", "call_1")

    output = asyncio.run(call())

    assert output == {"id": 1}


def test_mcp_server_stdio_builds_from_the_shared_config() -> None:
    """`MCPServer(name=...).stdio(...)` forwards shared kwargs into `MCPServerStdio`."""
    server = MCPServer(name="fs").stdio("npx", ["-y", "server-filesystem"])

    assert isinstance(server, MCPServerStdio)
    assert server.name == "fs"


def test_mcp_server_http_builds_from_the_shared_config() -> None:
    """`MCPServer(name=...).http(...)` carries the shared name onto the HTTP transport."""
    server = MCPServer(name="search").http("https://example.com/mcp")

    assert isinstance(server, MCPServerStreamableHttp)
    assert server.name == "search"


def test_stdio_passes_its_own_params_to_the_subprocess() -> None:
    """`env`/`cwd` are `stdio`'s own params, reaching the spawn parameters."""
    server = MCPServer().stdio("npx", ["serve"], env={"TOKEN": "x"}, cwd="/tmp")

    assert server._params.env == {"TOKEN": "x"}
    assert server._params.cwd == "/tmp"


def test_an_unknown_option_is_rejected_at_the_call_site() -> None:
    """A misspelled option fails where it was written, instead of being silently dropped."""
    with pytest.raises(TypeError, match="timeuot"):
        MCPServer(name="files", timeuot=30)  # pyright: ignore[reportCallIssue]

    with pytest.raises(TypeError, match="timeuot"):
        MCPServer(name="files").stdio("npx", timeuot=30)  # pyright: ignore[reportCallIssue]

    with pytest.raises(TypeError, match="timeuot"):
        MCPServer(name="search").http("https://example.com/mcp", timeuot=30)  # pyright: ignore[reportCallIssue]


@guardrail
def no_path_traversal(args: Any) -> bool:
    """Trip when a call's path escapes the directory the server was given."""
    return ".." in str(args.get("path", ""))


def test_a_servers_needs_approval_reaches_every_tool_it_lists() -> None:
    """A remote tool has no decorator to declare approval on, so the server declares it."""
    tools = [
        Tool(name="write_file", input_schema={"type": "object"}),
        Tool(name="read_file", input_schema={"type": "object"}),
    ]
    server = _FakeServer(_FakeSession(tools, {}), needs_approval=True)

    listed = asyncio.run(server.list_tools())

    assert [function_tool.needs_approval for function_tool in listed] == [True, True]


def test_a_servers_guardrails_are_bound_against_the_tool_call() -> None:
    """The server's predicates land on the tool phases, checking arguments rather than a turn."""
    tool = Tool(name="read_file", input_schema={"type": "object"})
    server = _FakeServer(_FakeSession([tool], {}), guardrails=[no_path_traversal.input])

    (function_tool,) = asyncio.run(server.list_tools())

    assert function_tool.guardrails[Phase.TOOL_OUTPUT] == []
    (entry,) = function_tool.guardrails[Phase.TOOL_INPUT]
    assert await_(entry.guardrail_function('{"path": "../etc/passwd"}')).tripped
    assert not await_(entry.guardrail_function('{"path": "notes.md"}')).tripped


def test_the_builder_carries_the_policy_onto_either_transport() -> None:
    """`needs_approval`/`guardrails` are shared config, like `name`, not a transport's own."""
    stdio = MCPServer(
        name="files", needs_approval=True, guardrails=[no_path_traversal.input]
    ).stdio("npx", ["serve"])
    http = MCPServer(name="search", needs_approval=True, guardrails=[no_path_traversal.input]).http(
        "https://example.com/mcp"
    )

    for server in (stdio, http):
        assert server._needs_approval is True
        assert len(server._guardrails[Phase.TOOL_INPUT]) == 1


def test_a_run_pauses_on_an_mcp_tool_that_needs_approval() -> None:
    """End to end: the approval gate the docs promise fires for an MCP tool, and resumes."""
    tool = Tool(name="write_file", input_schema={"type": "object"})
    written = CallToolResult(content=[TextContent(type="text", text="written")])
    session = _FakeSession([tool], {"write_file": written})
    server = _FakeServer(session, needs_approval=True)
    model: Any = _ScriptedModel(
        [_tool_call_message("fake__write_file", '{"path": "/tmp/x"}'), _final_message("done")]
    )
    agent = _FileAgent(model=model, mcp=[server])

    paused = agent.run_sync("write a file")

    assert paused.status == "paused"
    assert [item.name for item in paused.interruptions] == ["fake__write_file"]
    assert session.calls == []  # the gate held before anything reached the server

    state = paused.to_state()
    state.approve(state.pending[0])
    resumed = agent.run_sync(state)

    assert resumed.status == "completed"
    assert session.calls == [("write_file", {"path": "/tmp/x"})]


def test_a_servers_guardrail_trips_before_the_call_reaches_the_server() -> None:
    """End to end: a tripped tool input guardrail stops the call and ends the run as an error."""
    tool = Tool(name="read_file", input_schema={"type": "object"})
    session = _FakeSession([tool], {})
    server = _FakeServer(session, guardrails=[no_path_traversal.input])
    model: Any = _ScriptedModel(
        [_tool_call_message("fake__read_file", '{"path": "../etc/passwd"}')]
    )
    agent = _FileAgent(model=model, mcp=[server])

    result = agent.run_sync("read ../etc/passwd")

    assert result.status == "error"
    assert [entry.tripped for entry in result.tool_input_guardrail_results] == [True]
    assert session.calls == []


class _FileAgent(Agent):
    """A real `Agent`, so the gates under test are the ones the turn loop actually runs."""

    name = "file_agent"
    instructions = "Use the tools."


class _ScriptedModel:
    """A `Model` stand-in answering each turn with the next pre-scripted message."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = list(messages)

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            output=[self._messages.pop(0)],
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, requests=1),
        )


def _tool_call_message(name: str, arguments: str) -> dict[str, Any]:
    """One assistant message calling `name` with `arguments`."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


def _final_message(text: str) -> dict[str, Any]:
    """One assistant message that answers instead of calling a tool."""
    return {"role": "assistant", "content": text, "tool_calls": None}


def test_mcp_server_is_the_only_exported_name() -> None:
    """The builder is the whole public surface; the transports aren't a second call site."""
    assert mcp.__all__ == ["MCPServer"]
