"""Tests for the handoff / delegate / auto subagent wiring, and `Agent.run`/`run_sync`."""

import asyncio
import importlib.util
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from runa import Agent
from runa._types import ModelResponse, RunContextWrapper, Usage
from runa.agent import Subagent
from runa.exceptions import MaxTurnsExceeded, RunErrorDetails
from runa.knowledge import Knowledge
from runa.lifecycle import LoggingRunHooks
from runa.memory import Memory
from runa.tool import FunctionTool, tool


def _handoff_names(agent: Agent) -> list[str]:
    """Names of an agent's handoffs, narrowed away from the raw `Handoff` union member."""
    return [h.name for h in agent.handoffs if isinstance(h, Agent)]


def _tool_names(agent: Agent) -> list[str]:
    """Names of an agent's tools, narrowed away from the raw `Tool` union members."""
    return [t.name for t in agent.tools if isinstance(t, FunctionTool)]


class Researcher(Agent):
    """A subagent used across tests."""

    name = "Researcher"
    instructions = "You research topics."


class Translator(Agent):
    """Another subagent used across tests."""

    name = "Translator"
    instructions = "You translate text."


def _import_module_from_file(module_name: str, path: Path) -> ModuleType:
    """Import `path` as `module_name`, registered in `sys.modules` like a real package import.

    `inspect.getfile` (which `Agent`'s prompt auto-load relies on) needs the module registered
    there to resolve its source file; a bare `module_from_spec` without this leaves it unable to.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_agent_cannot_be_instantiated_directly() -> None:
    """`Agent` itself must be subclassed; it isn't a usable agent on its own."""
    with pytest.raises(TypeError):
        Agent(name="Bare")


def test_instructions_auto_load_from_a_sibling_prompts_file(tmp_path: Path) -> None:
    """Omitting `instructions` loads it from `app/prompts/<name>.md` next to the agent's module."""
    agents_dir = tmp_path / "app" / "agents"
    prompts_dir = tmp_path / "app" / "prompts"
    agents_dir.mkdir(parents=True)
    prompts_dir.mkdir(parents=True)
    (agents_dir / "greeter_agent.py").write_text(
        "from runa import Agent\n\n\nclass GreeterAgent(Agent):\n    name = 'greeter_agent'\n"
    )
    (prompts_dir / "greeter_agent.md").write_text("You greet warmly.\n")

    module = _import_module_from_file("greeter_agent", agents_dir / "greeter_agent.py")

    assert module.GreeterAgent().instructions == "You greet warmly."


def test_instructions_stay_empty_outside_an_agents_directory() -> None:
    """A class not defined under an `agents/` directory never gets a prompt file auto-created."""

    class NoPrompt(Agent):
        name = "no_prompt_agent"

    assert NoPrompt().instructions is None


def test_missing_prompt_file_is_created_from_the_template(tmp_path: Path) -> None:
    """No `instructions`, no prompt file: one is created from the `runa generate prompt` stub."""
    agents_dir = tmp_path / "app" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "greeter_agent.py").write_text(
        "from runa import Agent\n\n\nclass GreeterAgent(Agent):\n    name = 'greeter_agent'\n"
    )

    module = _import_module_from_file("greeter_agent_autocreate", agents_dir / "greeter_agent.py")
    agent = module.GreeterAgent()

    prompt_file = tmp_path / "app" / "prompts" / "greeter_agent.md"
    assert prompt_file.is_file()
    assert agent.instructions == prompt_file.read_text().strip()
    assert "TODO: write the prompt greeter_agent uses." in agent.instructions


def test_explicit_instructions_skip_the_prompt_file(tmp_path: Path) -> None:
    """An explicit `instructions` attribute wins even when a matching prompt file exists."""
    agents_dir = tmp_path / "app" / "agents"
    prompts_dir = tmp_path / "app" / "prompts"
    agents_dir.mkdir(parents=True)
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "greeter_agent.md").write_text("From the file.\n")
    (agents_dir / "greeter_agent.py").write_text(
        "from runa import Agent\n\n\n"
        "class GreeterAgent(Agent):\n"
        "    name = 'greeter_agent'\n"
        "    instructions = 'From the class.'\n"
    )

    module = _import_module_from_file("greeter_agent_explicit", agents_dir / "greeter_agent.py")

    assert module.GreeterAgent().instructions == "From the class."


def test_handoff_adds_only_to_handoffs() -> None:
    """`.handoff` wires the subagent as a handoff, not a tool."""

    class Main(Agent):
        name = "Main"
        instructions = "main"
        subagents = [Researcher.handoff]

    agent = Main()

    assert _handoff_names(agent) == ["Researcher"]
    assert agent.tools == []


def test_delegate_adds_only_to_tools() -> None:
    """`.delegate` wires the subagent as a tool, not a handoff."""

    class Main(Agent):
        name = "Main"
        instructions = "main"
        subagents = [Researcher.delegate]

    agent = Main()

    assert agent.handoffs == []
    assert _tool_names(agent) == [Researcher().as_tool(None, None).name]


def test_h_and_d_are_shorthand_for_handoff_and_delegate() -> None:
    """`.h`/`.d` bind exactly like `.handoff`/`.delegate`, same subagent, same mode."""

    class Main(Agent):
        name = "Main"
        instructions = "main"
        subagents = [Researcher.h, Translator.d]

    agent = Main()

    assert _handoff_names(agent) == ["Researcher"]
    assert _tool_names(agent) == [Translator().as_tool(None, None).name]


def test_delegate_tool_name_and_description_override() -> None:
    """Calling a `.delegate` subagent overrides the generated tool's name/description."""

    class Main(Agent):
        name = "Main"
        instructions = "main"
        subagents = [Researcher.delegate(tool_name="do_research", tool_description="Look into it.")]

    agent = Main()

    (tool_obj,) = agent.tools
    assert isinstance(tool_obj, FunctionTool)
    assert tool_obj.name == "do_research"
    assert tool_obj.description == "Look into it."


def test_bare_subagent_wires_both_handoff_and_delegate() -> None:
    """A subagent listed without `.handoff`/`.delegate` lets the model pick either mode."""

    class Main(Agent):
        name = "Main"
        instructions = "main"
        subagents = [Researcher]

    agent = Main()

    assert _handoff_names(agent) == ["Researcher"]
    assert _tool_names(agent) == [Researcher().as_tool(None, None).name]


def test_mixed_modes_wire_independently() -> None:
    """Handoff and delegate subagents in the same list don't interfere with each other."""

    class Main(Agent):
        name = "Main"
        instructions = "main"
        subagents = [Researcher.handoff, Translator.delegate]

    agent = Main()

    assert _handoff_names(agent) == ["Researcher"]
    assert _tool_names(agent) == [Translator().as_tool(None, None).name]


def test_no_subagents_leaves_handoffs_and_tools_empty() -> None:
    """An agent with no `subagents` attribute wires up cleanly."""

    class Main(Agent):
        name = "Main"
        instructions = "main"

    agent = Main()

    assert agent.handoffs == []
    assert agent.tools == []


def test_dict_subagents_wire_by_key() -> None:
    """A `{"handoff": [...], "delegate": [...], "auto": [...]}` dict wires each bucket."""

    class Helper(Agent):
        name = "Helper"
        instructions = "helper"

    class Main(Agent):
        name = "Main"
        instructions = "main"
        subagents = {
            "handoff": [Researcher],
            "delegate": [Translator],
            "auto": [Helper],
        }

    agent = Main()

    assert sorted(_handoff_names(agent)) == ["Helper", "Researcher"]
    assert sorted(_tool_names(agent)) == sorted(
        [Translator().as_tool(None, None).name, Helper().as_tool(None, None).name]
    )


def test_dict_subagents_delegate_bucket_keeps_tool_overrides() -> None:
    """A `.delegate(...)` override still applies inside the dict format's `delegate` bucket."""

    class Main(Agent):
        name = "Main"
        instructions = "main"
        subagents = {
            "delegate": [
                Researcher.delegate(tool_name="do_research", tool_description="Look into it.")
            ]
        }

    agent = Main()

    (tool_obj,) = agent.tools
    assert isinstance(tool_obj, FunctionTool)
    assert tool_obj.name == "do_research"
    assert tool_obj.description == "Look into it."


def test_subagent_descriptor_returns_fresh_immutable_instance() -> None:
    """Each access to `.handoff`/`.delegate` is a new `Subagent`; overrides don't mutate it."""
    first = Researcher.handoff
    second = Researcher.handoff

    assert first is not second
    assert first == second
    assert isinstance(first, Subagent)

    overridden = first(tool_name="custom", tool_description="d")
    assert overridden.tool_name == "custom"
    assert first.tool_name is None, "overriding a copy must not mutate the original"


def test_class_attributes_seed_init_defaults() -> None:
    """Class attributes like `name`/`instructions`/`model` become constructor defaults."""
    agent = Researcher()

    assert agent.name == "Researcher"
    assert agent.instructions == "You research topics."
    assert agent.model == "gpt-5.4-nano"


def test_explicit_kwarg_overrides_class_attribute() -> None:
    """An explicit constructor kwarg wins over the class attribute default."""
    agent = Researcher(model="gpt-4.1")

    assert agent.model == "gpt-4.1"


def test_memory_defaults_to_none() -> None:
    """Without `memory=`, an agent behaves exactly as if `runa.memory` didn't exist."""
    agent = Researcher()

    assert agent.memory is None
    assert "search_memory" not in _tool_names(agent)


def test_memory_auto_gives_a_default_memory_instance_and_no_tool() -> None:
    """`memory="auto"` builds a default `Memory` for the run lifecycle to use, with no tool.

    Automatic retrieval/persistence is the run lifecycle's job (see `test_runner.py`), not
    something `Agent.__init__` does.
    """

    class AutoMemory(Agent):
        name = "AutoMemory"
        instructions = "auto memory"
        memory = "auto"

    agent = AutoMemory()

    assert isinstance(agent.memory, Memory)
    assert "search_memory" not in _tool_names(agent)


def test_memory_llm_adds_a_tool_and_leaves_self_memory_none() -> None:
    """`memory="llm"` gives the model a `search_memory` tool instead of automatic retrieval."""

    class LLMMemory(Agent):
        name = "LLMMemory"
        instructions = "llm memory"
        memory = "llm"

    agent = LLMMemory()

    assert agent.memory is None
    assert "search_memory" in _tool_names(agent)


def test_memory_accepts_a_custom_instance_for_auto_mode() -> None:
    """A `Memory(...)` class attribute is used as-is, same as `"auto"` but with that instance."""
    custom = Memory(dimensions=4)

    class WithMemory(Agent):
        name = "WithMemory"
        instructions = "has memory"
        memory = custom

    agent = WithMemory()

    assert agent.memory is custom
    assert "search_memory" not in _tool_names(agent)


def test_memory_accepts_a_wholesale_custom_object_not_just_a_memory_instance() -> None:
    """A non-`Memory` object shaped like `MemoryLike` is accepted as-is: the escape hatch."""

    class CustomMemory:
        async def search(self, query: str, *, user_id: str | None = None, k: int = 5) -> list[Any]:
            return []

        async def remember_from_conversation(
            self, conversation: str, *, user_id: str | None, model: Any
        ) -> list[str]:
            return []

    custom = CustomMemory()

    class WithCustomMemory(Agent):
        name = "WithCustomMemory"
        instructions = "has custom memory"
        memory = custom

    agent = WithCustomMemory()

    assert agent.memory is custom
    assert "search_memory" not in _tool_names(agent)


def test_memory_kwarg_also_works_as_a_constructor_argument() -> None:
    """`Agent(..., memory="auto")` works the same as setting it as a class attribute."""

    class WithMemory(Agent):
        name = "WithMemory"
        instructions = "has memory"

    agent = WithMemory(memory="auto")

    assert isinstance(agent.memory, Memory)


def test_memory_rejects_an_unknown_string() -> None:
    """A `memory=` string other than `"auto"`/`"llm"` fails clearly instead of misbehaving."""
    from runa.exceptions import UserError

    class BadMemory(Agent):
        name = "BadMemory"
        instructions = "bad memory"
        memory = "sometimes"

    with pytest.raises(UserError, match="memory"):
        BadMemory()


def test_knowledge_defaults_to_none() -> None:
    """Without `knowledge=`, an agent behaves exactly as if `runa.knowledge` didn't exist."""
    agent = Researcher()

    assert agent.knowledge is None
    assert "search_knowledge" not in _tool_names(agent)


def test_knowledge_auto_gives_a_default_knowledge_instance_and_no_tool() -> None:
    """`knowledge="auto"` builds a default `Knowledge` for the run lifecycle to use, with no tool.

    Automatic retrieval is the run lifecycle's job (see `test_runner.py`), not something
    `Agent.__init__` does.
    """

    class AutoKnowledge(Agent):
        name = "AutoKnowledge"
        instructions = "auto knowledge"
        knowledge = "auto"

    agent = AutoKnowledge()

    assert isinstance(agent.knowledge, Knowledge)
    assert "search_knowledge" not in _tool_names(agent)


def test_knowledge_llm_adds_a_tool_and_leaves_self_knowledge_none() -> None:
    """`knowledge="llm"` gives the model a `search_knowledge` tool instead of automatic retrieval.

    Automatic retrieval is the run lifecycle's job, not something `Agent.__init__` does.
    """

    class LLMKnowledge(Agent):
        name = "LLMKnowledge"
        instructions = "llm knowledge"
        knowledge = "llm"

    agent = LLMKnowledge()

    assert agent.knowledge is None
    assert "search_knowledge" in _tool_names(agent)


def test_knowledge_accepts_a_custom_instance_for_auto_mode() -> None:
    """A `Knowledge(...)` class attribute is used as-is, same as `"auto"` but with that instance."""
    custom = Knowledge(dimensions=4)

    class WithKnowledge(Agent):
        name = "WithKnowledge"
        instructions = "has knowledge"
        knowledge = custom

    agent = WithKnowledge()

    assert agent.knowledge is custom
    assert "search_knowledge" not in _tool_names(agent)


def test_knowledge_accepts_a_wholesale_custom_object_not_just_a_knowledge_instance() -> None:
    """A non-`Knowledge` object shaped like `KnowledgeLike` is accepted as-is: the escape hatch."""

    class CustomKnowledge:
        async def search(self, query: str, *, k: int = 5) -> list[Any]:
            return []

    custom = CustomKnowledge()

    class WithCustomKnowledge(Agent):
        name = "WithCustomKnowledge"
        instructions = "has custom knowledge"
        knowledge = custom

    agent = WithCustomKnowledge()

    assert agent.knowledge is custom
    assert "search_knowledge" not in _tool_names(agent)


def test_knowledge_kwarg_also_works_as_a_constructor_argument() -> None:
    """`Agent(..., knowledge="auto")` works the same as setting it as a class attribute."""

    class WithKnowledge(Agent):
        name = "WithKnowledge"
        instructions = "has knowledge"

    agent = WithKnowledge(knowledge="auto")

    assert isinstance(agent.knowledge, Knowledge)


def test_knowledge_rejects_an_unknown_string() -> None:
    """A `knowledge=` string other than `"auto"`/`"llm"` fails clearly instead of misbehaving."""
    from runa.exceptions import UserError

    class BadKnowledge(Agent):
        name = "BadKnowledge"
        instructions = "bad knowledge"
        knowledge = "sometimes"

    with pytest.raises(UserError, match="knowledge"):
        BadKnowledge()


def test_history_starts_empty() -> None:
    """A freshly constructed agent has no conversation history yet."""
    agent = Researcher()

    assert agent.history == []


def test_usage_starts_empty() -> None:
    """A freshly constructed agent has no accumulated or last usage yet."""
    agent = Researcher()

    assert agent.usage == Usage()
    assert agent.last_usage == Usage()


@dataclass
class _Ctx:
    """A minimal run context used by the dynamic-instructions tests below."""

    label: str


def _single_arg_instructions(context: _Ctx) -> str:
    return f"context={context.label}"


def _two_arg_instructions(context: RunContextWrapper[_Ctx], agent: Any) -> str:
    return f"{agent.name}:{context.context.label}"


def test_single_arg_instructions_resolves_from_run_context() -> None:
    """A one-parameter `(context) -> str` `instructions` is adapted to the runner's 2-arg shape."""
    from runa.run_internal.agent_runner_helpers import _resolve_instructions

    class Dynamic(Agent):
        name = "Dynamic"
        instructions = _single_arg_instructions  # pyright: ignore[reportAssignmentType]

    agent = Dynamic()

    prompt = asyncio.run(_resolve_instructions(agent, RunContextWrapper(context=_Ctx(label="hi"))))

    assert prompt == "context=hi"


def test_two_arg_instructions_still_supported() -> None:
    """A native runner-style `(context, agent) -> str` `instructions` passes through unadapted."""
    from runa.run_internal.agent_runner_helpers import _resolve_instructions

    class Dynamic(Agent):
        name = "Dynamic"
        instructions = _two_arg_instructions

    agent = Dynamic()

    prompt = asyncio.run(_resolve_instructions(agent, RunContextWrapper(context=_Ctx(label="hi"))))

    assert prompt == "Dynamic:hi"


def test_string_instructions_pass_through_unchanged() -> None:
    """Plain string instructions are unaffected by the dynamic-instructions adapter."""
    assert Researcher().instructions == "You research topics."


class _FakeResult:
    """A stand-in for `RunResult`, just enough for `Agent.run`/`run_sync` to consume."""

    final_output = "ok"
    context_wrapper = RunContextWrapper(context=None, usage=Usage(input_tokens=1, output_tokens=2))
    interruptions: list[Any] = []
    trace = None

    def to_input_list(self) -> list[Any]:
        return []


def test_run_sync_defaults_to_logging_run_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_sync` passes a `LoggingRunHooks` when none is given."""
    captured: dict[str, Any] = {}

    def fake_run_sync(*args: Any, hooks: Any, **kwargs: Any) -> _FakeResult:
        captured["hooks"] = hooks
        return _FakeResult()

    monkeypatch.setattr("runa.agent.Runner.run_sync", staticmethod(fake_run_sync))

    Researcher().run_sync("hi")

    assert isinstance(captured["hooks"], LoggingRunHooks)


def test_run_sync_explicit_hooks_override_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit `hooks` argument is used instead of the default combined hooks."""
    captured: dict[str, Any] = {}
    custom_hooks = LoggingRunHooks()

    def fake_run_sync(*args: Any, hooks: Any, **kwargs: Any) -> _FakeResult:
        captured["hooks"] = hooks
        return _FakeResult()

    monkeypatch.setattr("runa.agent.Runner.run_sync", staticmethod(fake_run_sync))

    Researcher().run_sync("hi", hooks=custom_hooks)

    assert captured["hooks"] is custom_hooks


def test_run_sync_records_and_accumulates_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_sync` records the call's usage to `last_usage` and adds it to `usage`."""

    def fake_run_sync(*args: Any, **kwargs: Any) -> _FakeResult:
        return _FakeResult()

    monkeypatch.setattr("runa.agent.Runner.run_sync", staticmethod(fake_run_sync))

    agent = Researcher()
    agent.run_sync("hi")
    agent.run_sync("again")

    expected_call_usage = Usage(input_tokens=1, output_tokens=2)
    assert agent.last_usage == expected_call_usage
    assert agent.usage.input_tokens == 2
    assert agent.usage.output_tokens == 4


def test_run_sync_returns_a_completed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_sync` returns a `Run` with the final output, status, and this call's usage."""

    def fake_run_sync(*args: Any, **kwargs: Any) -> _FakeResult:
        return _FakeResult()

    monkeypatch.setattr("runa.agent.Runner.run_sync", staticmethod(fake_run_sync))

    run = Researcher().run_sync("hi")

    assert run.output == "ok"
    assert run.status == "completed"
    assert run.error is None
    assert run.usage == Usage(input_tokens=1, output_tokens=2)


def test_run_sync_passes_multimodal_message_content_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `runa.content` parts list is sent as the new user message's `content`, unchanged."""
    from runa import content

    captured: dict[str, Any] = {}

    def fake_run_sync(agent: Any, turn_input: Any, **kwargs: Any) -> _FakeResult:
        captured["turn_input"] = turn_input
        return _FakeResult()

    monkeypatch.setattr("runa.agent.Runner.run_sync", staticmethod(fake_run_sync))

    parts = [content.text("what's in this image?"), content.image("https://example.test/cat.png")]
    Researcher().run_sync(parts)

    assert captured["turn_input"] == [{"role": "user", "content": parts}]


def test_run_sync_auto_detects_images_in_a_plain_string_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain `list[str]` message auto-detects each item as text or an image by extension."""
    captured: dict[str, Any] = {}

    def fake_run_sync(agent: Any, turn_input: Any, **kwargs: Any) -> _FakeResult:
        captured["turn_input"] = turn_input
        return _FakeResult()

    monkeypatch.setattr("runa.agent.Runner.run_sync", staticmethod(fake_run_sync))

    Researcher().run_sync(["what's in this image?", "https://example.test/cat.jpg"])

    assert captured["turn_input"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.test/cat.jpg"}},
            ],
        }
    ]


def test_run_sync_wraps_multimodal_message_in_a_message_list_for_a_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a `session`, a multimodal message is wrapped in a one-item message list.

    Unlike a plain string (sent as-is, since `Runner.run`'s session path wraps it itself), a
    parts list isn't a valid top-level `input` for the session path, so `Agent` must wrap it.
    """
    from runa import content

    captured: dict[str, Any] = {}

    def fake_run_sync(agent: Any, turn_input: Any, **kwargs: Any) -> _FakeResult:
        captured["turn_input"] = turn_input
        return _FakeResult()

    monkeypatch.setattr("runa.agent.Runner.run_sync", staticmethod(fake_run_sync))

    parts = [content.image("https://example.test/cat.png")]
    Researcher().run_sync(parts, session=cast(Any, object()))

    assert captured["turn_input"] == [{"role": "user", "content": parts}]


def test_run_sync_catches_runa_error_as_error_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `RunaError` (guardrail tripwire, `MaxTurnsExceeded`, ...) is captured, not raised.

    `self.history` is left unchanged, since the turn never completed.
    """
    exc = MaxTurnsExceeded("too many turns")
    exc.run_data = RunErrorDetails(
        input="hi",
        new_items=[],
        raw_responses=[],
        last_agent=cast(Any, None),
        context_wrapper=RunContextWrapper(
            context=None, usage=Usage(input_tokens=5, output_tokens=6)
        ),
        input_guardrail_results=[],
        output_guardrail_results=[],
    )

    def fake_run_sync(*args: Any, **kwargs: Any) -> _FakeResult:
        raise exc

    monkeypatch.setattr("runa.agent.Runner.run_sync", staticmethod(fake_run_sync))

    agent = Researcher()
    run = agent.run_sync("hi")

    assert run.output is None
    assert run.status == "error"
    assert run.error == "too many turns"
    assert run.usage == Usage(input_tokens=5, output_tokens=6)
    assert agent.last_usage == Usage(input_tokens=5, output_tokens=6)
    assert agent.history == []


def test_run_sync_trace_populated_regardless_of_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Run.trace` is whatever `Runner.run_sync()` produced, independent of a custom `hooks=`."""
    from runa.tracing import Trace

    fake_trace = Trace(id="t1", name="Researcher", start_time=0.0)

    class _ResultWithTrace(_FakeResult):
        trace = fake_trace

    def fake_run_sync(*args: Any, **kwargs: Any) -> _ResultWithTrace:
        return _ResultWithTrace()

    monkeypatch.setattr("runa.agent.Runner.run_sync", staticmethod(fake_run_sync))

    run = Researcher().run_sync("hi", hooks=LoggingRunHooks())

    assert run.trace is not None
    assert run.trace.name == "Researcher"


def _final_message(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text, "tool_calls": None}


def _tool_call_message(name: str, arguments: str, call_id: str = "call_1") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


class _ScriptedModel:
    """A `Model` stand-in returning pre-scripted messages, run through the real `Runner`."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = list(messages)

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:  # noqa: ANN002, ANN003
        return ModelResponse(
            output=[self._messages.pop(0)],
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, requests=1),
        )


def test_run_sync_trace_has_agent_and_tool_spans() -> None:
    """A real run's `Run.trace` has an `"agent"` root span and a `"tool"` span for a called tool."""

    @tool
    def now() -> str:
        """Return a fixed time."""
        return "2024-01-01T00:00:00"

    class TimeAgent(Agent):
        name = "TimeAgent"
        instructions = "Answer with the time."
        tools = [now]
        model = _ScriptedModel(
            [_tool_call_message("now", "{}"), _final_message("the time is 2024")]
        )

    run = TimeAgent().run_sync("what time is it?")

    assert run.status == "completed"
    assert run.trace is not None
    types_by_name = {span.name: span.type for span in run.trace.spans}
    assert types_by_name["TimeAgent"] == "agent"
    assert types_by_name["now"] == "tool"
    assert all(span.status == "ok" for span in run.trace.spans)
    root = next(span for span in run.trace.spans if span.parent_id is None)
    assert root.input == "what time is it?"


def test_run_sync_trace_records_a_tool_error_span() -> None:
    """A tool that raises doesn't stop the run, but its span in `Run.trace` records the error."""

    @tool
    def boom() -> str:
        """Raise unconditionally."""
        raise ValueError("boom")

    class BoomAgent(Agent):
        name = "BoomAgent"
        instructions = "Call the tool."
        tools = [boom]
        model = _ScriptedModel(
            [_tool_call_message("boom", "{}"), _final_message("the tool failed")]
        )

    run = BoomAgent().run_sync("go")

    assert run.trace is not None
    tool_spans = [span for span in run.trace.spans if span.type == "tool"]
    assert len(tool_spans) == 1
    assert tool_spans[0].status == "error"
    assert tool_spans[0].error is not None


def test_run_streamed_yields_events_and_updates_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_streamed` yields every event, then appends the turn to history and records usage."""

    class _FakeStreaming:
        context_wrapper = RunContextWrapper(
            context=None, usage=Usage(input_tokens=3, output_tokens=4)
        )

        def __aiter__(self) -> AsyncIterator[Any]:
            async def _events() -> AsyncIterator[Any]:
                yield "event-1"
                yield "event-2"

            return _events()

        def to_input_list(self) -> list[Any]:
            return [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]

    def fake_run_streamed(*args: Any, **kwargs: Any) -> _FakeStreaming:
        return _FakeStreaming()

    monkeypatch.setattr("runa.agent.Runner.run_streamed", staticmethod(fake_run_streamed))

    agent = Researcher()

    async def _consume() -> list[Any]:
        return [event async for event in agent.run_streamed("hi")]

    events = asyncio.run(_consume())

    assert events == ["event-1", "event-2"]
    assert agent.history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]
    assert agent.last_usage == Usage(input_tokens=3, output_tokens=4)
    assert agent.usage == Usage(input_tokens=3, output_tokens=4)


def test_run_streamed_defaults_to_logging_run_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_streamed` defaults to a `LoggingRunHooks` when none is given."""
    captured: dict[str, Any] = {}

    class _FakeStreaming:
        context_wrapper = RunContextWrapper(context=None)

        def __aiter__(self) -> AsyncIterator[Any]:
            async def _events() -> AsyncIterator[Any]:
                return
                yield  # pragma: no cover -- makes this an async generator with no items

            return _events()

        def to_input_list(self) -> list[Any]:
            return []

    def fake_run_streamed(*args: Any, hooks: Any, **kwargs: Any) -> _FakeStreaming:
        captured["hooks"] = hooks
        return _FakeStreaming()

    monkeypatch.setattr("runa.agent.Runner.run_streamed", staticmethod(fake_run_streamed))

    async def _consume() -> None:
        async for _ in Researcher().run_streamed("hi"):
            pass

    asyncio.run(_consume())

    assert isinstance(captured["hooks"], LoggingRunHooks)


def test_evaluate_delegates_to_evaluate_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Agent.evaluate()` forwards straight to `runa.eval.evaluate.evaluate_agent`."""
    from runa.eval.case import Case

    captured: dict[str, Any] = {}

    async def fake_evaluate_agent(agent: Any, dataset: Any, **kwargs: Any) -> str:
        captured["agent"] = agent
        captured["dataset"] = dataset
        captured["kwargs"] = kwargs
        return "a report"

    monkeypatch.setattr("runa.eval.evaluate.evaluate_agent", fake_evaluate_agent)

    agent = Researcher()
    dataset = [Case(input="hi")]
    report = asyncio.run(agent.evaluate(dataset, judge="gpt-5.4", threshold=0.8))

    assert report == "a report"
    assert captured["agent"] is agent
    assert captured["dataset"] is dataset
    assert captured["kwargs"] == {"judge": "gpt-5.4", "threshold": 0.8, "thresholds": None}
