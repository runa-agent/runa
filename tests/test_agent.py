"""Tests for the handoff / delegate / auto subagent wiring, and `Agent.run`/`run_sync`."""

import asyncio
import importlib.util
import json
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, nullcontext
from dataclasses import dataclass, fields
from pathlib import Path
from types import ModuleType
from typing import Any, TypedDict, cast

import pytest
from helpers import scripted_agent, shape_of
from pydantic import BaseModel

from runa import Agent, _loop, db, tracing
from runa._models import StreamDelta
from runa._types import ModelResponse, ModelSettings, RunContext, Usage
from runa.agent import _AGENT_FIELDS, Subagent
from runa.db.sqlite import MemoryDatabase
from runa.exceptions import OperatorError, UserError
from runa.guardrail import Phase, guardrail
from runa.handoff import Delegate
from runa.knowledge import Knowledge
from runa.lifecycle import Hooks, LoggingHooks
from runa.mcp import MCPServer, _Connection, _MCPServer, _Transport
from runa.memory import Memory
from runa.run import Run
from runa.run_internal.agent_shape import shape_of as built_shape
from runa.run_state import RunState
from runa.session.store import DatabaseSession
from runa.tool import FunctionTool, tool
from runa.transcript import Transcript


def _handoff_names(agent: Agent) -> list[str]:
    """Names of the agents an agent hands off to."""
    return [handoff.agent.name for handoff in built_shape(agent).handoffs.values()]


def _tool_names(agent: Agent) -> list[str]:
    """Names of the tools an agent offers: its own plus memory's and knowledge's search."""
    return [t.name for t in built_shape(agent).tools if isinstance(t, FunctionTool)]


def _delegate_names(agent: Agent) -> list[str]:
    """The tool names an agent's delegates are called by."""
    return [delegate.tool.name for delegate in built_shape(agent).delegates.values()]


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


def test_graph_draws_delegates_and_handoffs_as_agents() -> None:
    """A delegate is an agent node with a dotted edge, a handoff one with a dashed edge."""

    class Lead(Agent):
        name = "Lead"
        instructions = "Lead."
        subagents = [Researcher.delegate, Translator.handoff]

    source = Lead().graph.source

    assert "label=Researcher" in source and "label=Translator" in source
    assert "style=dotted" in source and "style=dashed" in source
    assert "lightyellow" not in source  # no plain tool box for the delegate


def test_agent_cannot_be_instantiated_directly() -> None:
    """`Agent` itself must be subclassed; it isn't a usable agent on its own."""
    with pytest.raises(UserError):
        Agent(name="Bare")


def test_missing_name_is_a_user_error() -> None:
    """A subclass that never declares `name` fails like any other declaration mistake."""

    class Nameless(Agent):
        instructions = "nameless"

    with pytest.raises(UserError, match="missing the required 'name'"):
        Nameless()


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


def test_a_prompt_is_read_once_per_process(tmp_path: Path) -> None:
    """An agent built per request reads nothing from disk: its class's prompt is already held."""
    agents_dir = tmp_path / "app" / "agents"
    prompts_dir = tmp_path / "app" / "prompts"
    agents_dir.mkdir(parents=True)
    prompts_dir.mkdir(parents=True)
    (agents_dir / "greeter_agent.py").write_text(
        "from runa import Agent\n\n\nclass GreeterAgent(Agent):\n    name = 'greeter_agent'\n"
    )
    prompt = prompts_dir / "greeter_agent.md"
    prompt.write_text("You greet warmly.\n")
    module = _import_module_from_file("greeter_agent_held", agents_dir / "greeter_agent.py")
    module.GreeterAgent()

    prompt.write_text("You greet coldly.\n")

    assert module.GreeterAgent().instructions == "You greet warmly."


def test_instructions_stay_empty_outside_an_agents_directory() -> None:
    """A class not defined under an `agents/` directory looks for no prompt file at all."""

    class NoPrompt(Agent):
        name = "no_prompt_agent"

    assert NoPrompt().instructions is None


def test_a_missing_prompt_file_is_never_written(tmp_path: Path) -> None:
    """Construction only reads: no prompt file means empty `instructions`, not a new file.

    A constructor that scaffolds would make an immutable image (or any read-only `app/`) raise
    `PermissionError` out of `Agent()`, before a run exists to report it as `status="error"`.
    """
    agents_dir = tmp_path / "app" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "greeter_agent.py").write_text(
        "from runa import Agent\n\n\nclass GreeterAgent(Agent):\n    name = 'greeter_agent'\n"
    )

    module = _import_module_from_file("greeter_agent_no_prompt", agents_dir / "greeter_agent.py")
    agent = module.GreeterAgent()

    assert agent.instructions is None
    assert not (tmp_path / "app" / "prompts").exists()


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
    assert agent.tools == ()


def test_delegate_adds_only_to_tools() -> None:
    """`.delegate` wires the subagent as a tool, not a handoff."""

    class Main(Agent):
        name = "Main"
        instructions = "main"
        subagents = [Researcher.delegate]

    agent = Main()

    assert built_shape(agent).handoffs == {}
    assert _delegate_names(agent) == [Delegate.from_agent(Researcher()).tool.name]


def test_h_and_d_are_shorthand_for_handoff_and_delegate() -> None:
    """`.h`/`.d` bind exactly like `.handoff`/`.delegate`, same subagent, same mode."""

    class Main(Agent):
        name = "Main"
        instructions = "main"
        subagents = [Researcher.h, Translator.d]

    agent = Main()

    assert _handoff_names(agent) == ["Researcher"]
    assert _delegate_names(agent) == [Delegate.from_agent(Translator()).tool.name]


def test_two_agents_listing_each_other_build_into_one_cycle() -> None:
    """A specialist handing back to triage closes onto the triage agent being built."""

    class Triage(Agent):
        name = "Triage"
        instructions = "triage"
        subagents: Sequence[type[Agent] | Subagent] = ()  # filled in below, once `Billing` exists

    class Billing(Agent):
        name = "Billing"
        instructions = "billing"
        subagents = [Triage.handoff]

    Triage.subagents = [Billing.delegate]

    triage = Triage()

    (billing,) = [delegate.agent for delegate in built_shape(triage).delegates.values()]
    assert billing is not None
    assert [handoff.agent for handoff in built_shape(billing).handoffs.values()] == [triage]


def test_a_class_listed_twice_in_one_graph_is_built_once() -> None:
    """A leaf shared by two subagents is one instance, not one per path to it."""

    class Left(Agent):
        name = "Left"
        instructions = "left"
        subagents = [Translator.delegate]

    class Right(Agent):
        name = "Right"
        instructions = "right"
        subagents = [Translator.delegate]

    class Main(Agent):
        name = "Main"
        instructions = "main"
        subagents = [Left.delegate, Right.delegate]

    left, right = [delegate.agent for delegate in built_shape(Main()).delegates.values()]

    assert left is not None and right is not None
    assert (
        next(iter(built_shape(left).delegates.values())).agent
        is next(iter(built_shape(right).delegates.values())).agent
    )


def test_two_classes_sharing_a_name_in_one_graph_is_a_user_error() -> None:
    """A resumed run finds its agent by name, so one name must mean one agent in the graph."""

    class Impostor(Agent):
        name = "Researcher"
        instructions = "impostor"

    class Main(Agent):
        name = "Main"
        instructions = "main"
        subagents = [Researcher.delegate, Impostor.handoff]

    with pytest.raises(UserError, match="Researcher and Impostor are both named 'Researcher'"):
        Main()
    Researcher()  # builds: the failed graph left no names behind to collide with


def test_two_tools_offered_under_one_name_is_a_user_error() -> None:
    """An `@tool` named like a delegate would shadow it: only one of them could be called."""

    @tool
    def researcher(query: str) -> str:
        """Research a query."""
        return query

    class Main(Agent):
        name = "Main"
        instructions = "main"
        tools = [researcher]
        subagents = [Researcher.delegate]

    with pytest.raises(UserError, match="more than one tool named 'researcher'"):
        Main()


def test_delegate_tool_name_and_description_override() -> None:
    """Calling a `.delegate` subagent overrides the generated tool's name/description."""

    class Main(Agent):
        name = "Main"
        instructions = "main"
        subagents = [Researcher.delegate(tool_name="do_research", tool_description="Look into it.")]

    agent = Main()

    (delegate,) = built_shape(agent).delegates.values()
    assert delegate.tool.name == "do_research"
    assert delegate.tool.description == "Look into it."
    assert agent.tools == ()


def test_bare_subagent_wires_both_handoff_and_delegate() -> None:
    """A subagent listed without `.handoff`/`.delegate` lets the model pick either mode."""

    class Main(Agent):
        name = "Main"
        instructions = "main"
        subagents = [Researcher]

    agent = Main()

    assert _handoff_names(agent) == ["Researcher"]
    assert _delegate_names(agent) == [Delegate.from_agent(Researcher()).tool.name]


def test_mixed_modes_wire_independently() -> None:
    """Handoff and delegate subagents in the same list don't interfere with each other."""

    class Main(Agent):
        name = "Main"
        instructions = "main"
        subagents = [Researcher.handoff, Translator.delegate]

    agent = Main()

    assert _handoff_names(agent) == ["Researcher"]
    assert _delegate_names(agent) == [Delegate.from_agent(Translator()).tool.name]


def test_no_subagents_leaves_handoffs_and_tools_empty() -> None:
    """An agent with no `subagents` attribute wires up cleanly."""

    class Main(Agent):
        name = "Main"
        instructions = "main"

    agent = Main()

    assert built_shape(agent).handoffs == {}
    assert agent.tools == ()


def test_dict_subagents_are_rejected() -> None:
    """`subagents` is a list; a dict grouping it by mode is a `UserError`, not keys as agents."""

    class Main(Agent):
        name = "Main"
        instructions = "main"
        subagents = cast(Any, {"handoff": [Researcher]})

    with pytest.raises(UserError, match="subagents must be a list"):
        Main()


def test_mcp_servers_is_not_a_setting() -> None:
    """`mcp` is the one way to declare MCP servers; `mcp_servers` is only what it resolved to."""

    class Main(Agent):
        name = "Main"
        instructions = "main"
        mcp_servers = cast(Any, [])

    with pytest.raises(UserError, match="'mcp_servers', which is not an Agent setting"):
        Main()


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

    assert built_shape(agent).memory is None
    assert "search_memory" not in _tool_names(agent)


def test_memory_auto_gives_a_default_memory_instance_and_no_tool() -> None:
    """`memory="auto"` builds a default `Memory` for the run lifecycle to use, with no tool.

    Automatic retrieval/persistence is the run lifecycle's job (see `test_run_loop.py`), not
    something `Agent.__init__` does.
    """

    class AutoMemory(Agent):
        name = "AutoMemory"
        instructions = "auto memory"
        memory = "auto"

    agent = AutoMemory()

    assert isinstance(built_shape(agent).memory, Memory)
    assert "search_memory" not in _tool_names(agent)


def test_memory_llm_adds_a_tool_and_leaves_self_memory_none() -> None:
    """`memory="llm"` gives the model a `search_memory` tool instead of automatic retrieval."""

    class LLMMemory(Agent):
        name = "LLMMemory"
        instructions = "llm memory"
        memory = "llm"

    agent = LLMMemory()

    assert built_shape(agent).memory is None
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
        async def retrieve(self, query: str, run_context: Any) -> Any:
            return None

        async def after_turn(
            self, query: str, output: Any, run_context: Any, model: Any
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

    assert isinstance(built_shape(agent).memory, Memory)


def test_memory_rejects_an_unknown_string() -> None:
    """A `memory=` string other than `"auto"`/`"llm"` fails clearly instead of misbehaving."""
    from runa.exceptions import UserError

    class BadMemory(Agent):
        name = "BadMemory"
        instructions = "bad memory"
        memory = "sometimes"  # type: ignore[assignment]

    with pytest.raises(UserError, match="memory"):
        BadMemory()


def test_knowledge_defaults_to_none() -> None:
    """Without `knowledge=`, an agent behaves exactly as if `runa.knowledge` didn't exist."""
    agent = Researcher()

    assert built_shape(agent).knowledge is None
    assert "search_knowledge" not in _tool_names(agent)


def test_knowledge_auto_gives_a_default_knowledge_instance_and_no_tool() -> None:
    """`knowledge="auto"` builds a default `Knowledge` for the run lifecycle to use, with no tool.

    Automatic retrieval is the run lifecycle's job (see `test_run_loop.py`), not something
    `Agent.__init__` does.
    """

    class AutoKnowledge(Agent):
        name = "AutoKnowledge"
        instructions = "auto knowledge"
        knowledge = "auto"

    agent = AutoKnowledge()

    assert isinstance(built_shape(agent).knowledge, Knowledge)
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

    assert built_shape(agent).knowledge is None
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
        async def retrieve(self, query: str, run_context: Any) -> Any:
            return None

        async def after_turn(
            self, query: str, output: Any, run_context: Any, model: Any
        ) -> list[str]:
            return []

        async def ingest(self) -> int:
            return 0

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

    assert isinstance(built_shape(agent).knowledge, Knowledge)


def test_knowledge_rejects_an_unknown_string() -> None:
    """A `knowledge=` string other than `"auto"`/`"llm"` fails clearly instead of misbehaving."""
    from runa.exceptions import UserError

    class BadKnowledge(Agent):
        name = "BadKnowledge"
        instructions = "bad knowledge"
        knowledge = "sometimes"  # type: ignore[assignment]

    with pytest.raises(UserError, match="knowledge"):
        BadKnowledge()


def test_compact_defaults_to_no_compactor() -> None:
    """Without `compact=`, an agent behaves exactly as if `runa.compact` didn't exist."""
    assert shape_of(Researcher()).compactor is None


def test_compact_true_resolves_to_the_default_compactor() -> None:
    """`compact=True` is sugar for Runa's own rolling-window `Compactor`, resolved at construction.

    A run reads its shape's `compactor`, so `True` has become a callable by the time a turn
    starts -- it is not the loop's job to know what the bool meant.
    """
    from runa.compact import default_compactor

    class Compacting(Agent):
        name = "Compacting"
        instructions = "compacts"
        compact = True

    assert shape_of(Compacting()).compactor is default_compactor


def test_compact_accepts_a_custom_compactor_callable() -> None:
    """`compact=` also takes any `(items, usage_tokens) -> items|None` callable, passed through."""

    def drop_oldest_item(items: list[Any], usage_tokens: int) -> list[Any] | None:
        return items[1:]

    class Compacting(Agent):
        name = "Compacting"
        instructions = "compacts"
        compact = drop_oldest_item

    assert shape_of(Compacting()).compactor is drop_oldest_item


def test_compact_false_resolves_to_no_compactor() -> None:
    """`compact=False` is off, the same as not declaring it -- not a falsy value trusted onward."""

    class NotCompacting(Agent):
        name = "NotCompacting"
        instructions = "does not compact"
        compact = False

    assert shape_of(NotCompacting()).compactor is None


def test_compact_rejects_a_value_that_is_neither_a_bool_nor_a_callable() -> None:
    """A bad `compact=` fails at construction, not as a `TypeError` mid-run.

    The regression this guards: a truthy non-callable used to be "trusted as already being a
    `Compactor`" and only blew up inside the turn loop, after a model call had been paid for.
    """
    from runa.exceptions import UserError

    class BadCompact(Agent):
        name = "BadCompact"
        instructions = "bad compact"
        compact = cast(Any, "sometimes")

    with pytest.raises(UserError, match="compact"):
        BadCompact()


def test_model_rejects_a_value_that_is_neither_a_name_nor_a_model() -> None:
    """A bad `model=` fails at construction, the way `compact=`/`memory=`/`knowledge=` do.

    Resolving a model name has to wait for the run (an unset API key must not stop an agent from
    being built), but rejecting a value no provider could ever resolve does not: without this the
    first sign of it was an `AttributeError` inside the turn loop, an `llm` span already open.
    """
    from runa.exceptions import UserError

    class BadModel(Agent):
        name = "BadModel"
        instructions = "bad model"
        model = 3

    with pytest.raises(UserError, match="model"):
        BadModel()


def test_a_misspelled_model_name_is_rejected_like_a_misspelled_setting() -> None:
    """A typo in `model`'s value fails at construction too, not just a typo in its name.

    The regression this guards is the worse half of the pair below: `modell = "claude-sonnet-5"`
    at least ran a working model, while `model = "claud-sonnet-5"` used to fall through to the
    OpenAI endpoint, sending the app's prompts to another vendor under another key and reporting
    the 404 as model misbehaviour, mid-run.
    """
    from runa.exceptions import UserError

    class TypoedModel(Agent):
        name = "TypoedModel"
        instructions = "typo"
        model = "claud-sonnet-5"

    with pytest.raises(UserError, match="Did you mean a claude-\\* model"):
        TypoedModel()


def test_a_model_instance_is_the_escape_hatch_for_an_unnamed_endpoint() -> None:
    """Closing the prefix set doesn't close the door: a `Model` passes through untouched.

    The sanctioned shape for a gateway, a proxy or something self-hosted, and the reason an
    unroutable *name* can afford to be an error: there is another way to say "not one of these".
    """
    import httpx2 as httpx

    from runa._models import OpenAICompatibleModel

    gateway = OpenAICompatibleModel(
        "mixtral-8x7b", httpx.AsyncClient(base_url="https://gateway.internal/v1/")
    )

    class Gatewayed(Agent):
        name = "Gatewayed"
        instructions = "gateway"

    assert Gatewayed(model=gateway).model is gateway


def test_a_misspelled_class_attribute_is_rejected_by_name() -> None:
    """A typo'd setting fails at construction instead of silently configuring nothing.

    The regression this guards: `modell = "claude-sonnet-5"` used to leave the agent on the
    default model, which surfaces as "the agent answers oddly" rather than as an error.
    """
    from runa.exceptions import UserError

    class Typo(Agent):
        name = "Typo"
        instructions = "typo"
        modell = "claude-sonnet-5"

    with pytest.raises(UserError, match="'modell'.*Did you mean 'model'"):
        Typo()


def test_a_misspelled_constructor_kwarg_is_rejected_by_the_signature() -> None:
    """An override is a named parameter, so a typo is caught without any of Runa's own checking.

    A type checker flags the call before it runs (which is most of why the signature is written
    out rather than taken as `**kwargs`), and CPython itself suggests the name it meant -- so
    `_reject_unknown_settings` is left to the class body, where no such check exists.
    """

    class Fine(Agent):
        name = "Fine"
        instructions = "fine"

    with pytest.raises(TypeError, match="'tolls'.*Did you mean 'tools'"):
        Fine(tolls=[])  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="unexpected keyword argument 'nonsense'"):
        Fine(nonsense=1)  # type: ignore[call-arg]


_PER_RUN_STATE = frozenset({"history", "usage", "last_usage"})
"""What `copy.copy(agent)` clears: one run's state rather than configuration, so it goes nowhere."""


def _every_setting_agent() -> Agent:
    """An agent with every setting it can hold given a value that is nobody's default."""

    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    @tool
    def mcp_ping() -> str:
        """Ping, as an MCP server's tool."""
        return "pong"

    @guardrail
    def block_empty(x: str) -> bool:
        """Trip on an empty message."""
        return not x.strip()

    @dataclass
    class Answer:
        text: str

    class _Server:
        async def list_tools(self) -> list[Any]:
            return [mcp_ping]

    class Everything(Agent):
        name = "Everything"
        instructions = "everything"
        model = _ScriptedModel([_final_message("hi")])
        model_settings = ModelSettings(temperature=0.5)
        tools = [ping]
        subagents = [Researcher.handoff, Translator.delegate]
        guardrails = [block_empty.input]
        mcp = [_Server()]
        output_type = Answer
        hooks = Hooks[Any]()
        memory = "auto"
        knowledge = "auto"
        compact = True
        max_turns = 3
        max_tokens = 100
        timeout = 5.0

    return Everything()


def test_the_shape_holds_every_setting_the_agent_stores() -> None:
    """`_reject_unknown_settings`, inverted: a setting stored on the agent that no run reads.

    The turn loop reads an agent only through its shape, so a setting stored on the instance and
    missing from the shape is accepted, validated -- and reaches no run. Derived fields go by the
    setting they come from (`compactor` is `compact`'s).
    """
    from runa.run_internal.agent_shape import AgentShape

    agent = _every_setting_agent()
    stored = {name for name in vars(agent) if not name.startswith("_")} - _PER_RUN_STATE
    translated = {"agent", "compactor", "mcp_servers", "handoffs", "delegates"}
    read = {field.name for field in fields(AgentShape)} - translated | {
        "compact",
        "mcp",
        "subagents",
    }

    assert stored == read, f"stored but read by no run: {sorted(stored - read)}"


def test_with_mcp_tools_adds_the_servers_tools_for_one_run() -> None:
    """A run's shape lists the MCP servers' tools after the agent's own, leaving the agent's be."""
    agent = _every_setting_agent()

    shape = asyncio.run(built_shape(agent).with_mcp_tools())

    assert shape.agent is agent
    assert [t.name for t in shape.tools] == [*_tool_names(agent), "mcp_ping"]
    assert [t.name for t in built_shape(agent).tools] == _tool_names(agent)
    assert set(shape.handoffs) == {"transfer_to_researcher"}


def test_an_mcp_tool_named_like_the_agents_own_is_a_user_error() -> None:
    """A server's tools join the offered names when a run lists them, and are checked there."""

    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    class _Server:
        async def list_tools(self) -> list[Any]:
            return [ping]

    agent = scripted_agent(_ScriptedModel([]), tools=[ping], mcp=[_Server()])

    with pytest.raises(UserError, match="more than one tool named 'ping'"):
        asyncio.run(built_shape(agent).with_mcp_tools())


class _Declared(Agent):
    name = "declared"
    memory = "auto"
    knowledge = "auto"
    subagents = [Researcher.handoff]


def test_instances_of_a_class_share_what_it_declares_but_not_their_graph() -> None:
    """Two plain instances share one `Memory` and `Knowledge`, each with subagents of its own."""
    first, second = _Declared(), _Declared()

    assert built_shape(first).memory is built_shape(second).memory
    assert built_shape(first).knowledge is built_shape(second).knowledge
    assert built_shape(first).agent is first
    assert built_shape(second).agent is second
    (first_researcher,) = [handoff.agent for handoff in built_shape(first).handoffs.values()]
    (second_researcher,) = [handoff.agent for handoff in built_shape(second).handoffs.values()]
    assert first_researcher is not second_researcher


def test_an_override_resolves_on_its_own_instance_alone() -> None:
    """A passed setting or a reassigned one changes that instance, never its class's others."""
    plain = _Declared()
    passed = _Declared(memory=None)
    reassigned = _Declared()
    reassigned.max_turns = 3

    assert built_shape(passed).memory is None
    assert built_shape(reassigned).max_turns == 3
    assert built_shape(reassigned).memory is not built_shape(plain).memory
    assert built_shape(_Declared()).memory is built_shape(plain).memory
    assert built_shape(_Declared()).max_turns == 10


def test_a_run_reuses_the_shape_built_at_construction() -> None:
    """Recording a run's history and usage leaves the shape be: it is built once, not per run."""
    agent = scripted_agent(_ScriptedModel([_final_message("one"), _final_message("two")]))
    shape = built_shape(agent)

    agent.run_sync("hi")
    agent.run_sync("again")

    assert built_shape(agent) is shape


def test_reassigning_a_setting_rebuilds_the_shape_a_run_reads() -> None:
    """`agent.max_turns = 3` reaches the next run, and a bad value fails where it is set."""
    agent = scripted_agent(_ScriptedModel([_final_message("hi")]))

    agent.max_turns = 3

    assert built_shape(agent).max_turns == 3
    with pytest.raises(UserError, match="compact must be"):
        agent.compact = cast(Any, "yes")


def test_the_list_settings_change_by_reassignment_only() -> None:
    """Stored as tuples, so an in-place `append` fails rather than reaching no run."""
    agent = scripted_agent(_ScriptedModel([_final_message("hi")]))

    with pytest.raises(AttributeError):
        agent.tools.append(_refund_tool([]))  # type: ignore[attr-defined]


def test_every_setting_can_be_passed_to_the_constructor_instead_of_declared() -> None:
    """The merge's other half: a setting passed to `__init__` is read, not merely accepted.

    `_settings` falls every parameter back to the class attribute of the same name in one pass,
    but each lands on the instance on a line of its own, so the mistake to guard against is a
    parameter that is accepted and resolved and then never read -- which would
    leave `SupportAgent(model=...)` silently running the declared model. Every name in
    `_AGENT_FIELDS` is passed here, which is what makes that a line-by-line check.
    """

    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    @guardrail
    def block_empty(x: str) -> bool:
        """Trip on an empty message."""
        return not x.strip()

    @dataclass
    class Answer:
        text: str

    class _Server:
        async def list_tools(self) -> list[Any]:
            return []

    class Blank(Agent):
        """Declares nothing, so every setting below can only have come from the constructor."""

    hooks = Hooks[Any]()
    model_settings = ModelSettings(temperature=0.5)
    passed: dict[str, Any] = {
        "name": "Overridden",
        "instructions": "overridden",
        "model": "claude-sonnet-5",
        "model_settings": model_settings,
        "tools": [ping],
        "subagents": [Translator.delegate],
        "guardrails": [block_empty.input],
        "mcp": [_Server()],
        "output_type": Answer,
        "hooks": hooks,
        "memory": "auto",
        "knowledge": "auto",
        "compact": True,
        "max_turns": 3,
        "max_tokens": 100,
        "timeout": 5.0,
    }
    assert set(passed) == set(_AGENT_FIELDS), "a setting with no override checked here"

    agent = Blank(**passed)

    assert (agent.name, agent.instructions) == ("Overridden", "overridden")
    assert agent.model == "claude-sonnet-5"
    assert agent.model_settings is model_settings
    assert _tool_names(agent) == ["ping"]
    assert _delegate_names(agent) == [Delegate.from_agent(Translator()).tool.name]
    assert agent.guardrails == tuple(passed["guardrails"])
    assert built_shape(agent).mcp_servers == tuple(passed["mcp"])
    assert agent.output_type is Answer
    assert agent.hooks is hooks
    assert isinstance(built_shape(agent).memory, Memory)
    assert isinstance(built_shape(agent).knowledge, Knowledge)
    assert agent.compact is True
    assert (agent.max_turns, agent.max_tokens, agent.timeout) == (3, 100, 5.0)


def test_an_override_of_none_turns_a_declared_setting_off() -> None:
    """`None` is a value, not silence: the one reason a parameter defaults to a sentinel.

    `memory`/`knowledge`/`output_type`/`instructions` all take `None` to mean "none of that",
    which a `None` default would make indistinguishable from not passing the setting at all.
    """

    class Remembering(Agent):
        name = "Remembering"
        instructions = "remember"
        memory = "auto"
        knowledge = "auto"

    assert Remembering().memory is not None
    assert Remembering(memory=None, knowledge=None).memory is None
    assert Remembering(memory=None, knowledge=None).knowledge is None


def test_a_subclass_keeps_its_own_methods_and_underscored_state() -> None:
    """Only non-underscore data is checked, so helpers and private state stay free."""

    class Helpful(Agent):
        name = "Helpful"
        instructions = "helpful"
        _threshold = 3

        def helper(self) -> int:
            return self._threshold

        @property
        def doubled(self) -> int:
            return self._threshold * 2

        @staticmethod
        def tripled() -> int:
            return 9

    assert Helpful().helper() == 3


def test_a_mixin_may_carry_whatever_attributes_it_likes() -> None:
    """Only Agent subclasses in the MRO are checked; a mixin isn't claiming to be Runa config."""

    class Tenant:
        tenant_id = "acme"

    class Scoped(Tenant, Agent):
        name = "Scoped"
        instructions = "scoped"

    assert Scoped().name == "Scoped"


def test_an_inherited_setting_is_recognized_and_an_inherited_typo_is_not() -> None:
    """The check walks the subclass chain: a shared base's settings pass, its typos don't."""
    from runa.exceptions import UserError

    class Base(Agent):
        model = _ScriptedModel([_final_message("hi")])
        max_turns = 3

    class Child(Base):
        name = "Child"
        instructions = "child"

    assert Child().max_turns == 3

    class TypoBase(Agent):
        instrucshions = "oops"

    class TypoChild(TypoBase):
        name = "TypoChild"
        instructions = "child"

    with pytest.raises(UserError, match="TypoBase declares 'instrucshions'"):
        TypoChild()


def test_subagents_and_guardrails_can_also_be_passed_to_the_constructor() -> None:
    """Every accepted name works in both places, so neither list is quietly dropped as a kwarg."""

    @guardrail
    def block_empty(x: str) -> bool:
        """Trip on an empty message."""
        return not x.strip()

    class Lead(Agent):
        name = "Lead"
        instructions = "lead"

    agent = Lead(subagents=[Researcher.handoff], guardrails=[block_empty.input])

    assert _handoff_names(agent) == ["Researcher"]
    assert [g.name for g in shape_of(agent).guardrails[Phase.INPUT]] == ["block_empty"]


def test_model_accepts_a_model_instance_without_a_provider() -> None:
    """A scripted `Model` passes through untouched: the check is structural, not an isinstance."""

    class Scripted(Agent):
        name = "Scripted"
        instructions = "scripted"
        model = _ScriptedModel([_final_message("hi")])

    assert isinstance(Scripted().model, _ScriptedModel)


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


def _two_arg_instructions(context: RunContext[_Ctx], agent: Any) -> str:
    return f"{agent.name}:{context.context.label}"


async def _prompt_of(agent: Any, label: str) -> str | None:
    """`agent.instructions` as the loop resolves it."""
    from runa.run_internal.run_loop import _resolve_instructions

    return await _resolve_instructions(built_shape(agent), RunContext(context=_Ctx(label=label)))


def test_single_arg_instructions_resolves_from_run_context() -> None:
    """A one-parameter `(context) -> str` `instructions` is adapted to the runner's 2-arg shape."""

    class Dynamic(Agent):
        name = "Dynamic"
        instructions: Any = _single_arg_instructions

    agent = Dynamic()

    prompt = asyncio.run(_prompt_of(agent, "hi"))

    assert prompt == "context=hi"


def test_two_arg_instructions_still_supported() -> None:
    """A native runner-style `(context, agent) -> str` `instructions` passes through unadapted."""

    class Dynamic(Agent):
        name = "Dynamic"
        instructions: Any = _two_arg_instructions

    agent = Dynamic()

    prompt = asyncio.run(_prompt_of(agent, "hi"))

    assert prompt == "Dynamic:hi"


def test_string_instructions_pass_through_unchanged() -> None:
    """Plain string instructions are unaffected by the dynamic-instructions adapter."""
    assert Researcher().instructions == "You research topics."


def _transcript_of(input: list[Any], *generated: Any) -> Transcript:
    """A session-less run's transcript: started from `input`, having produced `generated`."""
    transcript = Transcript.start(input)
    transcript.record(*generated)
    return transcript


def _loop_run(**overrides: Any) -> Run:
    """The `Run` the turn loop hands back, for a test that fakes the loop out.

    There is one result shape now, so a stand-in for the loop's output is the real `Run` a caller
    receives rather than a second class shaped approximately like it. What `Agent.run` still does
    on top of this is the bookkeeping these tests are about: `last_usage`, `usage`, `history`.
    """
    defaults: dict[str, Any] = dict(
        output="ok", trace=None, usage=Usage(input_tokens=1, output_tokens=2)
    )
    defaults.update(overrides)
    return Run(**defaults)


def _async(fake: Any) -> Any:
    """Wrap a sync `_run_async` stand-in as the coroutine function `Agent.run` awaits."""

    async def run(*args: Any, **kwargs: Any) -> Any:
        return fake(*args, **kwargs)

    return run


def _streaming(*events: Any, **overrides: Any) -> Any:
    """A `_run_async` stand-in that emits `events`, then finishes as `_loop_run(**overrides)`.

    `Agent.run_streamed` drives the loop through its `emit` callback and turns that into an async
    iterator itself, so faking the loop (rather than a second streaming result class) is what
    puts the Agent's own queue plumbing under test.
    """

    async def run(agent: Any, turn_input: Any, *, emit: Any, **kwargs: Any) -> Run:
        for event in events:
            emit(event)
        return _loop_run(**overrides)

    return run


def test_every_run_shape_defaults_to_logging_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no `hooks=`, `run_sync` and `run_streamed` both log through `LoggingHooks`."""
    created: list[Any] = []

    class _Recording(LoggingHooks):
        def __init__(self) -> None:
            super().__init__()
            created.append(self)

    monkeypatch.setattr("runa.run_internal.run_loop.LoggingHooks", _Recording)

    class Echo(Agent):
        name = "Echo"
        instructions = "Echo."
        model = _ScriptedModel([_final_message("one")])

    agent = Echo()
    agent.run_sync("hi")

    async def consume() -> None:
        async for _ in agent.run_streamed("again"):
            pass

    agent.model = _ScriptedStreamModel("two")
    asyncio.run(consume())

    assert len(created) == 2


def test_run_sync_explicit_hooks_override_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit `hooks` argument is used instead of the default combined hooks."""
    captured: dict[str, Any] = {}
    custom_hooks = LoggingHooks()

    def fake_run_sync(*args: Any, hooks: Any, **kwargs: Any) -> Run:
        captured["hooks"] = hooks
        return _loop_run()

    monkeypatch.setattr("runa.agent._run_async", _async(fake_run_sync))

    Researcher().run_sync("hi", hooks=custom_hooks)

    assert captured["hooks"] is custom_hooks


def test_run_sync_records_and_accumulates_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_sync` records the call's usage to `last_usage` and adds it to `usage`."""

    def fake_run_sync(*args: Any, **kwargs: Any) -> Run:
        return _loop_run()

    monkeypatch.setattr("runa.agent._run_async", _async(fake_run_sync))

    agent = Researcher()
    agent.run_sync("hi")
    agent.run_sync("again")

    expected_call_usage = Usage(input_tokens=1, output_tokens=2)
    assert agent.last_usage == expected_call_usage
    assert agent.usage.input_tokens == 2
    assert agent.usage.output_tokens == 4


def test_run_sync_returns_a_completed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_sync` returns a `Run` with the final output, status, and this call's usage."""

    def fake_run_sync(*args: Any, **kwargs: Any) -> Run:
        return _loop_run()

    monkeypatch.setattr("runa.agent._run_async", _async(fake_run_sync))

    run = Researcher().run_sync("hi")

    assert run.output == "ok"
    assert run.status == "completed"
    assert run.error is None
    assert run.usage == Usage(input_tokens=1, output_tokens=2)


class _ClosableServer(_MCPServer):
    """An MCP server whose connection is a pair of counters: opening and closing cost nothing."""

    def __init__(self) -> None:
        super().__init__(MCPServer(name="fake"), _Transport("fake://server", nullcontext))
        self.opens = 0
        self.closes = 0

    async def _open(self) -> _Connection:
        self.opens += 1
        stack = AsyncExitStack()
        stack.callback(self._closed)
        return _Connection(stack=stack, session=cast(Any, None), tools=[])

    def _closed(self) -> None:
        self.closes += 1


def test_run_sync_keeps_every_mcp_server_the_turns_reached_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every `run_sync` shares one loop, so a second turn reuses the first turn's connection.

    Not just the root agent's own servers: one a subagent connected is a `.stdio` subprocess
    too, which `runa chat` would otherwise respawn for every turn it takes.
    """
    server = _ClosableServer()
    subagent = Translator(mcp=[server])

    async def reach_the_subagents_server(*args: Any, **kwargs: Any) -> Run:
        await built_shape(subagent).mcp_servers[0].list_tools()
        return _loop_run()

    monkeypatch.setattr("runa.agent._run_async", reach_the_subagents_server)
    agent = Researcher()

    agent.run_sync("hi")
    agent.run_sync("again")

    assert (server.opens, server.closes) == (1, 0)


def test_the_background_loops_shutdown_closes_the_mcp_servers_run_sync_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What `run_sync` connected, even on a turn that raised, is closed when the process exits."""
    server = _ClosableServer()

    async def connect_then_raise(*args: Any, **kwargs: Any) -> Run:
        await server.list_tools()
        raise OperatorError("no such session")

    monkeypatch.setattr("runa.agent._run_async", connect_then_raise)

    with pytest.raises(OperatorError):
        Researcher(mcp=[server]).run_sync("hi")
    loop, _ = _loop._background_loop()
    _loop.run_sync(_loop._close_resources(loop))

    assert (server.opens, server.closes) == (1, 1)


def test_run_sync_passes_multimodal_message_content_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `runa.content` parts list is sent as the new user message's `content`, unchanged."""
    from runa import content

    captured: dict[str, Any] = {}

    def fake_run_sync(agent: Any, turn_input: Any, **kwargs: Any) -> Run:
        captured["turn_input"] = turn_input
        return _loop_run()

    monkeypatch.setattr("runa.agent._run_async", _async(fake_run_sync))

    parts = [content.text("what's in this image?"), content.image("https://example.test/cat.png")]
    Researcher().run_sync(parts)

    assert captured["turn_input"] == [{"role": "user", "content": parts}]


def test_run_sync_auto_detects_images_in_a_plain_string_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain `list[str]` message auto-detects each item as text or an image by extension."""
    captured: dict[str, Any] = {}

    def fake_run_sync(agent: Any, turn_input: Any, **kwargs: Any) -> Run:
        captured["turn_input"] = turn_input
        return _loop_run()

    monkeypatch.setattr("runa.agent._run_async", _async(fake_run_sync))

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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With a `session`, a multimodal message is wrapped in a one-item message list.

    Unlike a plain string (sent as-is, since the loop's session path wraps it itself), a
    parts list isn't a valid top-level `input` for the session path, so `Agent` must wrap it.
    """
    from runa import content

    captured: dict[str, Any] = {}

    def fake_run_sync(agent: Any, turn_input: Any, **kwargs: Any) -> Run:
        captured["turn_input"] = turn_input
        return _loop_run()

    monkeypatch.setattr("runa.agent._run_async", _async(fake_run_sync))

    parts = [content.image("https://example.test/cat.png")]
    Researcher().run_sync(parts, session=db.at(f"sqlite:///{tmp_path / 'runa.db'}").session("s"))

    assert captured["turn_input"] == [{"role": "user", "content": parts}]


def test_run_sync_records_usage_of_an_error_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """An `"error"` `Run` still has its usage recorded.

    `self.history` is left unchanged, since the turn never completed.
    """

    def fake_run_sync(*args: Any, **kwargs: Any) -> Run:
        return _loop_run(
            output=None,
            status="error",
            error="too many turns",
            usage=Usage(input_tokens=5, output_tokens=6),
        )

    monkeypatch.setattr("runa.agent._run_async", _async(fake_run_sync))

    agent = Researcher()
    run = agent.run_sync("hi")

    assert run.output is None
    assert run.status == "error"
    assert run.error == "too many turns"
    assert run.usage == Usage(input_tokens=5, output_tokens=6)
    assert agent.last_usage == Usage(input_tokens=5, output_tokens=6)
    assert agent.history == []


def test_run_sync_trace_populated_regardless_of_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Run.trace` is whatever the turn loop produced, independent of a custom `hooks=`."""
    from runa.tracing import Trace

    fake_trace = Trace(id="t1", name="Researcher", start_time=0.0)

    def fake_run_sync(*args: Any, **kwargs: Any) -> Run:
        return _loop_run(trace=fake_trace)

    monkeypatch.setattr("runa.agent._run_async", _async(fake_run_sync))

    run = Researcher().run_sync("hi", hooks=LoggingHooks())

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


class _ScriptedStreamModel:
    """A streaming `Model` stand-in that answers `text` in one delta."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:  # noqa: ANN002, ANN003
        raise AssertionError("only ever streamed")

    async def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[StreamDelta]:  # noqa: ANN002, ANN003
        yield StreamDelta(text=self._text)
        yield StreamDelta(usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2, requests=1))


def test_runs_inside_a_trace_block_are_grouped_under_it() -> None:
    """Each `Agent.run()` in a `tracing.trace` block keeps its own trace, grouped by its id."""

    class StepAgent(Agent):
        name = "StepAgent"
        instructions = "Answer."
        model = _ScriptedModel([_final_message("a"), _final_message("b"), _final_message("c")])

    # An `Agent` per run, because two of these three overlap: one instance runs one conversation
    # at a time (see `Agent._exclusive`), and what this test is about is the grouping.
    async def workflow() -> list[Run]:
        first = await StepAgent().run("one")
        return [first, *await asyncio.gather(StepAgent().run("two"), StepAgent().run("three"))]

    with tracing.trace("workflow") as outer:
        runs = asyncio.run(workflow())
    # `_ScriptedModel` answers `get_response` only, which is all this run needs and less than
    # the `Model` protocol `model=` is annotated with.
    scripted: Any = _ScriptedModel([_final_message("d")])
    alone = StepAgent(model=scripted).run_sync("four")

    assert all(run.trace is not None for run in runs)
    assert {run.trace.metadata["group_id"] for run in runs if run.trace} == {outer.id}
    assert len({run.trace.id for run in runs if run.trace}) == 3
    assert alone.trace is not None and "group_id" not in alone.trace.metadata


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
    monkeypatch.setattr(
        "runa.agent._run_async",
        _streaming(
            "event-1",
            "event-2",
            usage=Usage(input_tokens=3, output_tokens=4),
            _transcript=_transcript_of(
                [{"role": "user", "content": "hi"}], {"role": "assistant", "content": "ok"}
            ),
        ),
    )

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


def test_run_streamed_with_a_session_sends_only_the_new_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With a `session`, `run_streamed` passes it through and leaves `self.history` alone."""
    captured: dict[str, Any] = {}

    async def fake_run(agent: Any, turn_input: Any, *, session: Any, emit: Any, **kwargs: Any):
        captured["input"] = turn_input
        captured["session"] = session
        emit("event")
        return _loop_run(_transcript=Transcript.start([{"role": "user", "content": "hi"}]))

    monkeypatch.setattr("runa.agent._run_async", fake_run)
    agent = Researcher()
    agent.history = [{"role": "user", "content": "earlier"}]
    session = db.at(f"sqlite:///{tmp_path / 'runa.db'}").session("s")

    async def _consume() -> None:
        async for _ in agent.run_streamed("hi", session=session):
            pass

    asyncio.run(_consume())

    assert captured == {"input": "hi", "session": session}
    assert agent.history == [{"role": "user", "content": "earlier"}]


def test_a_session_id_string_resolves_through_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """`session="user-42"` is the sanctioned shape: the id is the app's, the backend is `db`'s.

    Under `memory://` that means a `DatabaseSession` on the in-memory database, with no store named
    at the call site -- the same string on a Postgres deployment resolves there instead.
    """
    monkeypatch.setenv("RUNA_DATABASE_URL", "memory://")
    captured: dict[str, Any] = {}

    def fake_run_sync(agent: Any, turn_input: Any, *, session: Any, **kwargs: Any) -> Run:
        captured["session"] = session
        return _loop_run()

    monkeypatch.setattr("runa.agent._run_async", _async(fake_run_sync))

    Researcher().run_sync("hi", session="user-42")

    assert isinstance(captured["session"], DatabaseSession)
    assert isinstance(captured["session"].database.location, MemoryDatabase)  # type: ignore[attr-defined]
    assert captured["session"].session_id == "user-42"


def test_run_streamed_resolves_a_session_id_string_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """The streaming entry point builds its own turn input, so it has to resolve one as well."""
    monkeypatch.setenv("RUNA_DATABASE_URL", "memory://")
    captured: dict[str, Any] = {}

    async def fake_run(agent: Any, turn_input: Any, *, session: Any, emit: Any, **kwargs: Any):
        captured["session"] = session
        emit("event")
        return _loop_run(_transcript=Transcript.start([{"role": "user", "content": "hi"}]))

    monkeypatch.setattr("runa.agent._run_async", fake_run)

    async def _consume() -> None:
        async for _ in Researcher().run_streamed("hi", session="user-42"):
            pass

    asyncio.run(_consume())

    assert isinstance(captured["session"], DatabaseSession)
    assert isinstance(captured["session"].database.location, MemoryDatabase)  # type: ignore[attr-defined]
    assert captured["session"].session_id == "user-42"


def test_a_session_object_is_passed_through_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The fuller form still works: a `Session` object passes through untouched."""
    captured: dict[str, Any] = {}

    def fake_run_sync(agent: Any, turn_input: Any, *, session: Any, **kwargs: Any) -> Run:
        captured["session"] = session
        return _loop_run()

    monkeypatch.setattr("runa.agent._run_async", _async(fake_run_sync))

    session = db.at(f"sqlite:///{tmp_path / 'runa.db'}").session("s")
    Researcher().run_sync("hi", session=session)

    assert captured["session"] is session


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
    assert captured["kwargs"] == {
        "judge": "gpt-5.4",
        "threshold": 0.8,
        "thresholds": None,
        "concurrency": 8,
    }


def _refund_tool(calls: list[float]) -> FunctionTool:
    @tool(needs_approval=True)
    def refund(amount: float) -> str:
        """Refund `amount` dollars."""
        calls.append(amount)
        return f"refunded {amount}"

    return refund


def test_run_sync_pauses_for_approval_and_resumes_from_its_state() -> None:
    """An approval-gated call pauses the `Run`; the approved state resumes it to completion."""
    calls: list[float] = []

    class Support(Agent):
        name = "Support"
        instructions = "Refund when asked."
        tools = [_refund_tool(calls)]
        model = _ScriptedModel(
            [_tool_call_message("refund", '{"amount": 75}'), _final_message("refunded")]
        )

    agent = Support()
    run = agent.run_sync("refund $75")

    assert run.status == "paused"
    assert [i.name for i in run.interruptions] == ["refund"]
    assert calls == []
    assert agent.history == []

    state = run.to_state()
    state.approve(run.interruptions[0])
    resumed = agent.run_sync(state)

    assert resumed.status == "completed"
    assert resumed.output == "refunded"
    assert calls == [75]
    assert agent.history[0] == {"role": "user", "content": "refund $75"}
    assert agent.history[-1]["content"] == "refunded"


def test_to_payload_is_the_one_wire_shape_of_a_run() -> None:
    """`Run.to_payload()` owns what a run looks like as JSON, so no transport re-derives it.

    Every field is read off `Run` by attribute: a renamed or removed one is an `AttributeError`
    here rather than a `null` on the wire, which is how `runa serve` shipped a `tool_name` that
    `Interruption` never had.
    """

    class Support(Agent):
        name = "Support"
        instructions = "Refund when asked."
        tools = [_refund_tool([])]
        model = _ScriptedModel(
            [_tool_call_message("refund", '{"amount": 75}'), _final_message("refunded")]
        )

    agent = Support()
    paused = agent.run_sync("refund $75").to_payload()

    assert paused["status"] == "paused"
    assert paused["output"] is None
    assert paused["trace_id"]
    assert paused["usage"]["total_tokens"] == 2
    assert paused["interruptions"] == [
        {
            "name": "refund",
            "arguments": '{"amount": 75}',
            "call_id": "call_1",
            "agent": "Support",
        }
    ]
    assert "state" not in paused  # the approval ledger is the transport's to keep, not to send
    assert json.loads(json.dumps(paused)) == paused  # the whole payload is JSON, not just a dict


def test_to_payload_carries_the_output_once_completed() -> None:
    """A completed run has an answer and nothing left to resume."""

    class Echo(Agent):
        name = "Echo"
        instructions = "echo"
        model = _ScriptedModel([_final_message("hi")])

    payload = Echo().run_sync("hello").to_payload()

    assert payload["status"] == "completed"
    assert payload["output"] == "hi"
    assert payload["error"] is None
    assert payload["interruptions"] == []


def test_to_payload_renders_a_dataclass_output_type() -> None:
    """An `output_type` is rendered as JSON data, not `str()`-ed into prose."""

    @dataclass
    class Answer:
        city: str
        celsius: int

    class Weather(Agent):
        name = "Weather"
        instructions = "report"
        output_type = Answer
        model = _ScriptedModel([_final_message('{"city": "Paris", "celsius": 18}')])

    payload = Weather().run_sync("paris?").to_payload()

    assert payload["output"] == {"city": "Paris", "celsius": 18}


def test_to_payload_renders_the_other_output_types_as_data_too() -> None:
    """A Pydantic `output_type` and a `TypedDict` one are data on the wire, same as a dataclass."""

    class PydanticAnswer(BaseModel):
        city: str

    class TypedDictAnswer(TypedDict):
        city: str
        highlights: list[str]

    class Pydantic(Agent):
        name = "Pydantic"
        instructions = "report"
        output_type = PydanticAnswer
        model = _ScriptedModel([_final_message('{"city": "Paris"}')])

    class Typed(Agent):
        name = "Typed"
        instructions = "report"
        output_type = TypedDictAnswer
        model = _ScriptedModel([_final_message('{"city": "Paris", "highlights": ["Louvre"]}')])

    assert Pydantic().run_sync("paris?").to_payload()["output"] == {"city": "Paris"}
    assert Typed().run_sync("paris?").to_payload()["output"] == {
        "city": "Paris",
        "highlights": ["Louvre"],
    }


@pytest.mark.parametrize(("setting", "value"), [("context", {"user": "42"}), ("user_id", "u1")])
def test_resuming_with_a_context_or_a_user_is_refused(setting: str, value: Any) -> None:
    """A `RunState` carries the context and user of the run it paused, so a second is a mistake."""
    from runa.exceptions import UserError

    class Support(Agent):
        name = "Support"
        instructions = "Refund when asked."
        tools = [_refund_tool([])]
        model = _ScriptedModel(
            [_tool_call_message("refund", '{"amount": 75}'), _final_message("refunded")]
        )

    agent = Support()
    state = agent.run_sync("refund $75").to_state()

    with pytest.raises(UserError, match=setting):
        agent.run_sync(state, **{setting: value})


def test_to_state_on_a_run_that_did_not_pause_raises() -> None:
    """Only a paused `Run` has a state to resume."""
    from runa.exceptions import UserError

    class Echo(Agent):
        name = "Echo"
        instructions = "Echo."
        model = _ScriptedModel([_final_message("hi")])

    with pytest.raises(UserError, match="paused"):
        Echo().run_sync("hi").to_state()


def test_run_streamed_pauses_and_resumes_like_run() -> None:
    """A stream ends paused with `.run.interruptions`; streaming its state finishes the run."""
    calls: list[float] = []

    class _StreamedScript:
        def __init__(self, turns: list[list[StreamDelta]]) -> None:
            self._turns = turns

        async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:  # noqa: ANN002, ANN003
            raise AssertionError("only ever streamed")

        async def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[StreamDelta]:  # noqa: ANN002, ANN003
            for delta in self._turns.pop(0):
                yield delta

    class Support(Agent):
        name = "Support"
        instructions = "Refund when asked."
        tools = [_refund_tool(calls)]

    agent = Support()
    agent.model = _StreamedScript(
        [
            [
                StreamDelta(tool_call_index=0, tool_call_id="call_1", tool_call_name="refund"),
                StreamDelta(tool_call_index=0, tool_call_arguments='{"amount": 20}'),
            ],
            [StreamDelta(text="done")],
        ]
    )

    async def consume(message: Any) -> Any:
        stream = agent.run_streamed(message)
        async for _ in stream:
            pass
        return stream.run

    paused = asyncio.run(consume("refund $20"))
    assert paused.status == "paused"
    state = paused.to_state()
    state.approve(paused.interruptions[0])
    finished = asyncio.run(consume(state))

    assert finished.status == "completed"
    assert finished.output == "done"
    assert calls == [20]


def test_run_streamed_reports_an_error_as_an_error_run() -> None:
    """A `RunaError` mid-stream ends the stream with an `"error"` `Run`, like `run`."""

    class _AlwaysCallsATool:
        async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:  # noqa: ANN002, ANN003
            raise AssertionError("only ever streamed")

        async def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[StreamDelta]:  # noqa: ANN002, ANN003
            yield StreamDelta(tool_call_index=0, tool_call_id="call_1", tool_call_name="missing")

    class Echo(Agent):
        name = "Echo"
        instructions = "Echo."
        max_turns = 1

    agent = Echo()
    agent.model = _AlwaysCallsATool()

    async def consume() -> Any:
        stream = agent.run_streamed("hi")
        async for _ in stream:
            pass
        return stream.run

    run = asyncio.run(consume())

    assert run.status == "error"
    assert run.error is not None


class _Child(Agent):
    name = "Child"
    instructions = "Refund when asked."


def _parent_with_paused_delegate(calls: list[float]) -> Agent:
    """A parent whose `.delegate` child calls an approval-gated tool."""

    class Parent(Agent):
        name = "Parent"
        instructions = "Delegate refunds."
        subagents = [_Child.delegate]

    parent = Parent()
    parent.model = _ScriptedModel(
        [_tool_call_message("child", '{"input": "refund $5"}', "outer_1"), _final_message("ok")]
    )
    child = next(iter(built_shape(parent).delegates.values())).agent
    child.tools = [_refund_tool(calls)]
    child.model = _ScriptedModel(
        [_tool_call_message("refund", '{"amount": 5}', "inner_1"), _final_message("refunded 5")]
    )
    return parent


def test_a_delegate_that_pauses_pauses_its_caller_and_resumes_where_it_stopped() -> None:
    """The child's approval surfaces on the parent's `Run`; approving it finishes both runs."""
    calls: list[float] = []
    parent = _parent_with_paused_delegate(calls)

    run = parent.run_sync("please refund")

    assert run.status == "paused"
    assert [(i.name, i.agent.name) for i in run.interruptions] == [("refund", "Child")]

    state = run.to_state()
    state.approve(run.interruptions[0])
    resumed = parent.run_sync(state)

    assert resumed.status == "completed"
    assert resumed.output == "ok"
    assert calls == [5]
    assert parent.history[-2]["content"] == "refunded 5"  # the delegate's answer, as a tool result


def test_a_paused_delegate_survives_a_restart() -> None:
    """The stored pause carries the child's paused state, so a restart resumes it too."""
    calls: list[float] = []
    parent = _parent_with_paused_delegate(calls)
    run_id = parent.run_sync("please refund").run_id

    state = RunState.load_sync(parent, run_id or "")
    assert state is not None
    state.reject(state.pending[0], rejection_message="not today")
    resumed = parent.run_sync(state)

    assert resumed.status == "completed"
    assert calls == []


def _parent_with_a_paused_grandchild(calls: list[float]) -> Agent:
    """A parent delegating to a child that delegates to the agent holding the gated tool."""

    class Grandchild(Agent):
        name = "Grandchild"
        instructions = "Refund when asked."

    class Child(Agent):
        name = "Child"
        instructions = "Pass refunds down."
        subagents = [Grandchild.delegate]

    class Parent(Agent):
        name = "Parent"
        instructions = "Delegate refunds."
        subagents = [Child.delegate]

    parent = Parent()
    parent.model = _ScriptedModel(
        [_tool_call_message("child", '{"input": "refund $5"}', "outer_1"), _final_message("ok")]
    )
    child = next(iter(built_shape(parent).delegates.values())).agent
    child.model = _ScriptedModel(
        [
            _tool_call_message("grandchild", '{"input": "refund $5"}', "mid_1"),
            _final_message("passed on"),
        ]
    )
    grandchild = next(iter(built_shape(child).delegates.values())).agent
    grandchild.tools = [_refund_tool(calls)]
    grandchild.model = _ScriptedModel(
        [_tool_call_message("refund", '{"amount": 5}', "inner_1"), _final_message("refunded 5")]
    )
    return parent


def test_a_delegate_of_a_delegate_pauses_the_whole_chain_and_resumes_it() -> None:
    """Every depth pauses on the same call, and approving it on the outermost state runs it."""
    calls: list[float] = []
    parent = _parent_with_a_paused_grandchild(calls)

    run = parent.run_sync("please refund")

    assert run.status == "paused"
    assert [(i.name, i.agent.name) for i in run.interruptions] == [("refund", "Grandchild")]

    state = run.to_state()
    state.approve(run.interruptions[0])
    resumed = parent.run_sync(state)

    assert resumed.status == "completed"
    assert resumed.output == "ok"
    assert calls == [5]


def test_a_paused_delegate_of_a_delegate_survives_a_restart() -> None:
    """Ownership is stored, so a restart still resolves the call against the deepest run."""
    calls: list[float] = []
    parent = _parent_with_a_paused_grandchild(calls)
    run_id = parent.run_sync("please refund").run_id

    state = RunState.load_sync(parent, run_id or "")
    assert state is not None
    state.approve(state.pending[0])
    resumed = parent.run_sync(state)

    assert resumed.status == "completed"
    assert calls == [5]


def test_output_type_parses_the_final_answer() -> None:
    """A declared `output_type` turns the model's JSON answer into that type."""

    @dataclass
    class Weather:
        city: str
        temperature: int

    class Forecaster(Agent):
        name = "Forecaster"
        instructions = "Answer in JSON."
        output_type = Weather
        model = _ScriptedModel([_final_message('{"city": "Paris", "temperature": 21}')])

    run = Forecaster().run_sync("weather in Paris?")

    assert run.output == Weather(city="Paris", temperature=21)


def test_output_type_mismatch_is_an_error_run() -> None:
    """An answer that doesn't fit `output_type` is a model error, not a silently wrong value."""

    @dataclass
    class Weather:
        city: str
        temperature: int

    class Forecaster(Agent):
        name = "Forecaster"
        instructions = "Answer in JSON."
        output_type = Weather
        model = _ScriptedModel([_final_message("sunny, 21 degrees")])

    run = Forecaster().run_sync("weather in Paris?")

    assert run.status == "error"
    assert run.error is not None and "Weather" in run.error


def test_max_turns_is_a_class_attribute() -> None:
    """`max_turns` caps the run's model calls, set like any other agent attribute."""

    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    class Looper(Agent):
        name = "Looper"
        instructions = "Keep pinging."
        tools = [ping]
        max_turns = 2
        model = _ScriptedModel([_tool_call_message("ping", "{}", f"call_{i}") for i in range(3)])

    run = Looper().run_sync("go")

    assert run.status == "error"
    assert run.error == "max turns (2) exceeded"


@guardrail
def _never_trips(x: str) -> bool:
    """Let everything through."""
    return False


@guardrail
def _trips_on_empty(x: str) -> bool:
    """Trip on an empty message."""
    return not x.strip()


def test_a_completed_run_carries_every_guardrail_that_ran() -> None:
    """Passing agent and tool guardrails are all listed on `Run`, not just ones that trip."""

    @tool(guardrails=[_never_trips.input, _never_trips.output])
    def lookup(order_id: str) -> str:
        """Look up an order."""
        return "shipped"

    class Support(Agent):
        name = "Support"
        instructions = "Look orders up."
        tools = [lookup]
        guardrails = [_trips_on_empty.input, _never_trips.output]
        model = _ScriptedModel(
            [_tool_call_message("lookup", '{"order_id": "A1"}'), _final_message("shipped")]
        )

    run = Support().run_sync("where is A1?")

    assert run.status == "completed"
    assert [r.tripped for r in run.input_guardrail_results] == [False]
    assert [r.tripped for r in run.output_guardrail_results] == [False]
    assert [r.tripped for r in run.tool_input_guardrail_results] == [False]
    assert [r.tripped for r in run.tool_output_guardrail_results] == [False]


def test_an_error_run_carries_the_guardrail_that_stopped_it() -> None:
    """A tripped input guardrail ends the run with `status="error"` and shows up as tripped."""

    class Support(Agent):
        name = "Support"
        instructions = "Help."
        guardrails = [_trips_on_empty.input]
        model = _ScriptedModel([])

    run = Support().run_sync("   ")

    assert run.status == "error"
    assert [r.tripped for r in run.input_guardrail_results] == [True]


def test_a_delegates_guardrails_show_up_on_its_callers_run() -> None:
    """A delegate shares its caller's audit trail: its tool guardrails land on the caller's Run."""

    @tool(guardrails=[_never_trips.input])
    def lookup(order_id: str) -> str:
        """Look up an order."""
        return "shipped"

    class Parent(Agent):
        name = "Parent"
        instructions = "Delegate lookups."
        subagents = [_Child.delegate]

    parent = Parent()
    parent.model = _ScriptedModel(
        [_tool_call_message("child", '{"input": "find A1"}', "outer_1"), _final_message("ok")]
    )
    child = next(iter(built_shape(parent).delegates.values())).agent
    child.tools = [lookup]
    child.model = _ScriptedModel(
        [_tool_call_message("lookup", '{"order_id": "A1"}', "inner_1"), _final_message("shipped")]
    )

    run = parent.run_sync("where is A1?")

    assert run.status == "completed"
    assert [r.tripped for r in run.tool_input_guardrail_results] == [False]


def test_run_sync_rejects_a_transcript_as_the_message() -> None:
    """`message` is one turn: replaying past messages goes through `history`/`session` instead."""
    agent = Researcher()

    with pytest.raises(TypeError, match="does not take a list of past messages"):
        agent.run_sync([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}])

    assert agent.history == []
