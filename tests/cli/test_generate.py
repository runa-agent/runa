"""Tests for `runa.cli.generate`."""

from pathlib import Path

import pytest

from runa.cli._project import NotARunaProject
from runa.cli.generate import (
    AgentAlreadyExists,
    AmbiguousComponent,
    EvaluationAlreadyExists,
    GuardrailAlreadyExists,
    InvalidAgentName,
    PromptAlreadyExists,
    ToolAlreadyExists,
    generate_agent,
    generate_evaluation,
    generate_guardrail,
    generate_prompt,
    generate_tool,
    split_tool_name,
)
from runa.cli.new import scaffold_project
from runa.eval import Dataset


def test_generate_agent_writes_a_runa_agent_subclass(tmp_path: Path) -> None:
    """`generate_agent` derives both the file and the `name` attribute from the class name."""
    project_dir = scaffold_project("demo", root=tmp_path)

    agent_file = generate_agent("SupportAgent", root=project_dir)

    assert agent_file == project_dir / "app" / "agents" / "support_agent.py"
    content = agent_file.read_text()
    assert "class SupportAgent(Agent):" in content
    assert 'name = "support_agent"' in content


def test_generate_agent_writes_a_prompt_stub_when_instructions_omitted(tmp_path: Path) -> None:
    """Without `instructions=`, a stub `app/prompts/<name>.md` is created alongside it."""
    project_dir = scaffold_project("demo", root=tmp_path)

    agent_file = generate_agent("SupportAgent", root=project_dir)

    assert "instructions" not in agent_file.read_text()
    prompt_file = project_dir / "app" / "prompts" / "support_agent.md"
    assert prompt_file.is_file()
    assert "TODO: write the prompt support_agent uses." in prompt_file.read_text()


def test_generate_agent_with_explicit_instructions_skips_the_prompt_stub(tmp_path: Path) -> None:
    """Passing `instructions=` inlines it and leaves `app/prompts/` untouched."""
    project_dir = scaffold_project("demo", root=tmp_path)

    generate_agent(
        "SupportAgent",
        root=project_dir,
        instructions="Help users with account questions",
    )

    assert not (project_dir / "app" / "prompts" / "support_agent.md").exists()


def test_generate_agent_rejects_a_class_name_not_ending_in_agent(tmp_path: Path) -> None:
    """`Support` isn't a valid class name -- it must end in `Agent`."""
    project_dir = scaffold_project("demo", root=tmp_path)

    with pytest.raises(InvalidAgentName):
        generate_agent("Support", root=project_dir)


def test_generate_agent_rejects_a_lowercase_or_snake_case_name(tmp_path: Path) -> None:
    """The class name must be UpperCamelCase, not `support_agent` or `supportAgent`."""
    project_dir = scaffold_project("demo", root=tmp_path)

    with pytest.raises(InvalidAgentName):
        generate_agent("support_agent", root=project_dir)


def test_generate_agent_rejects_bare_agent_with_no_prefix(tmp_path: Path) -> None:
    """`Agent` alone has no `<Name>` before the suffix, so it's rejected too."""
    project_dir = scaffold_project("demo", root=tmp_path)

    with pytest.raises(InvalidAgentName):
        generate_agent("Agent", root=project_dir)


def test_generate_agent_raises_if_the_file_already_exists(tmp_path: Path) -> None:
    """`generate_agent` refuses to overwrite an existing agent file."""
    project_dir = scaffold_project("demo", root=tmp_path)
    generate_agent("SupportAgent", root=project_dir)

    with pytest.raises(AgentAlreadyExists):
        generate_agent("SupportAgent", root=project_dir)


def test_generate_agent_raises_if_the_derived_name_already_exists(tmp_path: Path) -> None:
    """A derived identity already declared elsewhere is a conflict, even from a different file."""
    project_dir = scaffold_project("demo", root=tmp_path)
    (project_dir / "app" / "agents" / "legacy.py").write_text(
        'from runa import Agent\n\n\nclass Legacy(Agent):\n    name = "support_agent"\n'
    )

    with pytest.raises(AgentAlreadyExists):
        generate_agent("SupportAgent", root=project_dir)


def test_generate_agent_exports_it_from_the_agents_package(tmp_path: Path) -> None:
    """The class is re-exported from `app/agents/__init__.py`, reachable as a package import."""
    project_dir = scaffold_project("demo", root=tmp_path)

    generate_agent("SupportAgent", root=project_dir)

    init_content = (project_dir / "app" / "agents" / "__init__.py").read_text()
    assert init_content == "from .support_agent import SupportAgent\n"


def test_generate_agent_appends_to_existing_exports(tmp_path: Path) -> None:
    """A second agent's export is appended, not overwriting the first."""
    project_dir = scaffold_project("demo", root=tmp_path)
    generate_agent("SupportAgent", root=project_dir)

    generate_agent("BillingAgent", root=project_dir)

    init_content = (project_dir / "app" / "agents" / "__init__.py").read_text()
    assert "from .support_agent import SupportAgent" in init_content
    assert "from .billing_agent import BillingAgent" in init_content


def test_generate_agent_raises_outside_a_runa_project(tmp_path: Path) -> None:
    """`generate_agent` refuses to run where `app/agents/` doesn't exist."""
    with pytest.raises(NotARunaProject):
        generate_agent("SupportAgent", root=tmp_path)


def test_generate_agent_writes_model_instructions_memory_knowledge_and_compact(
    tmp_path: Path,
) -> None:
    """Every scalar `--flag` lands as the matching class attribute."""
    project_dir = scaffold_project("demo", root=tmp_path)

    agent_file = generate_agent(
        "SupportAgent",
        root=project_dir,
        model="claude-sonnet-5",
        instructions="Help users with account questions",
        memory="auto",
        knowledge="llm",
        compact=True,
    )

    content = agent_file.read_text()
    assert 'model = "claude-sonnet-5"' in content
    assert "instructions = 'Help users with account questions'" in content
    assert 'memory = "auto"' in content
    assert 'knowledge = "llm"' in content
    assert "compact = True" in content


def test_generate_agent_references_an_existing_nested_tool(tmp_path: Path) -> None:
    """`--tool` finds a match anywhere under `app/tools/`, not just its top level."""
    project_dir = scaffold_project("demo", root=tmp_path)
    nested_dir = project_dir / "app" / "tools" / "email"
    nested_dir.mkdir()
    (nested_dir / "send_email.py").write_text("def send_email() -> str: ...\n")

    agent_file = generate_agent(
        "SupportAgent",
        root=project_dir,
        tools=["send_email"],
        confirm=lambda _msg: False,
    )

    content = agent_file.read_text()
    assert "from app.tools.email.send_email import send_email" in content
    assert "tools = [send_email]" in content


def test_generate_agent_raises_on_an_ambiguous_tool_name(tmp_path: Path) -> None:
    """Two files matching the same `--tool` name is a hard error, not a silent guess."""
    project_dir = scaffold_project("demo", root=tmp_path)
    tools_dir = project_dir / "app" / "tools"
    (tools_dir / "search.py").write_text("def search() -> str: ...\n")
    nested_dir = tools_dir / "web"
    nested_dir.mkdir()
    (nested_dir / "search.py").write_text("def search() -> str: ...\n")

    with pytest.raises(AmbiguousComponent):
        generate_agent(
            "SupportAgent",
            root=project_dir,
            tools=["search"],
            confirm=lambda _msg: False,
        )


def test_generate_agent_scaffolds_a_missing_tool_when_confirmed(tmp_path: Path) -> None:
    """A missing `--tool` name is created via `generate_tool` when `confirm` returns `True`."""
    project_dir = scaffold_project("demo", root=tmp_path)

    agent_file = generate_agent(
        "SupportAgent",
        root=project_dir,
        tools=["search_web"],
        confirm=lambda _msg: True,
    )

    assert (project_dir / "app" / "tools" / "core.py").is_file()
    content = agent_file.read_text()
    assert "from app.tools.core import search_web" in content
    assert "tools = [search_web]" in content


def test_generate_agent_scaffolds_a_missing_grouped_tool_when_confirmed(tmp_path: Path) -> None:
    """A `module:function` `--tool` name creates and imports from that module file."""
    project_dir = scaffold_project("demo", root=tmp_path)

    agent_file = generate_agent(
        "SupportAgent",
        root=project_dir,
        tools=["research:search_web"],
        confirm=lambda _msg: True,
    )

    assert (project_dir / "app" / "tools" / "research.py").is_file()
    content = agent_file.read_text()
    assert "from app.tools.research import search_web" in content
    assert "tools = [search_web]" in content


def test_generate_agent_still_references_a_declined_missing_tool(tmp_path: Path) -> None:
    """Declining creation still wires the reference, so it fails loudly rather than vanishing."""
    project_dir = scaffold_project("demo", root=tmp_path)

    agent_file = generate_agent(
        "SupportAgent",
        root=project_dir,
        tools=["search_web"],
        confirm=lambda _msg: False,
    )

    assert not (project_dir / "app" / "tools" / "core.py").exists()
    content = agent_file.read_text()
    assert "from app.tools.core import search_web" in content
    assert "tools = [search_web]" in content


def test_generate_agent_scaffolds_a_missing_guardrail_when_confirmed(tmp_path: Path) -> None:
    """A missing `--guardrail` name is created via `generate_guardrail` when confirmed."""
    project_dir = scaffold_project("demo", root=tmp_path)

    agent_file = generate_agent(
        "SupportAgent",
        root=project_dir,
        guardrails=["no_profanity"],
        confirm=lambda _msg: True,
    )

    assert (project_dir / "app" / "guardrails" / "no_profanity.py").is_file()
    content = agent_file.read_text()
    assert "from app.guardrails.no_profanity import no_profanity" in content
    assert "guardrails = [no_profanity]" in content


def test_generate_agent_prompt_message_names_the_missing_component(tmp_path: Path) -> None:
    """The `confirm` callback is given a message naming the kind, name, and search directory."""
    project_dir = scaffold_project("demo", root=tmp_path)
    seen: list[str] = []

    def confirm(message: str) -> bool:
        seen.append(message)
        return False

    generate_agent(
        "SupportAgent",
        root=project_dir,
        tools=["search_web"],
        confirm=confirm,
    )

    assert seen == ["tool 'search_web' not found in app/tools/, create it?"]


def test_generate_tool_writes_a_snake_case_tool_function(tmp_path: Path) -> None:
    """`generate_tool` writes an `@tool`-decorated function, not a `Tool` subclass."""
    project_dir = scaffold_project("demo", root=tmp_path)

    tool_file = generate_tool("SendEmail", root=project_dir)

    assert tool_file == project_dir / "app" / "tools" / "core.py"
    content = tool_file.read_text()
    assert "@tool" in content
    assert "def send_email() -> str:" in content


def test_generate_tool_without_a_module_prefix_defaults_to_core(tmp_path: Path) -> None:
    """A bare tool name (no `module:` prefix) lands in the catch-all `app/tools/core.py`."""
    project_dir = scaffold_project("demo", root=tmp_path)

    tool_file = generate_tool("search_web", root=project_dir)

    assert tool_file == project_dir / "app" / "tools" / "core.py"


def test_generate_tool_with_a_module_prefix_writes_to_that_module_file(tmp_path: Path) -> None:
    """`module:function` writes into `app/tools/<module>.py` instead of `core.py`."""
    project_dir = scaffold_project("demo", root=tmp_path)

    tool_file = generate_tool("research:search_web", root=project_dir)

    assert tool_file == project_dir / "app" / "tools" / "research.py"
    content = tool_file.read_text()
    assert "def search_web() -> str:" in content


def test_generate_tool_appends_a_second_tool_to_the_same_module(tmp_path: Path) -> None:
    """A second `module:function` sharing a module appends alongside the first, not overwrite."""
    project_dir = scaffold_project("demo", root=tmp_path)
    generate_tool("research:search_web", root=project_dir)

    tool_file = generate_tool("research:fetch_page", root=project_dir)

    content = tool_file.read_text()
    assert content.count("from runa import tool") == 1
    assert "def search_web() -> str:" in content
    assert "def fetch_page() -> str:" in content


def test_generate_tool_with_a_description_uses_it_as_the_docstring(tmp_path: Path) -> None:
    """`description=` lands as the function's docstring instead of the default TODO."""
    project_dir = scaffold_project("demo", root=tmp_path)

    tool_file = generate_tool(
        "SendEmail", root=project_dir, description="Send an email to the given address."
    )

    content = tool_file.read_text()
    assert '"""Send an email to the given address."""' in content
    assert "TODO" not in content


def test_generate_tool_raises_if_the_file_already_exists(tmp_path: Path) -> None:
    """`generate_tool` refuses to redefine a function that already exists in its module."""
    project_dir = scaffold_project("demo", root=tmp_path)
    generate_tool("search", root=project_dir)

    with pytest.raises(ToolAlreadyExists):
        generate_tool("search", root=project_dir)


def test_generate_tool_raises_if_the_function_exists_in_a_different_module(
    tmp_path: Path,
) -> None:
    """The same function name is still a conflict even when explicitly re-targeted."""
    project_dir = scaffold_project("demo", root=tmp_path)
    generate_tool("research:search_web", root=project_dir)

    with pytest.raises(ToolAlreadyExists):
        generate_tool("research:search_web", root=project_dir)


def test_split_tool_name_defaults_the_module_to_core() -> None:
    """A bare name has no `module:` prefix, so it defaults to `core`."""
    assert split_tool_name("search_web") == ("core", "search_web")


def test_split_tool_name_splits_module_and_function() -> None:
    """`module:function` splits into its two snake_case parts."""
    assert split_tool_name("research:search_web") == ("research", "search_web")


def test_generate_guardrail_writes_a_snake_case_guardrail_function(tmp_path: Path) -> None:
    """`generate_guardrail` writes an `@guardrail`-decorated function under `app/guardrails/`."""
    project_dir = scaffold_project("demo", root=tmp_path)

    guardrail_file = generate_guardrail("BlockEmpty", root=project_dir)

    assert guardrail_file == project_dir / "app" / "guardrails" / "block_empty.py"
    content = guardrail_file.read_text()
    assert "@guardrail" in content
    assert "def block_empty(value: str) -> bool:" in content


def test_generate_guardrail_raises_if_the_file_already_exists(tmp_path: Path) -> None:
    """`generate_guardrail` refuses to overwrite an existing guardrail file."""
    project_dir = scaffold_project("demo", root=tmp_path)
    generate_guardrail("block_empty", root=project_dir)

    with pytest.raises(GuardrailAlreadyExists):
        generate_guardrail("block_empty", root=project_dir)


def test_generate_guardrail_raises_outside_a_runa_project(tmp_path: Path) -> None:
    """`generate_guardrail` refuses to run where `app/guardrails/` doesn't exist."""
    with pytest.raises(NotARunaProject):
        generate_guardrail("block_empty", root=tmp_path)


def test_generate_prompt_writes_a_markdown_file(tmp_path: Path) -> None:
    """`generate_prompt` writes a `.md` file under `app/prompts/`, snake_cased like a tool."""
    project_dir = scaffold_project("demo", root=tmp_path)

    prompt_file = generate_prompt("MyAgent", root=project_dir)

    assert prompt_file == project_dir / "app" / "prompts" / "my_agent.md"
    assert "my_agent" in prompt_file.read_text()


def test_generate_prompt_raises_if_the_file_already_exists(tmp_path: Path) -> None:
    """`generate_prompt` refuses to overwrite an existing prompt file."""
    project_dir = scaffold_project("demo", root=tmp_path)
    generate_prompt("MyAgent", root=project_dir)

    with pytest.raises(PromptAlreadyExists):
        generate_prompt("MyAgent", root=project_dir)


def test_generate_prompt_raises_outside_a_runa_project(tmp_path: Path) -> None:
    """`generate_prompt` refuses to run where `app/prompts/` doesn't exist."""
    with pytest.raises(NotARunaProject):
        generate_prompt("MyAgent", root=tmp_path)


def test_generate_evaluation_writes_a_jsonl_dataset_named_after_the_agent(
    tmp_path: Path,
) -> None:
    """`name` is the agent's `runa chat`-style identity, and becomes `evals/<name>.jsonl`."""
    project_dir = scaffold_project("demo", root=tmp_path)

    eval_file = generate_evaluation("greeter_agent", root=project_dir)

    assert eval_file == project_dir / "evals" / "greeter_agent.jsonl"
    assert [case.input for case in Dataset.from_jsonl(eval_file)]


def test_generate_evaluation_raises_if_the_file_already_exists(tmp_path: Path) -> None:
    """`generate_evaluation` refuses to overwrite an existing evaluation module."""
    project_dir = scaffold_project("demo", root=tmp_path)
    generate_evaluation("Support", root=project_dir)

    with pytest.raises(EvaluationAlreadyExists):
        generate_evaluation("Support", root=project_dir)
