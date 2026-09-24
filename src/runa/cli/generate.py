"""cli/generate.py: generate scaffolding inside an existing Runa app.

Reads structure, not configuration: a new agent goes to `app/agents/`
because that's the convention `runa new` established, not because anything
is configured to say so. `tool`, `guardrail`, and `evaluation` follow the
same pattern into `app/tools/`/`app/guardrails/`/`evals/`.

Every template here is self-contained and immediately importable: `runa
eval`/`runa test` succeed against a freshly generated file (0 cases, an
inert stub Agent/tool) the same way they do against an empty
`evals/`/`app/tests/`, rather than crashing until the developer fills in
the TODOs.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path

from runa.agent import _PROMPT_TEMPLATE
from runa.cli._project import NotARunaProject
from runa.cli.chat import AgentNotFound

_TOOL_IMPORT = "from runa import tool"

_TOOL_FUNCTION_TEMPLATE = '''@tool
def {func_name}() -> str:
    """{description}"""
    raise NotImplementedError
'''

_GUARDRAIL_TEMPLATE = '''from runa import guardrail


@guardrail
def {func_name}(value: str) -> bool:
    """TODO: describe what trips this guardrail."""
    raise NotImplementedError
'''

_EVALUATION_TEMPLATE = '{"input": "Hello! What can you help me with?"}\n'


class InvalidAgentName(Exception):
    """Raised when the given class name isn't UpperCamelCase ending in `Agent`."""


class AgentAlreadyExists(Exception):
    """Raised when the target agent file, or its `name` identity, already exists."""


class ToolAlreadyExists(Exception):
    """Raised when the target tool file already exists."""


class GuardrailAlreadyExists(Exception):
    """Raised when the target guardrail file already exists."""


class PromptAlreadyExists(Exception):
    """Raised when the target prompt file already exists."""


class EvaluationAlreadyExists(Exception):
    """Raised when the target evaluation file already exists."""


class AmbiguousComponent(Exception):
    """Raised when a `--tool`/`--guardrail` name matches more than one file."""


def _snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


_AGENT_CLASS_NAME = re.compile(r"[A-Z][A-Za-z0-9]*Agent")


def split_tool_name(raw_name: str) -> tuple[str, str]:
    """Split a `--tool`/`NAME` value into (module, function).

    `research:search_web` -> `("research", "search_web")`: the tool lands in (or is looked up
    from) `app/tools/research.py`. A bare `search_web` defaults its module to `core` --
    `app/tools/core.py`, the catch-all file every ungrouped tool lands in.
    """
    module, sep, func = raw_name.rpartition(":")
    return (_snake_case(module), _snake_case(func)) if sep else ("core", _snake_case(raw_name))


def _require_dir(root: Path, *parts: str) -> Path:
    target_dir = root.joinpath(*parts)
    if not target_dir.is_dir():
        raise NotARunaProject(
            f"{target_dir} does not exist, run this from inside a Runa "
            "project created with `runa new`"
        )
    return target_dir


def _prompt_yes_no(message: str) -> bool:
    """Ask on stdin; the default `confirm` for `generate_agent`'s `--tool`/`--guardrail`."""
    return input(f"{message} [y/N] ").strip().lower() in ("y", "yes")


def _find_component(base_dir: Path, snake_name: str) -> Path | None:
    """Look for a `def <snake_name>(` anywhere under `base_dir`, not just a same-named file.

    Developers are free to organize `app/tools/`/`app/guardrails/` into subpackages, and
    `app/tools/` files can each hold several `@tool` functions grouped by `split_tool_name`'s
    module (e.g. `research.py` holding both `search_web` and `fetch_page`), so a
    `--tool`/`--guardrail` name is resolved by scanning function definitions rather than
    assuming `<name>.py`.
    """
    pattern = re.compile(rf"(?m)^def {re.escape(snake_name)}\(")
    matches = sorted(p for p in base_dir.rglob("*.py") if pattern.search(p.read_text()))
    if len(matches) > 1:
        found = ", ".join(str(match.relative_to(base_dir)) for match in matches)
        raise AmbiguousComponent(f"'{snake_name}' matches more than one file: {found}")
    return matches[0] if matches else None


def _find_agent_name(agents_dir: Path, agent_name: str) -> Path | None:
    """Look for an existing `name = "<agent_name>"` anywhere under `agents_dir`.

    Catches an identity collision even when the offending file's name doesn't match
    `agent_name` (e.g. it was hand-edited after generation), not just the common case where
    the file path itself already collides.
    """
    pattern = re.compile(rf"""(?m)^\s*name\s*=\s*["']{re.escape(agent_name)}["']""")
    matches = sorted(p for p in agents_dir.rglob("*.py") if pattern.search(p.read_text()))
    return matches[0] if matches else None


def _module_path(root: Path, file: Path) -> str:
    return ".".join(file.relative_to(root).with_suffix("").parts)


def _export_agent(agents_dir: Path, file_stem: str, class_name: str) -> None:
    """Re-export `class_name` from `app/agents/__init__.py`.

    So it's reachable as `from app.agents import {class_name}` instead of
    `from app.agents.{file_stem} import ...`.
    """
    init_file = agents_dir / "__init__.py"
    existing = init_file.read_text() if init_file.exists() else ""
    lines = {line for line in existing.splitlines() if line.strip()}
    lines.add(f"from .{file_stem} import {class_name}")
    init_file.write_text("\n".join(sorted(lines)) + "\n")


def _resolve_components(
    names: Sequence[str],
    *,
    root: Path,
    base_dir: Path,
    kind: str,
    generate: Callable[..., Path],
    confirm: Callable[[str], bool],
) -> tuple[list[str], list[str]]:
    """Resolve `--tool`/`--guardrail` names to (import lines, bare references).

    A missing name prompts for approval to scaffold it via `generate` (the same
    `generate_tool`/`generate_guardrail` a standalone `runa generate tool`/`guardrail` would run)
    before the agent file is written. Declined, it's still imported and referenced: the agent
    file fails loudly on import instead of silently dropping it from `tools`/`guardrails`.

    A `--tool` name may carry a `module:function` prefix (see `split_tool_name`); only the
    function part is used for lookup, import, and the `tools = [...]` reference.
    """
    imports: list[str] = []
    refs: list[str] = []
    for raw_name in names:
        snake_name = _snake_case(raw_name.rpartition(":")[-1])
        existing = _find_component(base_dir, snake_name)
        if existing is None:
            relative_dir = base_dir.relative_to(root)
            if confirm(f"{kind} '{snake_name}' not found in {relative_dir}/, create it?"):
                existing = generate(raw_name, root=root)
            else:
                module_name = split_tool_name(raw_name)[0] if kind == "tool" else snake_name
                existing = base_dir / f"{module_name}.py"
        imports.append(f"from {_module_path(root, existing)} import {snake_name}")
        refs.append(snake_name)
    return imports, refs


def _render_agent_source(
    *,
    class_name: str,
    name: str,
    model: str | None,
    instructions: str | None,
    tool_imports: list[str],
    tool_refs: list[str],
    guardrail_imports: list[str],
    guardrail_refs: list[str],
    memory: str | None,
    knowledge: str | None,
    compact: bool,
) -> str:
    imports = ["from runa import Agent", *sorted(tool_imports), *sorted(guardrail_imports)]

    lines = [f'    name = "{name}"']
    if instructions is not None:
        lines.append(f"    instructions = {instructions!r}")
    if model is not None:
        lines.append(f'    model = "{model}"')
    if tool_refs:
        lines.append(f"    tools = [{', '.join(tool_refs)}]")
    if guardrail_refs:
        lines.append(f"    guardrails = [{', '.join(guardrail_refs)}]")
    if memory is not None:
        lines.append(f'    memory = "{memory}"')
    if knowledge is not None:
        lines.append(f'    knowledge = "{knowledge}"')
    if compact:
        lines.append("    compact = True")

    return "\n".join(imports) + f"\n\n\nclass {class_name}(Agent):\n" + "\n".join(lines) + "\n"


def generate_agent(
    name: str,
    *,
    root: Path,
    model: str | None = None,
    instructions: str | None = None,
    tools: Sequence[str] = (),
    guardrails: Sequence[str] = (),
    memory: str | None = None,
    knowledge: str | None = None,
    compact: bool = False,
    confirm: Callable[[str], bool] = _prompt_yes_no,
) -> Path:
    """Write a new Agent subclass into `root/app/agents/`.

    `name` must be an UpperCamelCase class name ending in `Agent` (e.g. `SupportAgent`), the
    one naming convention this command enforces, so it's the single source of truth for an
    agent's identity. Anything else raises `InvalidAgentName`.

    Both the generated file and the class's declared `name` attribute are derived from it via
    snake_case (`SupportAgent` -> `app/agents/support_agent.py`, `name = "support_agent"`).
    There's no separate identity to pass in, and nothing that can drift out of sync with the
    class name. That derived identity must also be unique across `app/agents/`: generating a
    second agent that derives a `name` already declared elsewhere raises `AgentAlreadyExists`,
    since `runa chat <name>` couldn't tell the two apart, the same error a plain filename
    collision already raises.

    `tools`/`guardrails` are names looked up under `app/tools/`/`app/guardrails/` (searched
    recursively); a name that isn't found there prompts (via `confirm`, real `input()` by
    default) for approval to scaffold it before the agent file is written. A `--tool` name may
    carry a `module:function` prefix (see `split_tool_name`) to place a new tool outside the
    default `app/tools/core.py`.

    Without an explicit `instructions`, the class gets no `instructions` attribute at all.
    Instead a stub `app/prompts/<snake_case(name)>.md` is written alongside it, the same
    file `Agent.__init__` would lazily create on first instantiation (`agent.py`'s
    `_load_prompt`). Generating it upfront means it's there to edit before the first `runa chat`.
    Likewise `evals/<snake_case(name)>.jsonl` starts with one case, so `runa eval` grades the new
    agent from day one (see `generate_evaluation`).

    Also appends `from .{file_stem} import {class_name}` to `app/agents/__init__.py`, so the
    agent is reachable as `from app.agents import {class_name}` rather than reaching into its
    own submodule.
    """
    if not _AGENT_CLASS_NAME.fullmatch(name):
        raise InvalidAgentName(
            f"'{name}' is not a valid Agent class name, use UpperCamelCase ending in "
            "'Agent', e.g. SupportAgent"
        )

    agents_dir = _require_dir(root, "app", "agents")

    class_name = name
    file_stem = _snake_case(class_name)
    agent_name = file_stem
    agent_file = agents_dir / f"{file_stem}.py"
    if agent_file.exists():
        raise AgentAlreadyExists(f"{agent_file} already exists")

    duplicate = _find_agent_name(agents_dir, agent_name)
    if duplicate is not None:
        raise AgentAlreadyExists(f"an agent named '{agent_name}' already exists in {duplicate}")

    tool_imports, tool_refs = _resolve_components(
        tools,
        root=root,
        base_dir=_require_dir(root, "app", "tools"),
        kind="tool",
        generate=generate_tool,
        confirm=confirm,
    )
    guardrail_imports, guardrail_refs = _resolve_components(
        guardrails,
        root=root,
        base_dir=_require_dir(root, "app", "guardrails"),
        kind="guardrail",
        generate=generate_guardrail,
        confirm=confirm,
    )

    agent_file.write_text(
        _render_agent_source(
            class_name=class_name,
            name=agent_name,
            model=model,
            instructions=instructions,
            tool_imports=tool_imports,
            tool_refs=tool_refs,
            guardrail_imports=guardrail_imports,
            guardrail_refs=guardrail_refs,
            memory=memory,
            knowledge=knowledge,
            compact=compact,
        )
    )

    if instructions is None:
        prompts_dir = _require_dir(root, "app", "prompts")
        prompt_file = prompts_dir / f"{file_stem}.md"
        if not prompt_file.exists():
            prompt_file.write_text(_PROMPT_TEMPLATE.format(name=file_stem))

    eval_file = root / "evals" / f"{file_stem}.jsonl"
    if eval_file.parent.is_dir() and not eval_file.exists():
        eval_file.write_text(_EVALUATION_TEMPLATE)

    _export_agent(agents_dir, file_stem, class_name)

    return agent_file


def generate_tool(name: str, *, root: Path, description: str | None = None) -> Path:
    """Write a new `@tool`-decorated function into `root/app/tools/`.

    `name` is `module:function` (e.g. `research:search_web`, landing in `app/tools/research.py`)
    or a bare function name, which defaults its module to `core` -- `app/tools/core.py`, the
    catch-all every ungrouped tool lands in (see `split_tool_name`). A second tool sharing a
    module (e.g. `research:fetch_page` after `research:search_web`) appends alongside it in the
    same file rather than raising; only a duplicate function name in that module does.

    `description` becomes the function's docstring, the same one-liner `@tool` reads to
    describe the tool to the model; omitted, it's a `TODO` stub like every other template here.
    """
    tools_dir = _require_dir(root, "app", "tools")

    module_name, func_name = split_tool_name(name)
    tool_file = tools_dir / f"{module_name}.py"
    existing_source = tool_file.read_text() if tool_file.exists() else ""
    if re.search(rf"(?m)^def {re.escape(func_name)}\(", existing_source):
        raise ToolAlreadyExists(f"'{func_name}' already exists in {tool_file}")

    function_source = _TOOL_FUNCTION_TEMPLATE.format(
        func_name=func_name,
        description=description or "TODO: describe what this tool does.",
    )
    header = existing_source.rstrip("\n") if existing_source else _TOOL_IMPORT
    tool_file.write_text(f"{header}\n\n\n{function_source}")
    return tool_file


def generate_guardrail(name: str, *, root: Path) -> Path:
    """Write a new `@guardrail`-decorated function into `root/app/guardrails/`."""
    guardrails_dir = _require_dir(root, "app", "guardrails")

    func_name = _snake_case(name)
    guardrail_file = guardrails_dir / f"{func_name}.py"
    if guardrail_file.exists():
        raise GuardrailAlreadyExists(f"{guardrail_file} already exists")

    guardrail_file.write_text(_GUARDRAIL_TEMPLATE.format(func_name=func_name))
    return guardrail_file


def generate_prompt(name: str, *, root: Path) -> Path:
    """Write a new prompt file into `root/app/prompts/`.

    Plain markdown, not Python: a prompt is text an agent's `instructions` can load, kept out
    of source the same way a query lives outside application code.
    """
    prompts_dir = _require_dir(root, "app", "prompts")

    file_stem = _snake_case(name)
    prompt_file = prompts_dir / f"{file_stem}.md"
    if prompt_file.exists():
        raise PromptAlreadyExists(f"{prompt_file} already exists")

    prompt_file.write_text(_PROMPT_TEMPLATE.format(name=file_stem))
    return prompt_file


def generate_evaluation(name: str, *, root: Path) -> Path:
    """Write a new eval dataset into `root/evals/<name>.jsonl`.

    `generate_agent` already writes this file for every agent it creates, so this is for an
    agent written by hand, or to start over after deleting the file.

    `name` is the agent's snake_case identity, the same one `runa chat <name>` takes (e.g.
    `support_agent`), not the class name: `runa eval` resolves the Agent from the filename
    (see `cli/eval.py`), so the file needs nothing but cases, one JSON object per line. It
    starts with a single input-only case, graded on task completion and answer relevance. The
    Agent must already exist, checked by scanning source rather than importing the app.
    """
    evals_dir = _require_dir(root, "evals")

    file_stem = _snake_case(name)
    if _find_agent_name(_require_dir(root, "app", "agents"), file_stem) is None:
        raise AgentNotFound(f"no Agent named {file_stem!r} found under app/agents/")
    eval_file = evals_dir / f"{file_stem}.jsonl"
    if eval_file.exists():
        raise EvaluationAlreadyExists(f"{eval_file} already exists")

    eval_file.write_text(_EVALUATION_TEMPLATE)
    return eval_file
