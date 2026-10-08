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

Writing one of those files is a single recipe, `scaffold`, which the
`generate_*` functions parameterize; `generate_agent` reaches its
companion prompt and dataset by calling their own generators, so where
each convention lives is stated once. Each one returns a `Generated`
rather than a bare path, so that stays true of the import line too: `runa
generate`'s next-step instructions are printed from what the generator
reports, not rebuilt from the filename it returned.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from runa.exceptions import OperatorError
from runa.project import AgentNotFound, NotARunaProject

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

_PROMPT_TEMPLATE = """TODO: write the prompt {name} uses.
"""

_EVALUATION_TEMPLATE = '{"input": "Hello! What can you help me with?"}\n'


class InvalidAgentName(OperatorError):
    """Raised when the given class name isn't UpperCamelCase ending in `Agent`."""


class ScaffoldExists(OperatorError):
    """Raised when what a `runa generate` command would write is already there.

    One type for every kind, because the message already names the file (or the colliding
    `name` identity) and every surface reports all of them the same way. Which kind of thing
    collided is in the path, not in the class.
    """


class AmbiguousComponent(OperatorError):
    """Raised when a `--tool`/`--guardrail` name matches more than one file."""


@dataclass(frozen=True, kw_only=True)
class Generated:
    """What a `generate_*` wrote, and how application code reaches it.

    Where a generated file goes and what it's imported as are one decision, so a caller is
    handed the import line instead of assembling it out of `file` -- move `app/tools/` and
    every surface that prints an import follows, rather than printing a line that no longer
    resolves while the file itself is still written correctly.

    `symbol`/`import_line` are `None` for a file that isn't Python: a prompt, an eval dataset.
    """

    file: Path
    symbol: str | None = None
    import_line: str | None = None


@dataclass(frozen=True, kw_only=True)
class GeneratedAgent(Generated):
    """A generated agent, plus the companion files written alongside it.

    `name` is the identity the class declares, the one `runa chat <name>` takes; `symbol` is its
    class name. `prompt` is `None` exactly when `instructions` was inlined, so no stub was
    written -- a caller naming the prompt file doesn't re-derive that from its own flags.
    """

    name: str
    prompt: Path | None
    dataset: Path


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


def scaffold(
    root: Path,
    *parts: str,
    stem: str,
    suffix: str,
    template: str,
    exist_ok: bool = False,
) -> Path:
    """Write `template` to `root/*parts/<stem><suffix>`, and return that path.

    The one recipe every `runa generate` command follows: find the conventional directory,
    refuse to overwrite, write the template. The four things that vary are the directory, the
    filename, the template, and whether a file already being there is an error.

    `exist_ok=True` makes the call a companion stub rather than a command: an existing file is
    left as it is, and a missing directory is skipped rather than raised. Either way the path
    is returned, so a caller that only wants the file to exist afterwards doesn't branch.
    That's what `generate_agent` wants from `generate_prompt`/`generate_evaluation`, where the
    agent is the thing being generated and the prompt and dataset come along with it.

    `generate_tool` is the one generator that doesn't come through here: it appends to a module
    shared by every tool in its group, so its write isn't "create this file" (see
    `split_tool_name`).
    """
    target_dir = root.joinpath(*parts)
    file = target_dir / f"{stem}{suffix}"
    if not target_dir.is_dir():
        if exist_ok:
            return file
        raise NotARunaProject(
            f"{target_dir} does not exist, run this from inside a Runa "
            "project created with `runa new`"
        )
    if file.exists():
        if exist_ok:
            return file
        raise ScaffoldExists(f"{file} already exists")
    file.write_text(template)
    return file


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


def _import_line(root: Path, file: Path, symbol: str) -> str:
    """How application code imports `symbol` out of a generated `file`.

    Stated once, for both the `tools = [...]`/`guardrails = [...]` imports written into a new
    agent file and the import line `runa generate` prints as a next step. `file` may be a
    directory, for a symbol re-exported from a package's `__init__.py` (see `_export_agent`).
    """
    return f"from {_module_path(root, file)} import {symbol}"


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
    generate: Callable[..., Generated],
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
                existing = generate(raw_name, root=root).file
            else:
                module_name = split_tool_name(raw_name)[0] if kind == "tool" else snake_name
                existing = base_dir / f"{module_name}.py"
        imports.append(_import_line(root, existing, snake_name))
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
) -> GeneratedAgent:
    """Write a new Agent subclass into `root/app/agents/`.

    `name` must be an UpperCamelCase class name ending in `Agent` (e.g. `SupportAgent`), the
    one naming convention this command enforces, so it's the single source of truth for an
    agent's identity. Anything else raises `InvalidAgentName`.

    Both the generated file and the class's declared `name` attribute are derived from it via
    snake_case (`SupportAgent` -> `app/agents/support_agent.py`, `name = "support_agent"`).
    There's no separate identity to pass in, and nothing that can drift out of sync with the
    class name. That derived identity must also be unique across `app/agents/`: generating a
    second agent that derives a `name` already declared elsewhere raises `ScaffoldExists`,
    since `runa chat <name>` couldn't tell the two apart, the same error a plain filename
    collision already raises.

    `tools`/`guardrails` are names looked up under `app/tools/`/`app/guardrails/` (searched
    recursively); a name that isn't found there prompts (via `confirm`, real `input()` by
    default) for approval to scaffold it before the agent file is written. A `--tool` name may
    carry a `module:function` prefix (see `split_tool_name`) to place a new tool outside the
    default `app/tools/core.py`.

    Without an explicit `instructions`, the class gets no `instructions` attribute at all.
    Instead a stub `app/prompts/<snake_case(name)>.md` is written alongside it, the file
    `Agent.__init__` reads on construction (`agent.py`'s `_load_prompt`, which only ever reads).
    Scaffolding it here is what keeps an agent from starting life on empty instructions.
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
        raise ScaffoldExists(f"{agent_file} already exists")

    duplicate = _find_agent_name(agents_dir, agent_name)
    if duplicate is not None:
        raise ScaffoldExists(f"an agent named '{agent_name}' already exists in {duplicate}")

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

    prompt = None
    if instructions is None:
        prompt = generate_prompt(file_stem, root=root, exist_ok=True).file
    dataset = generate_evaluation(file_stem, root=root, exist_ok=True).file

    _export_agent(agents_dir, file_stem, class_name)

    return GeneratedAgent(
        file=agent_file,
        symbol=class_name,
        import_line=_import_line(root, agents_dir, class_name),
        name=agent_name,
        prompt=prompt,
        dataset=dataset,
    )


def generate_tool(name: str, *, root: Path, description: str | None = None) -> Generated:
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
        raise ScaffoldExists(f"'{func_name}' already exists in {tool_file}")

    function_source = _TOOL_FUNCTION_TEMPLATE.format(
        func_name=func_name,
        description=description or "TODO: describe what this tool does.",
    )
    header = existing_source.rstrip("\n") if existing_source else _TOOL_IMPORT
    tool_file.write_text(f"{header}\n\n\n{function_source}")
    return Generated(
        file=tool_file,
        symbol=func_name,
        import_line=_import_line(root, tool_file, func_name),
    )


def generate_guardrail(name: str, *, root: Path) -> Generated:
    """Write a new `@guardrail`-decorated function into `root/app/guardrails/`."""
    func_name = _snake_case(name)
    guardrail_file = scaffold(
        root,
        "app",
        "guardrails",
        stem=func_name,
        suffix=".py",
        template=_GUARDRAIL_TEMPLATE.format(func_name=func_name),
    )
    return Generated(
        file=guardrail_file,
        symbol=func_name,
        import_line=_import_line(root, guardrail_file, func_name),
    )


def generate_prompt(name: str, *, root: Path, exist_ok: bool = False) -> Generated:
    """Write a new prompt file into `root/app/prompts/`.

    Plain markdown, not Python: a prompt is text an agent's `instructions` can load, kept out
    of source the same way a query lives outside application code.

    The one place the prompt-file convention lives: `generate_agent` calls this with
    `exist_ok=True` for the stub it writes alongside a new agent, rather than restating where
    the file goes and what it starts out saying.
    """
    file_stem = _snake_case(name)
    return Generated(
        file=scaffold(
            root,
            "app",
            "prompts",
            stem=file_stem,
            suffix=".md",
            template=_PROMPT_TEMPLATE.format(name=file_stem),
            exist_ok=exist_ok,
        )
    )


def generate_evaluation(name: str, *, root: Path, exist_ok: bool = False) -> Generated:
    """Write a new eval dataset into `root/evals/<name>.jsonl`.

    `generate_agent` calls this with `exist_ok=True` for every agent it creates, so running it
    directly is for an agent written by hand, or to start over after deleting the file.

    `name` is the agent's snake_case identity, the same one `runa chat <name>` takes (e.g.
    `support_agent`), not the class name: `runa eval` resolves the Agent from the filename
    (see `cli/eval.py`), so the file needs nothing but cases, one JSON object per line. It
    starts with a single input-only case, graded on task completion and answer relevance. The
    Agent must already exist, checked by scanning source rather than importing the app.
    """
    file_stem = _snake_case(name)
    if _find_agent_name(_require_dir(root, "app", "agents"), file_stem) is None:
        raise AgentNotFound(f"no Agent named {file_stem!r} found under app/agents/")
    return Generated(
        file=scaffold(
            root,
            "evals",
            stem=file_stem,
            suffix=".jsonl",
            template=_EVALUATION_TEMPLATE,
            exist_ok=exist_ok,
        )
    )
