"""What an entry point is allowed to cost, and what the facade still has to offer.

`runa/__init__.py` resolves its public names lazily, so the one thing no other test would notice
breaking is the reason it does: a single eager `from runa.agent import Agent` added back here
would quietly reattach the Anthropic SDK and the MCP client to every `runa generate`, every
`@tool` file and every cold start. `test_facade_stays_light` is that tripwire, and it runs in a
subprocess because this one already has half of PyPI in `sys.modules`.

A lazy facade over an eager layer buys nothing, which is what `test_entry_points_stay_light`
covers: `runa.agent` reads `DEFAULT_MODEL`, `check_model_name` and the `Model` protocol at module
scope, and `runa.cli.main` reads `DEFAULT_MODEL`, so an eager backend import anywhere under
`runa._models` puts a provider's SDK back under every one of those entry points -- including the
commands and the `gpt-*`-only apps that can never call it. `ModelProvider.get_model` imports the
backend it routes to, and nothing else does.

The rest holds the namespace to what it was before it went lazy: every name in `__all__` still
resolves, `_EXPORTS` and `__all__` can't drift apart, and a name that isn't public still fails.
"""

import importlib.util
import subprocess
import sys

import runa

_HEAVY = ("anthropic", "mcp", "httpx2")
"""The imports worth a subprocess to catch: a vendor SDK, the MCP client, and the HTTP layer."""


def _heavy_modules_after(statement: str) -> set[str]:
    """Which of `_HEAVY` a fresh interpreter has loaded once it has run `statement`.

    In a subprocess, because this one already has half of PyPI in `sys.modules`.
    """
    probe = f"import sys\n{statement}\nprint(' '.join(m for m in {_HEAVY!r} if m in sys.modules))"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    return set(result.stdout.split())


_SELF_RESOLVING = {"db", "__version__"}
"""The two names `__getattr__` handles itself rather than reading out of a module."""

_EAGER = {"tool", "guardrail", "approval"}
"""The names that also name a submodule, so the facade has to bind them itself."""


def test_facade_stays_light():
    """`import runa` pulls in neither the Anthropic SDK nor the MCP client."""
    assert _heavy_modules_after("import runa") == set()


def test_entry_points_stay_light():
    """No entry point loads a provider's SDK, and the one that routes to it still does.

    `from runa import Agent` is every app's first line and `runa.cli.main` is every command's,
    and neither knows yet which provider -- if any -- this run will name. The HTTP layer is a
    fair cost for both (`runa._models.provider` needs it to build a client at all); a specific
    vendor's SDK is not.
    """
    assert "anthropic" not in _heavy_modules_after("from runa import Agent")
    assert "anthropic" not in _heavy_modules_after("import runa.cli.main")
    assert "anthropic" in _heavy_modules_after(
        "from runa._models import ModelProvider\nModelProvider().get_model('claude-sonnet-5')"
    )


def test_every_public_name_resolves():
    """Every name in `__all__` is reachable, so a wrong module in `_EXPORTS` fails here."""
    for name in runa.__all__:
        assert getattr(runa, name) is not None, name


def test_exports_and_all_agree():
    """`_EXPORTS` plus the self-resolving and eager names is exactly `__all__`."""
    assert set(runa._EXPORTS) | _SELF_RESOLVING | _EAGER == set(runa.__all__)


def test_a_name_that_shadows_a_submodule_is_bound_eagerly():
    """`from runa import tool` has to be the decorator, never the module of the same name.

    `__getattr__` runs only for an attribute the package doesn't have, and importing `runa.tool`
    anywhere gives it one. So a public name that doubles as a submodule name can't be lazy, and
    this is the check that says so for the next one added rather than for today's three.
    """
    for name in runa.__all__:
        if name == "db" or importlib.util.find_spec(f"runa.{name}") is None:
            continue
        assert name in _EAGER, f"{name} also names a submodule, so it can't be lazy"
        assert getattr(runa, name).__module__ == f"runa.{name}"


def test_a_resolved_name_becomes_a_plain_global():
    """The first read caches, so the import machinery runs once per name, not once per read."""
    assert runa.Agent is runa.Agent
    assert "Agent" in vars(runa)


def test_an_unknown_name_is_an_attribute_error():
    """A typo in a public name fails as `AttributeError`, naming what was asked for."""
    missing = "Nonexistent"
    try:
        getattr(runa, missing)
    except AttributeError as exc:
        assert missing in str(exc)
    else:
        raise AssertionError("expected AttributeError")


def test_dir_lists_the_public_surface():
    """`__getattr__` keeps names out of `globals()`, so `__dir__` has to name them itself."""
    assert set(runa.__all__) <= set(dir(runa))
