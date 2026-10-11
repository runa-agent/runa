"""embeddings.py: text -> vector, for `runa.memory` and `runa.knowledge`.

OpenAI is the default: it's the one provider among Runa's chat backends that also exposes an
embeddings endpoint, and `OPENAI_API_KEY` is already the convention every scaffolded project has
(see `cli/new.py`'s `.env` template). Where that endpoint lives and which variable holds its key
aren't restated here: they're read from `_models.OPENAI`, the same record `ModelProvider` routes
`gpt-*` with, so OpenAI's address is written in exactly one place in Runa. Talks to it directly
over `httpx2`, same as `OpenAICompatibleModel`, rather than pulling in the `openai` SDK for one
endpoint.

An agent's `model` doesn't pick it. Embeddings are the one place Runa needs a vendor no setting
named: `claude-sonnet-5` with `memory = "auto"` still needs `OPENAI_API_KEY`. A deployment that
can't send text there -- Azure, a corporate gateway, a self-hosted model -- sets
`RUNA_EMBEDDINGS_BASE_URL`, and `RUNA_EMBEDDINGS_API_KEY` when that endpoint's token isn't the one
`OPENAI_API_KEY` holds. It's an environment variable rather than an argument for the reason
`RUNA_DATABASE_URL` is: `memory = "auto"` names no instance to pass one to, and where a
deployment's embeddings come from is not a call site's decision.

Its client is held per event loop, like every other long-lived client in Runa (see `runa._loop`):
`Agent.run_sync` opens a loop per call, and a client carried over from a closed one fails the next
retrieval with "Event loop is closed".
"""

import os

import httpx2 as httpx

from runa._loop import LoopCache
from runa._models import OPENAI
from runa.exceptions import ModelBehaviorError, UserError

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# Shared by `runa.memory` and `runa.knowledge`: the vector size each built-in model returns, so
# `dimensions=` only needs setting explicitly for a model not listed here.
EMBEDDING_DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

_BASE_URL_ENV = "RUNA_EMBEDDINGS_BASE_URL"
_API_KEY_ENV = "RUNA_EMBEDDINGS_API_KEY"

_EMBEDDINGS_PATH = "embeddings"

_clients: LoopCache[str, httpx.AsyncClient] = LoopCache()


def resolve_dimensions(model: str, dimensions: int | None) -> int:
    """`dimensions` if given, else `model`'s built-in size; raises if neither is known.

    Shared by `runa.memory.Memory` and `runa.knowledge.Knowledge`'s `__init__`.
    """
    resolved = dimensions or EMBEDDING_DIMENSIONS.get(model)
    if resolved is None:
        raise ValueError(f"unknown embedding size for {model!r}; pass dimensions= explicitly")
    return resolved


def _base_url() -> str:
    """Where embeddings requests go: `RUNA_EMBEDDINGS_BASE_URL`, else OpenAI's own endpoint."""
    return os.environ.get(_BASE_URL_ENV) or OPENAI.base_url


def _api_key() -> str:
    """The token for `_base_url()`: `RUNA_EMBEDDINGS_API_KEY`, else `OPENAI_API_KEY`'s.

    Raises `UserError` when neither is set, which is why the client is built lazily: an app that
    declares neither `memory` nor `knowledge` should not need an embeddings key at all.
    """
    api_key = os.environ.get(_API_KEY_ENV) or os.environ.get(OPENAI.api_key_env)
    if api_key is None:
        raise UserError(
            f"Neither {_API_KEY_ENV} nor {OPENAI.api_key_env} is set, and runa.memory/"
            f"runa.knowledge embed through {_base_url()} whatever an agent's own model is. "
            f"Set one of them, or {_BASE_URL_ENV} to embed somewhere else."
        )
    return api_key


def _new_client() -> httpx.AsyncClient:
    """A client pointed at `_base_url()`, carrying the token `_api_key()` resolves."""
    return httpx.AsyncClient(
        base_url=_base_url(), headers={"Authorization": f"Bearer {_api_key()}"}, timeout=60.0
    )


def _get_client() -> httpx.AsyncClient:
    """This loop's client, keyed by base URL the way `ModelProvider` keys its own by prefix."""
    return _clients.get(_base_url(), _new_client)


async def embed(texts: list[str], *, model: str = DEFAULT_EMBEDDING_MODEL) -> list[list[float]]:
    """Embed `texts` with `model`, returning one vector per input, in the same order."""
    response = await _get_client().post(_EMBEDDINGS_PATH, json={"model": model, "input": texts})
    if response.status_code >= 400:
        raise ModelBehaviorError(
            f"embeddings request failed with {response.status_code}: {response.text[:2000]}"
        )
    data = response.json()["data"]
    return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]


__all__ = ["DEFAULT_EMBEDDING_MODEL", "EMBEDDING_DIMENSIONS", "embed", "resolve_dimensions"]
