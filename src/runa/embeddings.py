"""embeddings.py: text -> vector, for `runa.memory` and `runa.knowledge`.

Embeddings go to OpenAI, at the address and key variable `ModelProvider` routes `gpt-*` with,
through the transport a chat turn uses (`runa._models.transport`), so they get its per-loop
client, timeout and retries. An agent's `model` doesn't pick it: `claude-sonnet-5` with
`memory = "auto"` still needs `OPENAI_API_KEY`. `RUNA_EMBEDDINGS_BASE_URL` points it elsewhere
(Azure, a gateway, something self-hosted), with `RUNA_EMBEDDINGS_API_KEY` when that endpoint's
token differs.
"""

import os

import httpx2 as httpx

from runa._metering import metered
from runa._models import OPENAI, transport
from runa._types import Usage
from runa.exceptions import UserError

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# Shared by `runa.memory` and `runa.knowledge`: the vector size each built-in model returns, so
# `dimensions=` only needs setting explicitly for a model not listed here.
EMBEDDING_DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

_BASE_URL_ENV = "RUNA_EMBEDDINGS_BASE_URL"
_API_KEY_ENV = "RUNA_EMBEDDINGS_API_KEY"

_EMBEDDINGS_PATH = "embeddings"


def resolve_dimensions(model: str, dimensions: int | None) -> int:
    """`dimensions` if given, else `model`'s built-in size; raises if neither is known.

    Shared by `runa.memory.Memory` and `runa.knowledge.Knowledge`'s `__init__`.
    """
    resolved = dimensions or EMBEDDING_DIMENSIONS.get(model)
    if resolved is None:
        raise UserError(f"unknown embedding size for {model!r}; pass dimensions= explicitly")
    return resolved


def _base_url() -> str:
    """Where embeddings requests go: `RUNA_EMBEDDINGS_BASE_URL`, else OpenAI's own endpoint."""
    return os.environ.get(_BASE_URL_ENV) or OPENAI.base_url


def _api_key() -> str:
    """The token for `_base_url()`: `RUNA_EMBEDDINGS_API_KEY`, else `OPENAI_API_KEY`'s.

    Raises `UserError` when neither is set; the client is built lazily, so an app that declares
    neither `memory` nor `knowledge` needs no embeddings key.
    """
    api_key = os.environ.get(_API_KEY_ENV) or os.environ.get(OPENAI.api_key_env)
    if api_key is None:
        raise UserError(
            f"Neither {_API_KEY_ENV} nor {OPENAI.api_key_env} is set, and runa.memory/"
            f"runa.knowledge embed through {_base_url()} whatever an agent's own model is. "
            f"Set one of them, or {_BASE_URL_ENV} to embed somewhere else."
        )
    return api_key


def _client() -> httpx.AsyncClient:
    """This loop's client for `_base_url()`, carrying the token `_api_key()` resolves."""
    return transport.client(_base_url(), _api_key())


async def embed(texts: list[str], *, model: str = DEFAULT_EMBEDDING_MODEL) -> list[list[float]]:
    """Embed `texts` with `model`, returning one vector per input, in the same order.

    Made inside a run, the request is one of that run's: its tokens are in the run's usage, and
    it is an `"llm"` span in the run's trace.
    """
    with metered(model, input=texts) as charge:
        client = _client()
        body = {"model": model, "input": texts}
        response = await transport.send(lambda: client.post(_EMBEDDINGS_PATH, json=body))
        transport.raise_for_status(response.status_code, response.text)
        reply = response.json()
        charge(_usage(reply.get("usage") or {}))
    data = reply["data"]
    return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]


def _usage(reported: dict[str, int]) -> Usage:
    """One embeddings request's `Usage`: every token it reports is an input token."""
    tokens = reported.get("prompt_tokens", 0)
    return Usage(
        requests=1,
        input_tokens=tokens,
        total_tokens=reported.get("total_tokens", tokens),
    )


__all__ = ["DEFAULT_EMBEDDING_MODEL", "EMBEDDING_DIMENSIONS", "embed", "resolve_dimensions"]
