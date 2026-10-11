"""Tests for `runa.embeddings.embed`."""

import asyncio
import json
from typing import Any

import httpx2 as httpx
import pytest

from runa import embeddings
from runa._loop import LoopCache
from runa._models import OPENAI
from runa.exceptions import ModelBehaviorError, UserError


@pytest.fixture(autouse=True)
def _reset_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with no cached client, no API key, and no endpoint override."""
    monkeypatch.setattr(embeddings, "_clients", LoopCache())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RUNA_EMBEDDINGS_BASE_URL", raising=False)
    monkeypatch.delenv("RUNA_EMBEDDINGS_API_KEY", raising=False)


def _mock_client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="https://api.openai.com/v1/", transport=httpx.MockTransport(handler)
    )


def _serve(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    """Answer every embeddings request with `handler`, through the real client cache."""
    monkeypatch.setattr(embeddings, "_new_client", lambda: _mock_client(handler))


def test_embed_without_api_key_raises_user_error() -> None:
    """No `OPENAI_API_KEY` fails clearly instead of trying the request."""
    with pytest.raises(UserError):
        asyncio.run(embeddings.embed(["hi"]))


def test_embed_posts_model_and_input_and_returns_vectors_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The response's `data` may arrive out of order; `embed` reorders it by `index`."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body == {"model": "text-embedding-3-small", "input": ["a", "b"]}
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.4, 0.5]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            },
        )

    _serve(monkeypatch, handler)

    vectors = asyncio.run(embeddings.embed(["a", "b"]))

    assert vectors == [[0.1, 0.2], [0.4, 0.5]]


def test_embed_raises_on_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-2xx response is surfaced as a `ModelBehaviorError`."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad request"})

    _serve(monkeypatch, handler)

    with pytest.raises(ModelBehaviorError):
        asyncio.run(embeddings.embed(["a"]))


def test_each_event_loop_gets_its_own_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two `asyncio.run` calls -- two `run_sync` turns -- don't share one loop's connections.

    A client reused across loops fails the second turn with "Event loop is closed", which
    retrieval would swallow as "no matches": the symptom `runa._loop` exists to prevent.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    built: list[httpx.AsyncClient] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1]}]})

    def new_client() -> httpx.AsyncClient:
        built.append(_mock_client(handler))
        return built[-1]

    monkeypatch.setattr(embeddings, "_new_client", new_client)

    assert asyncio.run(embeddings.embed(["a"])) == [[0.1]]
    assert asyncio.run(embeddings.embed(["a"])) == [[0.1]]

    assert len(built) == 2
    assert built[0] is not built[1]


def test_default_endpoint_is_the_record_provider_routes_gpt_models_with() -> None:
    """OpenAI's address is written once, in `_models`, not restated by `runa.embeddings`."""
    assert embeddings._base_url() == OPENAI.base_url


def test_missing_key_error_names_both_variables_and_the_endpoint() -> None:
    """An all-Anthropic app turning on memory learns which key it now needs, and why."""
    with pytest.raises(UserError) as caught:
        asyncio.run(embeddings.embed(["hi"]))

    message = str(caught.value)
    assert "RUNA_EMBEDDINGS_API_KEY" in message
    assert "OPENAI_API_KEY" in message
    assert OPENAI.base_url in message


def test_base_url_env_points_embeddings_at_a_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployment that can't reach api.openai.com embeds elsewhere, with no code change."""
    monkeypatch.setenv("RUNA_EMBEDDINGS_BASE_URL", "https://gateway.example.com/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    client = embeddings._new_client()

    assert str(client.base_url) == "https://gateway.example.com/v1/"
    assert client.headers["Authorization"] == "Bearer sk-test"


def test_gateway_token_can_differ_from_the_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """`RUNA_EMBEDDINGS_API_KEY` wins, so chat and embeddings can hold two different tokens."""
    monkeypatch.setenv("RUNA_EMBEDDINGS_BASE_URL", "https://gateway.example.com/v1/")
    monkeypatch.setenv("RUNA_EMBEDDINGS_API_KEY", "gw-token")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    assert embeddings._new_client().headers["Authorization"] == "Bearer gw-token"


def test_each_endpoint_gets_its_own_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client cache is keyed by the resolved base URL, not by a module constant."""
    monkeypatch.setenv("RUNA_EMBEDDINGS_API_KEY", "gw-token")

    default = embeddings._get_client()
    monkeypatch.setenv("RUNA_EMBEDDINGS_BASE_URL", "https://gateway.example.com/v1/")
    gateway = embeddings._get_client()

    assert default is not gateway
    assert str(gateway.base_url) == "https://gateway.example.com/v1/"
