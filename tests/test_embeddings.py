"""Tests for `runa.embeddings.embed`."""

import asyncio
import json
from typing import Any

import httpx2 as httpx
import pytest

from runa import embeddings
from runa._loop import LoopCache
from runa.exceptions import ModelBehaviorError, UserError


@pytest.fixture(autouse=True)
def _reset_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with no cached client and no API key set."""
    monkeypatch.setattr(embeddings, "_clients", LoopCache())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


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
