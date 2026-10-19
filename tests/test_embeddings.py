"""Tests for `runa.embeddings.embed`."""

import asyncio
import json
from typing import Any

import httpx2 as httpx
import pytest
from helpers import ScriptedModel, scripted_agent, text_response, tool_call_response

from runa import embeddings
from runa._loop import LoopCache
from runa._models import OPENAI, transport
from runa.exceptions import ModelBehaviorError, ServiceUnavailable, UserError
from runa.tool import tool


@pytest.fixture(autouse=True)
def _reset_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with no cached client, no API key, and no endpoint override."""
    monkeypatch.setattr(transport, "_clients", LoopCache())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RUNA_EMBEDDINGS_BASE_URL", raising=False)
    monkeypatch.delenv("RUNA_EMBEDDINGS_API_KEY", raising=False)


def _mock_client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="https://api.openai.com/v1/", transport=httpx.MockTransport(handler)
    )


def _serve(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    """Answer every embeddings request with `handler`, through the real client cache."""
    monkeypatch.setattr(transport, "_new_client", lambda base_url, api_key: _mock_client(handler))


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


def test_embed_raises_user_error_for_a_refused_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A revoked key is configuration, so retrieval ends the run instead of skipping memory."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-revoked")
    _serve(monkeypatch, lambda request: httpx.Response(401, json={"error": "invalid api key"}))

    with pytest.raises(UserError, match="401"):
        asyncio.run(embeddings.embed(["a"]))


def test_embed_raises_service_unavailable_once_retries_run_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 503 that outlasts the retries is the one failure a context source is skipped for."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    async def sleep(seconds: float) -> None:
        pass

    monkeypatch.setattr("runa._models.transport.asyncio.sleep", sleep)
    _serve(monkeypatch, lambda request: httpx.Response(503))

    with pytest.raises(ServiceUnavailable, match="503"):
        asyncio.run(embeddings.embed(["a"]))


def test_embed_retries_rate_limits_like_a_chat_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 during retrieval is retried, honoring `retry-after`, not taken as "no memories"."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    delays: list[float] = []

    async def sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("runa._models.transport.asyncio.sleep", sleep)
    replies = [
        httpx.Response(429, headers={"retry-after": "2"}),
        httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1]}]}),
    ]
    _serve(monkeypatch, lambda request: replies.pop(0))

    assert asyncio.run(embeddings.embed(["a"])) == [[0.1]]
    assert delays == [2.0]


def test_each_event_loop_gets_its_own_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two `asyncio.run` calls -- two `run_sync` turns -- don't share one loop's connections.

    A client reused across loops fails the second turn with "Event loop is closed", which
    retrieval would swallow as "no matches": the symptom `runa._loop` exists to prevent.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    built: list[httpx.AsyncClient] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1]}]})

    def new_client(base_url: str, api_key: str) -> httpx.AsyncClient:
        built.append(_mock_client(handler))
        return built[-1]

    monkeypatch.setattr(transport, "_new_client", new_client)

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

    client = embeddings._client()

    assert str(client.base_url) == "https://gateway.example.com/v1/"
    assert client.headers["Authorization"] == "Bearer sk-test"


def test_gateway_token_can_differ_from_the_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """`RUNA_EMBEDDINGS_API_KEY` wins, so chat and embeddings can hold two different tokens."""
    monkeypatch.setenv("RUNA_EMBEDDINGS_BASE_URL", "https://gateway.example.com/v1/")
    monkeypatch.setenv("RUNA_EMBEDDINGS_API_KEY", "gw-token")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    assert embeddings._client().headers["Authorization"] == "Bearer gw-token"


def test_each_endpoint_gets_its_own_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client cache is keyed by the resolved base URL, not by a module constant."""
    monkeypatch.setenv("RUNA_EMBEDDINGS_API_KEY", "gw-token")

    default = embeddings._client()
    monkeypatch.setenv("RUNA_EMBEDDINGS_BASE_URL", "https://gateway.example.com/v1/")
    gateway = embeddings._client()

    assert default is not gateway
    assert str(gateway.base_url) == "https://gateway.example.com/v1/"


class _EmbeddingSource:
    """A context source that embeds the user's message before the turn, and keeps nothing."""

    async def context(self, query: str, run_context: Any) -> None:
        await embeddings.embed([query])

    async def after_turn(self, *args: Any) -> list[str]:
        return []


def _serve_usage(monkeypatch: pytest.MonkeyPatch, tokens: int) -> None:
    """Answer every embeddings request with one vector, reporting `tokens` spent."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        body = {
            "data": [{"index": 0, "embedding": [0.1]}],
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }
        return httpx.Response(200, json=body)

    _serve(monkeypatch, handler)


def test_an_embedding_made_inside_a_run_is_charged_to_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A retrieval's embedding is in the run's usage, and an `"llm"` span under the retrieval."""
    _serve_usage(monkeypatch, 7)
    model = ScriptedModel([text_response("ok")])
    agent = scripted_agent(model=model, memory=_EmbeddingSource())

    result = agent.run_sync("hi")

    assert result.usage.total_tokens == 2 + 7
    assert result.usage.requests == 2
    (retrieval,) = [s for s in result.trace.spans if s.type == "retrieval"]
    (embedding,) = [s for s in result.trace.spans if s.parent_id == retrieval.id]
    assert embedding.type == "llm"
    assert embedding.name == "text-embedding-3-small"
    assert embedding.output["usage"]["input_tokens"] == 7


def test_an_embedding_counts_against_the_runs_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run's budget is its whole spend: a retrieval's embedding can exhaust it."""
    _serve_usage(monkeypatch, 100)
    model = ScriptedModel([text_response("ok")])
    agent = scripted_agent(model=model, memory=_EmbeddingSource(), max_tokens=50)

    result = agent.run_sync("hi")

    assert result.status == "error"
    assert "max tokens (50) exceeded: 102 used" in (result.error or "")


def test_an_embedding_made_by_a_tool_is_charged_under_its_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool that embeds (`memory = "llm"`'s `search_memory`, say) puts its span under the call."""
    _serve_usage(monkeypatch, 3)

    @tool
    async def lookup(query: str) -> str:
        """Embed `query`.

        query: what to look up
        """
        await embeddings.embed([query])
        return "found"

    model = ScriptedModel(
        [tool_call_response("lookup", '{"query": "cats"}'), text_response("done")]
    )
    agent = scripted_agent(model=model, tools=[lookup])

    result = agent.run_sync("hi")

    assert result.usage.total_tokens == 2 + 3 + 2
    (call,) = [s for s in result.trace.spans if s.type == "tool"]
    (embedding,) = [s for s in result.trace.spans if s.parent_id == call.id]
    assert embedding.type == "llm"
