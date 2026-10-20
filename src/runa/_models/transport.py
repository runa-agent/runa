"""transport.py: the one HTTP transport every bearer-token JSON API Runa calls goes through.

The chat-completions backend and `runa.embeddings` both send through it, so a client's timeout,
its auth header, its per-loop lifetime and the retry policy are decided once. Each caller resolves
its own address and key and owns its request body; nothing here knows which API it is talking to.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable

import httpx2 as httpx

from runa._loop import LoopCache
from runa.exceptions import ModelBehaviorError, RunaError, ServiceUnavailable, UserError

DEFAULT_MAX_RETRIES = 2  # the same default as the `anthropic` SDK's, so both backends agree
_MAX_BACKOFF = 8.0
_TIMEOUT = 600.0

_clients: LoopCache[tuple[str, str], httpx.AsyncClient] = LoopCache()


def client(base_url: str, api_key: str) -> httpx.AsyncClient:
    """This event loop's client for `base_url`, sending `api_key` as a bearer token.

    Held per loop (see `runa._loop`) and per address and key, so callers that resolve the same
    endpoint share connections, and a rotated key gets a client of its own.
    """
    return _clients.get((base_url, api_key), lambda: _new_client(base_url, api_key))


def _new_client(base_url: str, api_key: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=_TIMEOUT
    )


def _retryable(status_code: int) -> bool:
    return status_code in (408, 409, 429) or status_code >= 500


def _backoff(attempt: int, retry_after: str | None = None) -> float:
    """Seconds to wait before retry `attempt`: the server's `retry-after`, else jittered 2^n."""
    if retry_after is not None:
        try:
            return min(float(retry_after), _MAX_BACKOFF)
        except ValueError:
            pass
    return min(0.5 * 2**attempt, _MAX_BACKOFF) * random.uniform(0.75, 1.0)


async def send(
    request: Callable[[], Awaitable[httpx.Response]], max_retries: int | None = None
) -> httpx.Response:
    """`request()` until it succeeds, retrying connection errors and 408/409/429/5xx with backoff.

    `max_retries` is `ModelSettings.max_retries`: `None` means `DEFAULT_MAX_RETRIES`, and `0`
    disables retrying. A connection that still fails after the last attempt raises
    `ServiceUnavailable`; a status that still fails is returned, for `raise_for_status`.
    """
    retries = DEFAULT_MAX_RETRIES if max_retries is None else max(0, max_retries)
    attempt = 0
    while True:
        last = attempt == retries
        try:
            response = await request()
        except httpx.TransportError as exc:
            if last:
                raise ServiceUnavailable(f"model request failed: {exc!r}") from exc
            await asyncio.sleep(_backoff(attempt))
        else:
            if last or not _retryable(response.status_code):
                return response
            await response.aclose()
            await asyncio.sleep(_backoff(attempt, response.headers.get("retry-after")))
        attempt += 1


def status_error(status_code: int, body: str) -> RunaError:
    """The error a request that failed with `status_code` ends in; `body` is capped for legibility.

    401/403 is a `UserError`: the key was sent and refused, which retrying or skipping won't fix.
    A retryable status is a `ServiceUnavailable`, since `send` already retried it. Any other is a
    `ModelBehaviorError`, a request the service couldn't make sense of.
    """
    message = f"model request failed with {status_code}: {body[:2000]}"
    if status_code in (401, 403):
        return UserError(f"{message}\nThe API key was refused; check that it is current.")
    if _retryable(status_code):
        return ServiceUnavailable(message)
    return ModelBehaviorError(message)


def raise_for_status(status_code: int, body: str) -> None:
    """Raise `status_error`'s error for a status of 400 or above."""
    if status_code >= 400:
        raise status_error(status_code, body)


__all__ = ["DEFAULT_MAX_RETRIES", "client", "raise_for_status", "send", "status_error"]
