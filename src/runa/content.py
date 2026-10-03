"""content.py: build multimodal message content parts, OpenAI's `image_url`/`text` shape.

`Agent.run`/`run_sync`/`run_streamed` accept `message` as a plain list, auto-detected by
`parts()`: a URL or `data:image/...` URI that reads as an image becomes an `image()` part, a
`Path` is read off disk into one, anything else becomes `text()`. A bare `str` naming a local
file is refused rather than read, because a string in a message list is just as likely to be a
user's own words as the program's own asset -- `Path("photo.png")` or `image("photo.png")` is
how you say the path is yours. Those two are also the escape hatch for a string the extension
heuristic can't classify (a signed URL with no file extension, say): build the part explicitly
and mix it into the list. `runa._models.openai_chatcompletions` passes parts straight through;
`runa._models.anthropic` translates them into Claude's own content blocks.
"""

import base64
import mimetypes
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
_URL_SCHEMES = ("http://", "https://", "data:")


def image(source: str | Path) -> dict[str, Any]:
    """Build one `image_url` content part from a URL, a `data:` URI, or a local file path.

    A `str` starting with `http://`, `https://`, or `data:` passes through as the URL as-is;
    anything else is treated as a local file path, read and base64-encoded into a `data:` URI,
    with the media type guessed from its extension.

    Being explicit is the whole point: `image()` reads whatever path it is handed, so the path
    has to be the program's own. Never pass it a string that came from a user -- `parts()`
    refuses to guess for exactly that reason.
    """
    text_source = str(source)
    if text_source.startswith(_URL_SCHEMES):
        url = text_source
    else:
        path = Path(source)
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = base64.b64encode(path.read_bytes()).decode()
        url = f"data:{media_type};base64,{data}"
    return {"type": "image_url", "image_url": {"url": url}}


def text(value: str) -> dict[str, Any]:
    """Build one `text` content part, for mixing with `image()` in a multimodal message."""
    return {"type": "text", "text": value}


def _looks_like_image(value: str) -> bool:
    """Whether a bare string in a `message` list reads as an image, not text.

    A `data:image/...` URI, or a path/URL whose extension -- ignoring a trailing query string
    or fragment -- is a common image format.
    """
    if value.startswith("data:image/"):
        return True
    path = value.split("?", 1)[0].split("#", 1)[0]
    return path.lower().endswith(_IMAGE_EXTENSIONS)


_NOT_ONE_TURN = (
    "message only takes one user turn: a string, or a list of text and image parts. It does "
    "not take a list of past messages. To start from an earlier conversation, set "
    "agent.history = [...] directly, or seed a session with session.add_items([...]) before "
    "the first run."
)

_LOCAL_PATH_IS_NOT_A_STRING = (
    "{value!r} names a local image file, and a bare string is never read from disk: a string in "
    "a message list can be a user's own words, so reading the path it names would hand a user "
    "the filesystem. Pass Path({value!r}) or content.image({value!r}) instead -- both say the "
    "path is the program's own."
)


def _part(item: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Classify one item of a `message` list, rejecting anything that isn't a content part."""
    if isinstance(item, Path):
        return image(item)
    if isinstance(item, dict):
        if "type" not in item:
            raise TypeError(_NOT_ONE_TURN)
        return item
    if not _looks_like_image(item):
        return text(item)
    if item.startswith(_URL_SCHEMES):
        return image(item)
    raise ValueError(_LOCAL_PATH_IS_NOT_A_STRING.format(value=item))


def parts(message: Sequence[str | Path | dict[str, Any]]) -> list[dict[str, Any]]:
    """Build content parts from a `message` list, auto-detecting each item.

    A `Path` is always an image, read off disk by `image()`. A string is an image only when it is
    a URL or `data:` URI that `_looks_like_image` recognizes, and `text()` otherwise; a string
    naming a local file raises `ValueError`, since the list may carry user input and a local read
    has to be the program's own choice (`Path(...)` or `image(...)` says so). A dict (already a
    content part, built explicitly for a string the heuristic can't classify) passes through
    unchanged.

    A dict with no `"type"` isn't a content part at all -- almost always a whole message
    (`{"role": ..., "content": ...}`) from a transcript someone is trying to replay -- and raises
    `TypeError` rather than reaching a provider, where it would be dropped silently (Anthropic)
    or rejected as a bad request (OpenAI).
    """
    return [_part(item) for item in message]


__all__ = ["image", "parts", "text"]
