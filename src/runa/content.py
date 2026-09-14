"""content.py: build multimodal message content parts, OpenAI's `image_url`/`text` shape.

`Agent.run`/`run_sync`/`run_streamed` accept `message` as a plain list of strings, auto-detected
by `parts()`: a string ending in an image extension (or a `data:image/...` URI) becomes an
`image()` part, anything else becomes `text()`. `image()`/`text()` are the escape hatch, for a
string the extension heuristic can't classify (a signed URL with no file extension, say) --
build the part explicitly and mix it into the list. `runa._models.openai_chatcompletions` passes
parts straight through; `runa._models.anthropic` translates them into Claude's own content blocks.
"""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")


def image(source: str | Path) -> dict[str, Any]:
    """Build one `image_url` content part from a URL, a `data:` URI, or a local file path.

    A `str` starting with `http://`, `https://`, or `data:` passes through as the URL as-is;
    anything else is treated as a local file path, read and base64-encoded into a `data:` URI,
    with the media type guessed from its extension.
    """
    text_source = str(source)
    if text_source.startswith(("http://", "https://", "data:")):
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


def parts(message: Sequence[str | dict[str, Any]]) -> list[dict[str, Any]]:
    """Build content parts from a `message` list, auto-detecting each bare string.

    A string is passed to `image()` when `_looks_like_image` recognizes it, `text()` otherwise;
    a dict (already a content part, built explicitly for a string the heuristic can't classify)
    passes through unchanged.
    """
    return [
        item if isinstance(item, dict) else image(item) if _looks_like_image(item) else text(item)
        for item in message
    ]


__all__ = ["image", "parts", "text"]
