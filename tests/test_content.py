"""Tests for `runa.content`: OpenAI-shaped multimodal message content parts."""

import base64
from pathlib import Path

from runa import content


def test_text_builds_a_text_part() -> None:
    """`text()` builds a plain `{"type": "text", "text": ...}` part."""
    assert content.text("hello") == {"type": "text", "text": "hello"}


def test_image_passes_an_http_url_through_unchanged() -> None:
    """An `http(s)://` string is used as the `image_url` URL as-is, no encoding."""
    part = content.image("https://example.test/cat.png")

    assert part == {"type": "image_url", "image_url": {"url": "https://example.test/cat.png"}}


def test_image_passes_a_data_uri_through_unchanged() -> None:
    """A `data:` URI is used as the `image_url` URL as-is, no re-encoding."""
    uri = "data:image/png;base64,aGVsbG8="

    assert content.image(uri) == {"type": "image_url", "image_url": {"url": uri}}


def test_image_encodes_a_local_file_into_a_data_uri(tmp_path: Path) -> None:
    """A local file path is read and base64-encoded into a `data:` URI, media type from ext."""
    file = tmp_path / "cat.png"
    file.write_bytes(b"fake-png-bytes")

    part = content.image(file)

    expected_data = base64.b64encode(b"fake-png-bytes").decode()
    assert part == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{expected_data}"},
    }


def test_image_accepts_a_string_local_path_too(tmp_path: Path) -> None:
    """A local path passed as a plain `str` (not `Path`) is encoded the same way."""
    file = tmp_path / "photo.jpg"
    file.write_bytes(b"jpeg-bytes")

    part = content.image(str(file))

    expected_data = base64.b64encode(b"jpeg-bytes").decode()
    assert part["image_url"]["url"] == f"data:image/jpeg;base64,{expected_data}"


def test_parts_auto_detects_text_and_image_strings() -> None:
    """A plain string list is classified by extension, without calling `text()`/`image()`."""
    result = content.parts(["what's in this?", "https://example.test/cat.jpg"])

    assert result == [
        {"type": "text", "text": "what's in this?"},
        {"type": "image_url", "image_url": {"url": "https://example.test/cat.jpg"}},
    ]


def test_parts_recognizes_a_data_image_uri_as_an_image() -> None:
    """A `data:image/...` URI string is classified as an image, not text."""
    uri = "data:image/png;base64,aGVsbG8="

    assert content.parts([uri]) == [{"type": "image_url", "image_url": {"url": uri}}]


def test_parts_ignores_a_query_string_when_checking_the_extension() -> None:
    """A signed URL keeps its extension check on the path, not the trailing query string."""
    url = "https://example.test/cat.jpg?X-Amz-Signature=abc&X-Amz-Expires=60"

    assert content.parts([url]) == [{"type": "image_url", "image_url": {"url": url}}]


def test_parts_treats_an_extensionless_url_as_text() -> None:
    """A URL with no recognizable image extension is classified as plain text."""
    url = "https://example.test/api/img?id=123"

    assert content.parts([url]) == [{"type": "text", "text": url}]


def test_parts_passes_an_explicit_content_dict_through_unchanged() -> None:
    """A pre-built part -- the escape hatch for an unclassifiable string -- is untouched."""
    explicit = content.image("https://example.test/api/img?id=123")

    assert content.parts(["hi", explicit]) == [{"type": "text", "text": "hi"}, explicit]
