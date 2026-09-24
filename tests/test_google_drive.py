from __future__ import annotations

from io import BytesIO

from PIL import Image

from src.integrations import google_drive


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content
        self.headers = {"Content-Length": str(len(content))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, limit: int) -> bytes:
        return self.content[:limit]


def jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 24), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_download_drive_pair_validates_both_images(monkeypatch):
    payloads = iter((jpeg_bytes((20, 30, 40)), jpeg_bytes((80, 90, 100))))
    monkeypatch.setattr(google_drive, "urlopen", lambda *_args, **_kwargs: FakeResponse(next(payloads)))

    before, after = google_drive.download_drive_demo_pair()

    assert Image.open(BytesIO(before)).size == (32, 24)
    assert Image.open(BytesIO(after)).size == (32, 24)


def test_download_drive_pair_rejects_html(monkeypatch):
    monkeypatch.setattr(google_drive, "urlopen", lambda *_args, **_kwargs: FakeResponse(b"<html>private</html>"))

    try:
        google_drive.download_drive_demo_pair()
    except ValueError as exc:
        assert "image valide" in str(exc)
    else:
        raise AssertionError("An HTML Drive response must be rejected")
