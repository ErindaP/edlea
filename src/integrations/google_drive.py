from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError


GOOGLE_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1qk3OodpVP_QWwN9RQH7WQBB-pMG8qqz6"
MAX_IMAGE_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class DriveDemoImage:
    name: str
    file_id: str

    @property
    def download_url(self) -> str:
        return f"https://drive.google.com/uc?export=download&id={self.file_id}"


# Explicit IDs make the demo deterministic and avoid depending on the private
# Google Drive API. If files are replaced in the shared folder, update these IDs.
DRIVE_DEMO_IMAGES = (
    DriveDemoImage("avant.jpg", "1-R6XuL4VBl4XMk4KYNKcLveGkB2zmz0W"),
    DriveDemoImage("apres.jpg", "1k_neRO6cHbvgZqOmayFUNgP0hA9MfNHe"),
)


def _download_image(item: DriveDemoImage, timeout: float = 30.0) -> bytes:
    request = Request(item.download_url, headers={"User-Agent": "PropertyChangeDetectionDemo/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_IMAGE_BYTES:
                raise ValueError(f"{item.name} dépasse la limite de 25 Mo.")
            content = response.read(MAX_IMAGE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Téléchargement impossible pour {item.name}: {exc}") from exc
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError(f"{item.name} dépasse la limite de 25 Mo.")
    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError(f"Format inattendu pour {item.name}: {image.format}")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            f"Google Drive n’a pas renvoyé une image valide pour {item.name}. "
            "Vérifiez que le dossier et les fichiers sont toujours publics."
        ) from exc
    return content


def download_drive_demo_pair(timeout: float = 30.0) -> tuple[bytes, bytes]:
    """Download and validate the fixed public Before/After demonstration pair."""
    before, after = (_download_image(item, timeout) for item in DRIVE_DEMO_IMAGES)
    return before, after
