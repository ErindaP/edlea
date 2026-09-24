from io import BytesIO

from PIL import Image

from src.image_io import load_image


def test_load_image_applies_exif_orientation():
    source = Image.new("RGB", (60, 40), (120, 80, 40))
    exif = Image.Exif()
    exif[274] = 6
    buffer = BytesIO()
    source.save(buffer, format="JPEG", exif=exif)

    loaded = load_image(buffer.getvalue(), max_size=None)

    assert loaded.shape == (60, 40, 3)
