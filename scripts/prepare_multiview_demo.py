"""Download a licensed wall photo and synthesize repeatable viewpoint tests.

The scan frames are *simulated projections of the same photograph*, not
independent real-world captures. They test registration/coverage mechanics.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image


SOURCE_URL = "https://upload.wikimedia.org/wikipedia/commons/d/dc/Brick_wall%2C_inside.jpg"
SOURCE_PAGE = "https://commons.wikimedia.org/wiki/File:Brick_wall,_inside.jpg"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"


def make_view(image: np.ndarray, quad: list[tuple[float, float]], size: tuple[int, int] = (680, 680)) -> np.ndarray:
    height, width = image.shape[:2]
    source = np.float32([[u * (width - 1), v * (height - 1)] for u, v in quad])
    target = np.float32([[0, 0], [size[0] - 1, 0], [size[0] - 1, size[1] - 1], [0, size[1] - 1]])
    transform = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(image, transform, size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[1] / "examples" / "multiview")
    args = parser.parse_args()
    request = Request(SOURCE_URL, headers={"User-Agent": "PropertyChangeDetectionDemo/1.0 (educational demo)"})
    with urlopen(request, timeout=20) as response:
        source = np.asarray(Image.open(BytesIO(response.read())).convert("RGB"))
    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(source).save(args.output / "reference.jpg", quality=93)

    left = [(0.00, 0.04), (0.70, 0.00), (0.75, 0.96), (0.03, 0.91)]
    right = [(0.27, 0.05), (0.96, 0.01), (0.99, 0.96), (0.31, 0.93)]
    changed = source.copy()
    height, width = changed.shape[:2]
    # An unambiguous simulated new mark on the same planar wall.
    cv2.line(changed, (int(width * 0.54), int(height * 0.30)),
             (int(width * 0.59), int(height * 0.67)), (22, 22, 25), 8, cv2.LINE_AA)
    for name, data in (("scan_gauche.jpg", make_view(source, left)),
                       ("scan_droite.jpg", make_view(source, right)),
                       ("scan_marque.jpg", make_view(changed, right))):
        Image.fromarray(data).save(args.output / name, quality=94)
    (args.output / "SOURCE.txt").write_text(
        "Photographie originale : Mia Gaitanidis, « Brick wall, inside »\n"
        f"{SOURCE_PAGE}\nLicence CC BY-SA 4.0 : {LICENSE_URL}\n"
        "Les autres images sont des vues perspectivées synthétiques de cette même photo. "
        "La marque sombre dans scan_marque.jpg a été ajoutée numériquement pour la démonstration.\n",
        encoding="utf-8",
    )
    print(f"Images créées dans {args.output}")


if __name__ == "__main__":
    main()
