from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    out = Path(__file__).parents[1] / "examples"
    out.mkdir(exist_ok=True)
    before = np.full((480, 720, 3), (232, 220, 196), dtype=np.uint8)
    cv2.rectangle(before, (0, 345), (720, 480), (130, 150, 170), -1)
    cv2.rectangle(before, (70, 100), (220, 340), (190, 150, 110), -1)
    cv2.rectangle(before, (480, 95), (650, 300), (110, 160, 205), -1)
    cv2.circle(before, (380, 275), 48, (65, 80, 95), -1)
    cv2.putText(before, "BEFORE", (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 1, (45, 45, 45), 2)
    after = before.copy()
    cv2.line(after, (275, 165), (430, 215), (30, 30, 30), 8)
    cv2.rectangle(after, (485, 270), (650, 300), (90, 115, 155), -1)
    cv2.putText(after, "AFTER", (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 1, (45, 45, 45), 2)
    cv2.imwrite(str(out / "before.jpg"), cv2.cvtColor(before, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "after.jpg"), cv2.cvtColor(after, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()

