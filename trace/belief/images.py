"""PNG encoding inside the size window Jev accepts.

Jev rejects images outside 280..1568 px per side and below 78_400 px total.
It does not resize them. Callers that already drew a legal frame pass through.
"""

from __future__ import annotations

import io

import numpy as np

MIN_SIDE = 280
MAX_SIDE = 1568
MIN_PIXELS = 78_400


def png_bytes(frame: np.ndarray) -> bytes:
    from PIL import Image

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    image = _fit(image)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _fit(image):
    from PIL import Image

    width, height = image.size
    scale_up = max(MIN_SIDE / width, MIN_SIDE / height, (MIN_PIXELS / (width * height)) ** 0.5, 1.0)
    if scale_up > 1.0:
        image = image.resize((max(1, int(round(width * scale_up))), max(1, int(round(height * scale_up)))), Image.NEAREST)
        width, height = image.size
    scale_down = min(MAX_SIDE / width, MAX_SIDE / height, 1.0)
    if scale_down < 1.0:
        image = image.resize((max(1, int(width * scale_down)), max(1, int(height * scale_down))), Image.BILINEAR)
    return image
