#!/usr/bin/env python3
"""Draw Chinese/Unicode text on OpenCV BGR images using Pillow."""

import os
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-DemiLight.ttc',
]

_font_cache = {}


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    if size in _font_cache:
        return _font_cache[size]
    for path in FONT_CANDIDATES:
        if os.path.isfile(path):
            _font_cache[size] = ImageFont.truetype(path, size)
            return _font_cache[size]
    _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def bgr_to_rgb(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    b, g, r = color
    return (r, g, b)


def draw_text(
    bgr_image: np.ndarray,
    text: str,
    org: Tuple[int, int],
    size: int = 18,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Draw UTF-8 text on a BGR numpy image in-place style (returns new image)."""
    if not text:
        return bgr_image
    x, y = org
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text((x, y), text, font=_load_font(size), fill=bgr_to_rgb(color))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def render_panel(
    width: int,
    height: int,
    lines: list,
) -> np.ndarray:
    """
    Render a dark panel with text lines.

    Each line: (text, y, size, color_bgr) or (text, y, size, color_bgr, bold_extra)
    """
    panel_rgb = Image.new('RGB', (width, height), (28, 28, 28))
    draw = ImageDraw.Draw(panel_rgb)
    for item in lines:
        text, y, size, color = item[:4]
        draw.text((16, y), text, font=_load_font(size), fill=bgr_to_rgb(color))
    panel_bgr = cv2.cvtColor(np.array(panel_rgb), cv2.COLOR_RGB2BGR)
    return panel_bgr
