"""LOCK stream encode must keep native field size; photos may stretch."""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from fpvscan.dsp.cvbs import Frame, encode


def _frame(h: int = 40, w: int = 80) -> Frame:
    return Frame(
        luma=np.full((h, w), 128, dtype=np.uint8),
        line_rate=15625.0,
        lines=h,
        standard="PAL",
        locked=True,
    )


def test_stream_encode_keeps_native_luma_size():
    """Engine passes height=None for the WS frame. Resizing here letterboxes LOCK."""
    frame = _frame(40, 80)
    blob = encode(frame, "webp", 80, height=None)
    img = Image.open(BytesIO(blob))
    assert img.size == (80, 40)
    assert blob[:4] == b"RIFF"


def test_photo_encode_stretches_to_requested_height():
    frame = _frame(40, 80)
    blob = encode(frame, "webp", 80, height=576)
    img = Image.open(BytesIO(blob))
    assert img.size == (80, 576)
