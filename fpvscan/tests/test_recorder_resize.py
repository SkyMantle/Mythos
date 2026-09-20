"""ffmpeg rawvideo stdin is a fixed WxH. A 287-line PAL field pushed
into a 288-high recorder must be resized, not written as-is — a short
write desyncs every later frame and the mp4 is snow.
"""
from __future__ import annotations

import numpy as np

from fpvscan.recorder import _resize_nearest


def test_resize_nearest_hits_the_requested_shape():
    src = np.arange(12, dtype=np.uint8).reshape(3, 4)
    out = _resize_nearest(src, 6, 8)
    assert out.shape == (6, 8)
    assert out.dtype == np.uint8
    # corner mapping: last dst row/col come from last src row/col
    assert int(out[0, 0]) == int(src[0, 0])
    assert int(out[-1, -1]) == int(src[-1, -1])


def test_resize_nearest_is_identity_on_matching_size():
    src = np.arange(20, dtype=np.uint8).reshape(4, 5)
    out = _resize_nearest(src, 4, 5)
    np.testing.assert_array_equal(out, src)
