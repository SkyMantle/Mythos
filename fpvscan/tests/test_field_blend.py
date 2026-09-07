from __future__ import annotations

import numpy as np

from fpvscan.dsp.field_blend import blend_same_field


def test_opposite_fields_are_not_woven() -> None:
    even = np.full((8, 8), 40.0, dtype=np.float32)
    odd = np.full((8, 8), 200.0, dtype=np.float32)
    acc, parity = blend_same_field(even, 0, odd, 1, strength=5.0, motion_thresh=24.0)
    assert parity == 1
    assert np.allclose(acc, odd)


def test_same_field_blends_toward_current() -> None:
    prev = np.full((8, 8), 0.0, dtype=np.float32)
    cur = np.full((8, 8), 10.0, dtype=np.float32)
    acc, parity = blend_same_field(prev, 0, cur, 0, strength=5.0, motion_thresh=24.0)
    assert parity == 0
    assert 0.0 < float(acc.mean()) < 10.0
