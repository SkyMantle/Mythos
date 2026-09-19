"""Ribbon bins go through json.dumps. Unrounded floats bloat the WS frame."""
from __future__ import annotations

import json

import numpy as np

from fpvscan.dsp.spectrum import downsample_for_display


def test_downsample_rounds_to_one_decimal_and_is_json():
    """pump() json.dumps the spectrum. A raw float32 dump is huge and
    can emit NaN, which the console JSON.parse then rejects — the
    ribbon freezes while LOCK still runs.
    """
    psd = np.linspace(-87.123, -11.987, 2048, dtype=np.float32)
    psd[100] = 5.04
    out = downsample_for_display(psd, target=128)
    assert len(out) == 128
    assert all(isinstance(v, float) for v in out)
    assert all(round(v, 1) == v for v in out)
    assert max(out) == 5.0
    json.dumps(out)
