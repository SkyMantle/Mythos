"""Tracking widens the V-sync window before counting a miss, and a hit resets lost."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp import cvbs
from fpvscan.dsp.cvbs import DecodeState, Frame


def test_tracking_retries_0_55_then_0_75_then_0_95(monkeypatch):
    """#7 asserted a 50-line PAL/NTSC miss cannot fit even ±0.95 line.
    The production loop still tries 0.55 → 0.75 → 0.95 before lost++.
    Dropping the wider windows means a V-sync 0.7 line late (USB
    hiccup, short ring) increments lost on the first try and rolls
    the picture after three such fields.
    """
    seen: list[float] = []
    hit = Frame(
        luma=np.zeros((40, 48), dtype=np.uint8),
        line_rate=15625.0, lines=40, standard="PAL", locked=True,
    )

    def spy(_v, _fs, _width, _max_lines, _state, abs_start, tol_frac=0.55, **_k):
        seen.append(float(tol_frac))
        if tol_frac < 0.70:
            return None
        return hit, float(abs_start) + 10.0

    monkeypatch.setattr(cvbs, "_attempt_tracked", spy)
    fs = 2e6
    v = np.zeros(int(fs * 0.04), dtype=np.float32)
    state = DecodeState(sign=1.0, period=fs / 15625.0, standard="PAL",
                        abs_t0=0.0, lost=2)
    frame = cvbs.decode(v, fs, width=48, state=state, abs_start=100.0)
    assert frame is hit
    assert seen == [0.55, 0.75]
    assert state.lost == 0, (
        f"a tracking hit must clear lost (was 2), got {state.lost} — "
        f"the next miss would skip tracking at lost>=3"
    )
    assert state.abs_t0 == 110.0
