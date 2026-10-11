"""After the first lock, picture knobs must still reach the tracked renderer."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp import cvbs
from fpvscan.dsp.cvbs import DecodeState, Frame


def test_decode_forwards_sharpen_auto_levels_and_width_to_tracked(monkeypatch):
    """#56 asserted Engine._do_lock passes knobs into cvbs.decode.
    decode() then prefers `_attempt_tracked` whenever state.period is
    set. Dropping the kwargs on that call makes sharpen / auto_levels
    / width dead after field 1 — every live LOCK frame goes soft /
    grey / 640-wide while the YAML still looks correct.
    """
    seen: dict = {}
    hit = Frame(
        luma=np.zeros((40, 80), dtype=np.uint8),
        line_rate=15625.0, lines=40, standard="PAL", locked=True,
    )

    def spy(_v, _fs, width, _max_lines, _state, abs_start, tol_frac=0.55,
            auto_levels=True, sharpen=0.0):
        seen.update({
            "width": int(width),
            "auto_levels": bool(auto_levels),
            "sharpen": float(sharpen),
            "tol": float(tol_frac),
        })
        return hit, float(abs_start) + 10.0

    monkeypatch.setattr(cvbs, "_attempt_tracked", spy)
    fs = 2e6
    v = np.zeros(int(fs * 0.04), dtype=np.float32)
    state = DecodeState(sign=1.0, period=fs / 15625.0, standard="PAL",
                        abs_t0=0.0, lost=0)
    frame = cvbs.decode(v, fs, width=80, state=state, abs_start=100.0,
                        auto_levels=False, sharpen=0.5)
    assert frame is hit
    assert seen["width"] == 80
    assert seen["auto_levels"] is False
    assert seen["sharpen"] == 0.5
    assert seen["tol"] == 0.55
