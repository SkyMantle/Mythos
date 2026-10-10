"""TBC must keep the predicted start when the only edge is outside ±6 %."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp import cvbs


def test_genlock_keeps_prediction_when_edge_is_outside_window():
    """#55 followed a 1-sample/line error *inside* ±max_corr_frac.
    Equalizing pulses and a neighbor H-sync sit ~0.5 period away —
    well past the 0.06 window (12 samples at period=200). Without the
    clip, one noise spike yanks a line onto the next period and the
    rest of the field slants. No prior TBC test asserted the reject.
    """
    period = 200.0
    n_lines = 10
    t0 = 80.0
    n = int(t0 + n_lines * period + 80)
    v = np.full(n, 0.70, dtype=np.float32)
    predicted = t0 + np.arange(n_lines, dtype=np.float64) * period
    for p in predicted:
        i = int(round(p + 40.0))
        v[i:i + 10] = 0.0

    starts = cvbs._genlock_starts(v, t0, period, n_lines, thr=0.18)

    assert np.max(np.abs(starts - predicted)) < 1.0
    assert np.min(np.abs(starts - (predicted + 40.0))) > 30.0
