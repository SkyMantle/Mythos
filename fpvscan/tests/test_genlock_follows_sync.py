"""Per-line genlock must track the real H-sync edge, not t0 + i·period."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp import cvbs


def test_genlock_corrects_one_sample_per_line_period_error():
    """A 0.5 % period error (200 vs 201 samples) accumulates to 11
    samples by line 12 — enough to slant every vertical on a 288-line
    field. `_genlock_starts` searches ±6 % of period around the
    prediction; those starts must land on the real falling edge, not
    the drifted global grid. Reverting to `t0 + i*period` is the
    pre-#4 TBC regression.
    """
    period = 200.0
    actual = 201.0
    n_lines = 12
    t0 = 80.0
    n = int(t0 + n_lines * actual + 80)
    v = np.full(n, 0.70, dtype=np.float32)
    edges = t0 + np.arange(n_lines) * actual
    for e in edges:
        i = int(round(e))
        v[i:i + 10] = 0.0

    starts = cvbs._genlock_starts(v, t0, period, n_lines, thr=0.18)
    predicted = t0 + np.arange(n_lines) * period

    assert abs(starts[-1] - edges[-1]) < 1.0
    assert abs(predicted[-1] - edges[-1]) > 8.0
    assert abs(starts[-1] - edges[-1]) < 0.2 * abs(predicted[-1] - edges[-1])
