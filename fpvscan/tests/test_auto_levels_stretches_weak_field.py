"""auto_levels must stretch a weak field; the 0.30–1.0 map leaves it grey."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp import cvbs


def test_auto_levels_uses_percentiles_not_the_fixed_sync_map():
    """A VTx that only swings 0.35–0.55 (weak SNR, or deemphasis
    leftover) maps to ~18–91 on the fixed `(samp-0.30)/0.70` scale —
    the washed-out grey #4 replaced with 2nd/99th percentiles.
    `auto_levels: false` is already the YAML pin in #54; this is the
    stretch itself. A constant field (hi-lo < 1e-3) still falls back
    to 0.30/1.0, so the checker is the case that must not stay grey.
    """
    period = 100.0
    n_lines = 16
    width = 32
    a0, a1 = 0.10, 0.90
    v = np.full(int(n_lines * period + 50), 0.30, dtype=np.float32)
    starts = np.arange(n_lines, dtype=np.float64) * period + 5.0
    for s in starts:
        i0, i1 = int(s + a0 * period), int(s + a1 * period)
        row = np.where(np.arange(i1 - i0) % 2 == 0, 0.35, 0.55)
        v[i0:i1] = row.astype(np.float32)

    auto = cvbs._render(v, starts, period, a0, a1, width,
                        auto_levels=True, sharpen=0.0)
    fixed = cvbs._render(v, starts, period, a0, a1, width,
                         auto_levels=False, sharpen=0.0)

    assert int(auto.max()) - int(auto.min()) > 200
    assert int(fixed.max()) < 120
    assert int(auto.max()) > int(fixed.max()) + 100
