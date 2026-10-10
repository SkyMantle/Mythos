"""Aperture correction is a horizontal 1-2-1 unsharp mask, not a 2-D blur."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp import cvbs


def test_sharpen_overshoots_a_vertical_edge_and_leaves_rows_alone():
    """#5 only checked that mean |Δx| rose on a full PAL decode.
    The kernel is `0.25, 0.5, 0.25` on x, then
    `luma + sharpen*(luma-blur)`. A 0.40→0.70 step must grow an
    undershoot/overshoot pair; a horizontal stripe (constant in x)
    must not change. A vertical blur here is the 'muddy text'
    regression #4's aperture correction was written to avoid.
    """
    period = 100.0
    n_lines, width = 12, 32
    a0, a1 = 0.10, 0.90
    v = np.full(int(n_lines * period + 40), 0.30, dtype=np.float32)
    starts = np.arange(n_lines, dtype=np.float64) * period + 5.0
    mid = width // 2
    for s in starts:
        i0, i1 = int(s + a0 * period), int(s + a1 * period)
        row = np.full(i1 - i0, 0.40, dtype=np.float32)
        split = (i1 - i0) // 2
        row[split:] = 0.70
        v[i0:i1] = row

    plain = cvbs._render(v, starts, period, a0, a1, width,
                         auto_levels=False, sharpen=0.0)
    sharp = cvbs._render(v, starts, period, a0, a1, width,
                         auto_levels=False, sharpen=0.8)

    # Vertical edge: dark side drops, bright side rises.
    assert int(sharp[0, mid - 1]) < int(plain[0, mid - 1])
    assert int(sharp[0, mid]) > int(plain[0, mid])
    assert int(sharp[0, mid]) - int(sharp[0, mid - 1]) > \
        int(plain[0, mid]) - int(plain[0, mid - 1])

    # Horizontal stripe (same value on a row) is unchanged — kernel is 1-D.
    stripe_v = np.full_like(v, 0.50)
    stripe_plain = cvbs._render(stripe_v, starts, period, a0, a1, width,
                                auto_levels=False, sharpen=0.0)
    stripe_sharp = cvbs._render(stripe_v, starts, period, a0, a1, width,
                                auto_levels=False, sharpen=0.8)
    assert np.array_equal(stripe_plain, stripe_sharp)
