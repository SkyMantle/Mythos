"""Active-line window must start after the sync tip, not at sample 0."""
from __future__ import annotations

from fpvscan.dsp.cvbs import STD_GEOM, SYNC_US


def test_active_region_starts_after_the_sync_pulse():
    """STD_GEOM[std][:2] are fractions of the line used as active
    luma. PR #17 only checks 0 < a0 < a1 < 1, which still allows
    a0≈0 — the 4.7 µs sync tip (and most of the back porch) is then
    sampled into every LOCK row as a black bar that auto-levels
    stretch into a washed-out frame. PAL line 64 µs / NTSC 63.556 µs;
    a0 * T_line must clear SYNC_US. The right edge must also leave
    ~1 µs of front porch so the next H-sync is not in the picture.
    """
    pal_t = 64.0e-6
    ntsc_t = 63.556e-6
    pal_a0, pal_a1, _ = STD_GEOM["PAL"]
    ntsc_a0, ntsc_a1, _ = STD_GEOM["NTSC"]
    assert pal_a0 * pal_t > SYNC_US
    assert ntsc_a0 * ntsc_t > SYNC_US
    assert (1.0 - pal_a1) * pal_t >= 1.0e-6
    assert (1.0 - ntsc_a1) * ntsc_t >= 1.0e-6
    # Unknown geometry must also skip the sync tip — '?' is what
    # LOCK uses when classify will not commit PAL vs NTSC.
    unk_a0, unk_a1, _ = STD_GEOM["?"]
    assert unk_a0 * 64.0e-6 > SYNC_US
    assert (1.0 - unk_a1) * 64.0e-6 >= 1.0e-6
