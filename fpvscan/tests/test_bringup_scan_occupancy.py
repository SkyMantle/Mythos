"""bringup --scan must overlap analog channels and keep a low occupancy floor."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "bringup.py").read_text(encoding="utf-8")


def test_scan_band_step_overlaps_a_10mhz_channel():
    """A 20 MHz analog bird on the stitch of two full-fs steps is only
    half-visible in each. Engine uses YAML channel_bw; bringup hard-codes
    10 MHz. Changing this to `step = fs` is how a live VTx disappears
    from `bringup.py --scan` while LOCK on a known frequency still works.
    """
    assert "step = max(fs * 0.25, fs * 0.9 - 10e6)" in SRC
    assert "np.arange(lo_hz + fs / 2, hi_hz, step)" in SRC


def test_bringup_occupancy_stays_looser_than_the_engine_fallback():
    """Engine `_do_sweep` falls back to min_bw 4 MHz / threshold 8 dB.
    The 4.3 MHz bench VTx is visible in bringup only because this path
    uses 6 dB / 1 MHz. Unifying the two silently re-hides that bird
    during first-stand bringup.
    """
    assert "find_occupied(psd, f, fs, threshold_db=6, min_bw_hz=1e6)" in SRC
    assert "find_occupied(psd, fc, fs, threshold_db=6, min_bw_hz=1e6)" in SRC
