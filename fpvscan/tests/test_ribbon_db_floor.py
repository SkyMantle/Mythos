"""Ribbon height is a dBFS window, not a 0–255 or 0–1 linear scale."""
from __future__ import annotations

from pathlib import Path


HTML = (Path(__file__).resolve().parents[1]
        / "fpvscan" / "web" / "static" / "index.html").read_text(encoding="utf-8")


def test_ribbon_maps_minus_100_to_minus_30_over_70_db():
    """pushSpectrum paints `u = (ribData[x] + 100) / 70`. That is
    −100…−30 dBFS → 0…1. Mapping as if bins were 0–255 (or dividing
    by 100) saturates the 5.8 GHz tick on noise and hides a −40 dB
    VTx — the operator thinks the band is empty.
    """
    assert "(ribData[x] + 100) / 70" in HTML
    assert "-100..-30" in HTML


def test_unseen_bins_start_below_the_display_floor():
    """ribData is −120, 20 dB under the −100 floor. Init or Clear to
    0 paints every unseen megahertz as full-scale and the coverage
    bar lies about a finished sweep.
    """
    assert "new Float32Array(RW).fill(-120)" in HTML
    assert "ribData.fill(-120)" in HTML
    assert "seen.fill(0)" in HTML
