"""Headless detection lines are MHz — the same scale as --lock."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "headless.py").read_text(encoding="utf-8")


def test_headless_detection_print_divides_hz_by_1e6():
    """`--lock` is MHz (`freq_hz=a.lock * 1e6`). The detection line
    prints `freq_hz/1e6`. Dropping the divide reports
    5800000000.0 МГц next to a LOCK command that wants 5800, and
    the operator tunes 8.6 THz off. Same scale as the status line
    (`tuned_hz/1e6`).
    """
    assert "ЗНАЙДЕНО {d['freq_hz']/1e6:.1f} МГц" in SRC
    assert "s['tuned_hz']/1e6" in SRC
    assert "eng.command(\"lock\", freq_hz=a.lock * 1e6)" in SRC
