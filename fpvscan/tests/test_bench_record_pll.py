"""Operator scripts: Msps flags and the PLL dump before a capture."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bench_retune_rate_flag_is_msps():
    """`python scripts/bench_retune.py -s 40` is 40 Msps. Treating -s as
    Hz asks the xA4 for a 40 Hz stream; the sweep-time estimate becomes
    nonsense and quick-tune priming walks a 40 Hz grid.
    """
    src = (ROOT / "scripts" / "bench_retune.py").read_text(encoding="utf-8")
    assert "fs = a.rate * 1e6" in src
    assert 'help="Мвідл/с"' in src
    assert "default=40.0" in src
    assert "arange(400e6 + fs / 2, 6000e6, step)" in src


def test_record_dumps_pll_before_capture_and_warns_on_adc_clip():
    """First 50 ms after set_center_freq is the LO transient. Writing it
    into the .cf32 makes FileSource replay look empty at the start of
    every clip. clip_frac > 0.1% is the same saturation the console
    paints red — a recording taken in clip is not a usable golden.
    """
    src = (ROOT / "scripts" / "record.py").read_text(encoding="utf-8")
    assert "src.read(int(fs * 0.05))" in src
    assert "if src.clip_frac > 0.001:" in src
    assert "write_capture(a.out, iq, fc, fs, a.gain, a.note)" in src
