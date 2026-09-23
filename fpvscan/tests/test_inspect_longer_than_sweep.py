"""INSPECT exists because the sweep dwell is too short to see 15.7 kHz."""
from __future__ import annotations

from pathlib import Path

from fpvscan.config import load
from fpvscan.engine import Engine


CFG = load(Path(__file__).resolve().parents[1] / "config.yaml")


def test_shipped_inspect_grab_is_many_line_periods_past_the_sweep_fft():
    """Sweep grab is fft_size × averages (1.87 ms at 35 Msps / 8192 / 8).
    A PAL line is 64 µs — that dwell holds ~29 lines, not enough for
    classify_video's 8192-after-decimation floor. inspect_ms=25 is ~390
    lines. Collapsing inspect onto the sweep buffer (or cutting
    inspect_ms toward 2) makes every analog bird look like Wi-Fi.
    """
    sc = CFG["scan"]
    fs = float(sc["sample_rate"])
    sweep_n = int(sc["fft_size"]) * int(sc["averages"])
    inspect_n = int(fs * float(sc["inspect_ms"]) / 1000)
    assert sweep_n == 8192 * 8
    assert sc["inspect_ms"] == 25
    assert inspect_n > 8 * sweep_n
    assert inspect_n >= int(fs * 0.02)
    # classify_video rejects after decimation to ~400 kHz if < 8192
    dec = max(1, int(fs / 400e3))
    assert inspect_n // dec >= 8192
    assert sweep_n // dec < 8192
