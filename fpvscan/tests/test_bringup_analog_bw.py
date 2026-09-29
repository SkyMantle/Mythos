"""bringup must print analog BW after set_sample_rate, from src.bandwidth."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "bringup.py").read_text(encoding="utf-8")


def test_bringup_prints_analog_bw_after_setting_rate():
    """bladeRF analog filter is 0.9·fs and is written inside
    set_sample_rate. Printing sample_rate (or printing before the
    write) hides a 40 Msps / 10 MHz filter mismatch: the occupancy
    looks 8 MHz wide, classify fails, and the operator raises gain.
    `src.bandwidth` is the analog figure of merit; `_bw` is 0.0
    until set_sample_rate runs.
    """
    rate = SRC.index("src.set_sample_rate(fs)")
    bw = SRC.index("смуга ФНЧ")
    assert bw > rate
    assert "src.bandwidth/1e6" in SRC[bw:bw + 80]
