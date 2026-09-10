"""Shipped YAML must keep scan/LOCK on the same rate and give INSPECT enough airtime."""
from __future__ import annotations

from pathlib import Path

from fpvscan.config import load


def _cfg() -> dict:
    return load(Path(__file__).resolve().parents[1] / "config.yaml")


def test_scan_and_video_sample_rates_match():
    """SWEEP and LOCK sharing one radio: a split here makes LOCK restart the
    IQ reader every frame (fs mismatch > 1 Hz) and the picture never holds.
    """
    cfg = _cfg()
    assert cfg["scan"]["sample_rate"] == cfg["video"]["sample_rate"]
    assert cfg["scan"]["sample_rate"] >= 20e6


def test_inspect_ms_is_long_enough_for_line_rate():
    """classify_video needs tens of PAL/NTSC periods. Sub-20 ms inspect
    confirms Wi-Fi skirts as analog or rejects a live VTx at random.
    """
    assert _cfg()["scan"]["inspect_ms"] >= 20


def test_scan_channel_bw_is_ten_mhz():
    """Sweep step = max(0.25 fs, 0.9 fs − channel_bw/2). Doubling this
    to the engine fallback of 20 MHz opens seams that hide a 10 MHz VTx.
    """
    assert _cfg()["scan"]["channel_bw_hz"] == 10e6


def test_lo_offset_is_zero_at_shipped_sample_rate():
    """Comment in YAML: 8 MHz offset needs 61.44 Msps. At 35 Msps an
    8 MHz offset plus a 10 MHz channel does not fit the 0.45·fs gate,
    so LOCK would silently drop the offset — worse, a non-zero value
    that then gets zeroed is a trap. Shipped config keeps it off.
    """
    cfg = _cfg()
    off = cfg["video"]["lo_offset_hz"]
    bw = cfg["video"]["channel_bw_hz"]
    fs = cfg["video"]["sample_rate"]
    assert off == 0.0
    assert abs(off) + bw / 2 <= fs * 0.45
