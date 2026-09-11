"""Shipped YAML values that override engine defaults — silent if lost."""
from __future__ import annotations

from pathlib import Path

from fpvscan import config


def _cfg() -> dict:
    return config.load(Path(__file__).resolve().parents[1] / "config.yaml")


def test_idle_ms_is_10_not_engine_default_120():
    """Engine default idle_ms=120 hitchs LOCK to ~8 fps. Shipped 10 ms
    is the Pi-5 budget; losing it makes the console look stalled.
    """
    v = _cfg()["video"]
    assert v["idle_ms"] == 10
    assert v["capture_ms"] == 80
    assert v["ring_seconds"] == 0.5


def test_auto_peek_secs_is_5_not_engine_default_8():
    """Engine default auto_peek_secs=8. Shipped 5 is the operator dwell.
    Losing it doubles time spent on the first hit while others wait.
    """
    sc = _cfg()["scan"]
    assert sc["auto_peek"] is True
    assert sc["auto_peek_secs"] == 5
    assert sc["auto_peek_cooldown_s"] == 60
    assert sc["priority_bands"] is False


def test_motion_average_and_sharpen_are_on():
    """average=0 / sharpen=0 in code. Shipped 3 / 0.5 is the picture-
    quality path from the LOCK-pipeline work; dropping them returns the
    noisy, soft field that #4 existed to fix.
    """
    v = _cfg()["video"]
    assert v["average"] == 3
    assert v["sharpen"] == 0.5
    assert v["motion_thresh"] == 24.0


def test_fft_size_and_console_port():
    sc, web = _cfg()["scan"], _cfg()["web"]
    assert sc["fft_size"] == 8192
    assert web["port"] == 8080
    assert _cfg()["sdr"]["bias_tee"] is True
