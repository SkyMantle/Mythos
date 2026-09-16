"""Shipped inspect stays on the FIR channelizer; headless must not drop events."""
from __future__ import annotations

from pathlib import Path

from fpvscan import config


def test_shipped_yaml_does_not_enable_fast_inspect_channelizer():
    """Engine defaults fast_channelizer to False, so INSPECT uses FIR
    (less aliasing, slower). Putting `fast_channelizer: true` in the
    shipped YAML would make classify see boxcar images of neighbors
    and confirm the wrong VTx. LOCK may stay fast; inspect must not.
    """
    cfg = config.load(Path(__file__).resolve().parents[1] / "config.yaml")
    assert "fast_channelizer" not in cfg["scan"]


def test_headless_event_queue_is_large_enough_for_debug():
    """run.py bounds the web queue at 64 so a disconnected browser
    cannot ram the Pi. headless.py is the 'is it the radio or the
    web?' tool and must keep spectra + candidates + frames. Copying
    maxsize=64 here silently drops detections during inspect.
    """
    text = (Path(__file__).resolve().parents[1] / "scripts" / "headless.py"
            ).read_text(encoding="utf-8")
    assert "maxsize=2000" in text
