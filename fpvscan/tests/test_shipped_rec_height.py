"""Shipped recording height is one analog field, not a full frame."""
from __future__ import annotations

from pathlib import Path

from fpvscan import config


def test_shipped_rec_height_is_one_pal_field():
    """LOCK emits one field per capture (PAL ~288 lines, NTSC ~240).
    Recording at 576 would upsample every field, double x264 work on
    the Pi 5, and still not reconstruct a frame — clips drop FPS and
    the operator thinks the radio stalled.
    """
    cfg = config.load(Path(__file__).resolve().parents[1] / "config.yaml")
    assert cfg["video"]["rec_height"] == 288
    assert cfg["video"]["width"] == 640
