"""bringup.py must blame the USB cable and Pi PSU when overflows are non-zero."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "bringup.py").read_text(encoding="utf-8")


def test_overflows_name_usb3_and_five_amp_supply():
    """A non-zero `overflows` after a successful decode is a dropped
    USB packet, not a CVBS bug. If the hint drops USB 3.0 / 5 В /
    5 А, the operator reinstalls libbladeRF on a Pi that is already
    brown-outing on a 3 A charger.
    """
    assert "hasattr(src, \"overflows\")" in SRC
    assert "зривів потоку USB" in SRC
    assert "USB 3.0" in SRC
    assert "5 В" in SRC
    assert "5 А" in SRC
