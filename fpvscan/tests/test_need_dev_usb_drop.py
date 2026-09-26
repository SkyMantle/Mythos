"""A USB drop must raise a radio error, not a null-pointer crash."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "fpvscan" / "sdr" / "bladerf.py").read_text(encoding="utf-8")


def test_need_dev_mentions_usb_unplug():
    """_recover leaves _dev null after the board disappears. The next
    libbladeRF call with a zero handle is an access violation that
    kills the process with no journal line. _need_dev must raise
    BladeRFError naming USB so Engine's 8-strike retry can notice
    and the operator replugs instead of chasing a segfault.
    """
    start = SRC.index("def _need_dev")
    end = SRC.index("def _recover")
    body = SRC[start:end]
    assert "BladeRFError" in body
    assert "USB" in body
    assert "не відкритий" in body
