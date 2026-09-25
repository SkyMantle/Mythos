"""A failed USB recover must name the hardware, not a generic lib error."""
from __future__ import annotations

import inspect

from fpvscan.sdr import bladerf as B


def test_recover_exhausted_hint_names_usb3_cable_and_hub():
    """After 6 retries `_recover` raises. «sync_rx: LIBUSB_ERROR_IO»
    looks like a software bug. The message must still point at the
    USB 3.0 port, a short cable, and a powered hub — those are the
    three stand failures that drop the xA4 mid-sweep.
    """
    src = inspect.getsource(B.BladeRF._recover)
    assert "USB 3.0" in src
    assert "без хабів" in src
    assert "кабель" in src
    assert "retries" in src
    assert "0.4 * (i + 1)" in src
