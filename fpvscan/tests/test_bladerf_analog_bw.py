"""bladeRF analog filter must track sample rate or the channel aliases."""
from __future__ import annotations

import inspect
from pathlib import Path

from fpvscan.sdr import bladerf

SRC = Path(bladerf.__file__).read_text(encoding="utf-8")


def test_analog_bandwidth_is_90_percent_of_sample_rate():
    """set_sample_rate also programs the RFIC LPF at fs*bandwidth_ratio.
    1.0*fs lets the anti-alias filter pass the image; a tight 0.5*fs
    slices a 20 MHz analog FPV channel at 35 Msps.
    """
    sig = inspect.signature(bladerf.BladeRF.__init__)
    assert sig.parameters["bandwidth_ratio"].default == 0.9
    assert "int(self._fs * self.bandwidth_ratio)" in SRC


def test_usb_buffer_size_is_a_multiple_of_1024():
    """libbladeRF rejects buffer_size that is not a multiple of 1024
    and the first LOCK read then raises — looks like a dead radio.
    """
    sig = inspect.signature(bladerf.BladeRF.__init__)
    buf = sig.parameters["buffer_size"].default
    assert buf % 1024 == 0
    assert buf >= 4096


def test_sc16_formats_are_plain_then_meta():
    """sync_config fmt=1 without use_meta is a metadata buffer on a
    plain stream (or the reverse). Either way reads return garbage IQ.
    """
    assert bladerf.FORMAT_SC16_Q11 == 0
    assert bladerf.FORMAT_SC16_Q11_META == 1
    assert "FORMAT_SC16_Q11_META if self.use_meta else FORMAT_SC16_Q11" in SRC
