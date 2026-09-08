"""bladeRF constants that, if swapped, decode TX or enable AGC on LOCK."""
from pathlib import Path

from fpvscan.sdr.bladerf import (
    ERR_IO,
    ERR_TIMEOUT,
    FORMAT_SC16_Q11,
    FORMAT_SC16_Q11_META,
    GAIN_MGC,
    GAIN_SLOWATTACK_AGC,
    META_STATUS_OVERRUN,
    SC16_SCALE,
)


SRC = (Path(__file__).resolve().parents[1]
       / "fpvscan" / "sdr" / "bladerf.py").read_text(encoding="utf-8")


def test_manual_gain_is_mgc_not_agc():
    """AGC on a CVBS discriminator walks the sync tips every burst."""
    assert GAIN_MGC == 1
    assert GAIN_SLOWATTACK_AGC == 3
    assert "GAIN_SLOWATTACK_AGC if self.agc else GAIN_MGC" in SRC


def test_overrun_and_usb_error_codes():
    """read() treats -5 as recover-the-device and bit0 as a dropped USB buffer."""
    assert ERR_IO == -5
    assert ERR_TIMEOUT == -6
    assert META_STATUS_OVERRUN == 1
    assert "meta.status & META_STATUS_OVERRUN" in SRC


def test_adc_clip_threshold_is_below_q11_full_scale():
    """Full-scale Q11 is 2048. Waiting until 2048 hides saturation until LOCK dies."""
    assert SC16_SCALE == 2048.0
    assert "np.abs(raw) > 2000" in SRC
    assert FORMAT_SC16_Q11 == 0
    assert FORMAT_SC16_Q11_META == 1
