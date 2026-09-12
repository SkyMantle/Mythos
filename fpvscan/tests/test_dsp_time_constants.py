"""DSP windows that silently drop NTSC or smear the picture."""
from __future__ import annotations

import inspect
from pathlib import Path

from fpvscan.dsp import cvbs, demod

CVBS_SRC = Path(cvbs.__file__).read_text(encoding="utf-8")


def test_deemphasis_tau_is_video_half_microsecond_not_audio():
    """Broadcast-FM audio is 50 µs. That constant on CVBS turns every
    edge into a smear and LOCK text becomes unreadable. Analog FPV
    video deemphasis is 0.5 µs.
    """
    tau = inspect.signature(demod.deemphasis).parameters["tau"].default
    assert tau == 0.5e-6
    assert tau < 5e-6


def test_blind_decode_line_window_covers_pal_and_ntsc():
    """_attempt rejects line_rate outside 14–17.5 kHz before it names
    a standard. A PAL-only window (15.5–15.7 kHz) drops NTSC at 15734
    and LOCK stays on snow for half the 5.8 GHz grid.
    """
    assert "14000 < line_rate < 17500" in CVBS_SRC
    assert 14000 < 15625 < 17500
    assert 14000 < 15734 < 17500
    # PAL–NTSC spacing is 109 Hz — both must sit in the same gate.
    assert (15734 - 15625) < (17500 - 14000)
