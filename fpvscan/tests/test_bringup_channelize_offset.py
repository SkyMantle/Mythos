"""bringup channelize offset is occupancy minus the grab LO, not 0."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "bringup.py").read_text(encoding="utf-8")


def test_bringup_channelize_offset_is_occupancy_minus_lo():
    """grab() sits on `fc` (the -f argument, or the strongest --scan
    hit). Occupancy center is often a few MHz off that LO. Offset 0
    demodulates the LO leak / DC hump; `top.center_hz - fc` is the
    same contract as Engine._inspect after it retunes. Both the
    classify grab and the decode grab must use it — classify-only
    still produces a PNG of the wrong slice.
    """
    assert SRC.count("top.center_hz - fc") == 2
    assert "demod.channelize(x, fs, top.center_hz - fc" in SRC
