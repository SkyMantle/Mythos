"""record.py must demand -f in MHz before opening the radio."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "record.py").read_text(encoding="utf-8")


def test_record_freq_flag_is_required_mhz():
    """`python scripts/record.py` with no -f used to be easy to
    'fix' by giving --freq a leftover default. Opening the xA4
    on whatever Hz the last process left behind writes a 960 MB
    file of the wrong band, and FileSource replay then looks
    like 'the detector is broken'. -f stays required, and the
    value is megahertz (`* 1e6`), same as bringup.
    """
    assert 'required=True' in SRC
    assert 'help="МГц"' in SRC
    assert "fc, fs = a.freq * 1e6, a.rate * 1e6" in SRC
    assert "write_capture(a.out, iq, fc, fs, a.gain, a.note)" in SRC
