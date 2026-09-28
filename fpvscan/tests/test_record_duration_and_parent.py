"""record.py duration is seconds, size is complex64, and -o may nest."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "record.py").read_text(encoding="utf-8")


def test_record_duration_is_seconds_and_counts_complex64_bytes():
    """`-d 3` is 3 seconds at `-s 40` → 960 MB. Treating -d as ms
    writes 120 kB that FileSource then loops as a 3 ms flash; dropping
    the `* 8` print makes a 3 s grab look like 120 MB and the Pi
    fills the SD card. Default stays 3.0 so a forgotten -d does not
    dump tens of seconds.
    """
    assert "n = int(fs * a.dur)" in SRC
    assert 'default=3.0' in SRC
    assert "n*8/1e6" in SRC
    assert 'help="секунд"' in SRC


def test_record_creates_the_parent_of_minus_o():
    """README's `-o caps/vtx_f4.cf32` must not require a hand-made
    caps/ on a fresh clone. write_capture does not mkdir; record.py
    has to, or the 3 s grab dies after the radio is already open.
    """
    assert "Path(a.out).parent.mkdir(parents=True, exist_ok=True)" in SRC
    assert "write_capture(a.out, iq, fc, fs, a.gain, a.note)" in SRC
