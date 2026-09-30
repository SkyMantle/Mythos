"""`--file` must demodulate at the capture's fc/fs, not `-s`/`-f`."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "bringup.py").read_text(encoding="utf-8")


def test_bringup_file_takes_fc_fs_from_the_capture():
    """`-s`/`-f` are BladeRF knobs (MHz). Applying them on `--file`
    demodulates a 40 Msps / 5800 MHz capture at 20 Msps / 0 Hz:
    PAL falls to ~7.8 kHz, classify_video rejects a real analog
    bird, and the operator thinks the .cf32 is empty.
    """
    assert "if a.file:" in SRC
    assert "fc, fs = src.center_freq, src.sample_rate" in SRC
    assert "fs = a.rate * 1e6" in SRC
    file_branch = SRC.split("if a.file:", 1)[1].split("else:", 1)[0]
    assert "src.center_freq" in file_branch
    assert "src.sample_rate" in file_branch
    assert "a.rate" not in file_branch
    assert "a.freq" not in file_branch


def test_bringup_file_classify_does_not_dump_pll():
    """`grab()` discards 20 ms after every retune. On a radio that
    is the LO transient. On a 50–80 ms file it eats the analog
    content the classify/decode path was about to measure. The
    file branch reads `n_long` in place.
    """
    assert ("x = grab(src, fc, fs, n_long) if not a.file "
            "else src.read(n_long)") in SRC


def test_bringup_forwards_lib_to_bladerf():
    """doctor.py's `--lib` path is how Windows finds bladeRF.dll.
    Dropping `lib_path=a.lib` here sends the operator back to
    'бібліотеку не знайдено' after doctor just succeeded.
    """
    assert "BladeRF(device=a.device, lib_path=a.lib)" in SRC
