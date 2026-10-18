"""A partial capture sidecar must not invent 5.8 GHz or a 0 Hz sample rate."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fpvscan.sdr.file import FileSource


def test_sidecar_omitted_sample_rate_is_40msps(tmp_path: Path):
    """#14 no sidecar at all: center 0 / 40 Msps. A hand-written
    `.json` with only `center_hz` (truncated sidecar, operator
    note) must still open at 40 Msps — leftover 0 divides
    `duration_s` / FileSource realtime sleep by zero and looks
    like a hung replay — and must keep the written LO, not
    invent F4.
    """
    p = tmp_path / "partial.cf32"
    np.zeros(256, dtype=np.complex64).tofile(p)
    (tmp_path / "partial.json").write_text(
        '{"center_hz": 1280000000.0}', encoding="utf-8"
    )
    src = FileSource(p, loop=True)
    src.open()
    try:
        assert src.center_freq == 1280e6
        assert src.sample_rate == 40e6
    finally:
        src.close()


def test_sidecar_omitted_center_is_zero(tmp_path: Path):
    """Only `sample_rate` in the sidecar: must not invent 5800 MHz.
    Replay would then inspect the wrong LO and look empty.
    """
    p = tmp_path / "rate_only.cf32"
    np.zeros(256, dtype=np.complex64).tofile(p)
    (tmp_path / "rate_only.json").write_text(
        '{"sample_rate": 20000000.0}', encoding="utf-8"
    )
    src = FileSource(p, loop=True)
    src.open()
    try:
        assert src.center_freq == 0.0
        assert src.sample_rate == 20e6
    finally:
        src.close()
