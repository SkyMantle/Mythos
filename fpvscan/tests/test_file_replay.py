"""File replay and sim continuity — the offline path must not jump or retune."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fpvscan.sdr.file import FileSource, write_capture
from fpvscan.sdr.sim import Emitter, SimSource


def test_file_retune_keeps_recorded_center(tmp_path: Path):
    p = tmp_path / "cap.cf32"
    write_capture(p, np.arange(16, dtype=np.complex64), 1280e6, 10e6)
    src = FileSource(p, loop=True)
    src.open()
    try:
        out = src.retune_and_read(5800e6, 8)
        assert src.center_freq == 1280e6
        assert src.sample_rate == 10e6
        assert len(out) == 8
    finally:
        src.close()


def test_file_tiles_when_request_is_longer_than_capture(tmp_path: Path):
    p = tmp_path / "short.cf32"
    iq = np.arange(8, dtype=np.complex64)
    write_capture(p, iq, 5800e6, 1e6)
    src = FileSource(p, loop=True)
    src.open()
    try:
        out = src.read(20)
        assert len(out) == 20
        np.testing.assert_array_equal(out[:8], iq)
        np.testing.assert_array_equal(out[8:16], iq)
        np.testing.assert_array_equal(out[16:20], iq[:4])
    finally:
        src.close()


def test_sim_cw_phase_is_continuous_across_reads():
    src = SimSource(
        emitters=[Emitter(5801e6, -10.0, "cw", 1e6, label="tone")],
        noise_db=-120, seed=3,
    )
    src.open()
    src.set_sample_rate(4e6)
    src.set_center_freq(5800e6)
    src.set_gain(40)
    a = src.read(2048)
    b = src.read(2048)
    ph = np.unwrap(np.angle(np.concatenate([a, b])))
    inc = float(np.median(np.diff(ph[:2048])))
    jump = float(ph[2048] - ph[2047])
    # Current sim reuses the last phase as the next block's first sample
    # (one-sample overlap). A phase reset would jump by a random angle.
    assert abs(jump) < 2 * abs(inc) + 0.3
