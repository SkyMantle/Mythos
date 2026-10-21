"""File replay must not scale recorded IQ when Engine sets gain."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fpvscan.sdr.file import FileSource


def test_file_set_gain_does_not_scale_recorded_iq(tmp_path: Path):
    """#66 `set_sample_rate` and #68 `set_center_freq` are no-ops
    so sidecar 40 Msps / 5800 MHz survive `_run`. Gain is the
    third boot write: `_run` calls `set_gain(gain_db)` then
    `_apply_bias_tee_gain` subtracts the 15 dB LNA offset.
    Honoring that on a .cf32 would scale a capture already taken
    at the stand gain — classify would see a dimmer 16.2 kHz
    line and a real VTx would look like Wi-Fi. Replay must leave
    the recorded amplitude alone.
    """
    p = tmp_path / "vtx.cf32"
    iq = np.array([1+0j, 0.5+0.5j, -0.25+0.75j, 0.8-0.2j],
                  dtype=np.complex64)
    iq.tofile(p)
    (tmp_path / "vtx.json").write_text(
        '{"center_hz": 5800000000.0, "sample_rate": 40000000.0}',
        encoding="utf-8",
    )
    src = FileSource(p, loop=True)
    src.open()
    try:
        before = src.read(4)
        src.set_gain(0.0)
        after_zero = src.read(4)
        src.set_gain(60.0)
        after_hot = src.read(4)
        np.testing.assert_array_equal(before, iq)
        np.testing.assert_array_equal(after_zero, iq)
        np.testing.assert_array_equal(after_hot, iq)
        assert src.center_freq == 5800e6
        assert src.sample_rate == 40e6
    finally:
        src.close()
