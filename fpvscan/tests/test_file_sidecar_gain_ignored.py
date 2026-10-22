"""File replay must not scale recorded IQ from sidecar gain_db."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fpvscan.sdr.file import FileSource, write_capture


def test_file_open_ignores_sidecar_gain_db(tmp_path: Path):
    """#69 `set_gain` is a no-op so `_run` / `_apply_bias_tee_gain`
    cannot dim a .cf32. `write_capture` still records `gain_db`
    as stand metadata. Honoring that on `open()` — scaling like
    SimSource `(gain-40)/40`, or applying the 15 dB LNA offset
    a second time — would change IQ amplitude of a capture already
    taken at stand gain. classify would miss the 16.2 kHz line
    and a real VTx would look like Wi-Fi. Replay must leave the
    recorded samples alone.
    """
    p = tmp_path / "vtx.cf32"
    iq = np.array([1+0j, 0.5+0.5j, -0.25+0.75j, 0.8-0.2j],
                  dtype=np.complex64)
    write_capture(p, iq, center_hz=5800e6, sample_rate=40e6,
                  gain_db=60.0, note="stand")
    src = FileSource(p, loop=True)
    src.open()
    try:
        got = src.read(4)
        np.testing.assert_array_equal(got, iq)
        assert src.center_freq == 5800e6
        assert src.sample_rate == 40e6
        src.set_gain(0.0)
        np.testing.assert_array_equal(src.read(4), iq)
    finally:
        src.close()
