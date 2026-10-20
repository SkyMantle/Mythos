"""File replay must keep the sidecar LO after set_center_freq."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fpvscan.sdr.file import FileSource


def test_file_set_center_freq_does_not_change_recorded_lo(tmp_path: Path):
    """#66 `set_sample_rate` is a no-op so `_run` copies the sidecar
    40 Msps, not YAML 35. #38 `_sweep_plan` is `[src.center_freq]`.
    #26 out-of-band `retune_and_read` still returns the recorded IQ
    and leaves `_fc` alone — but that override never calls
    `set_center_freq`. The in-band path is the one the comment
    leaves empty: `abs(hz-_fc) <= fs/2` is `pass`, same as out of
    band. Honoring an in-band hop (5790 next to a 5800 sidecar)
    would rewrite the recorded center, `_sweep_plan` would inspect
    the wrong LO, and occupancy bins would be computed as if the
    .cf32 were taken at 5790. Replay must ignore the request.
    """
    p = tmp_path / "vtx.cf32"
    np.zeros(256, dtype=np.complex64).tofile(p)
    (tmp_path / "vtx.json").write_text(
        '{"center_hz": 5800000000.0, "sample_rate": 40000000.0}',
        encoding="utf-8",
    )
    src = FileSource(p, loop=True)
    src.open()
    try:
        assert src.center_freq == 5800e6
        src.set_center_freq(5790e6)          # in-band (10 MHz < fs/2)
        assert src.center_freq == 5800e6
        src.set_center_freq(1280e6)          # well outside the capture
        assert src.center_freq == 5800e6
        assert src.sample_rate == 40e6
    finally:
        src.close()
