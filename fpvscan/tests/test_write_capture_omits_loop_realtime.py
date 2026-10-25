"""write_capture must not persist loop/realtime into the sidecar."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fpvscan.sdr.file import FileSource, write_capture


def test_write_capture_sidecar_omits_loop_and_realtime(tmp_path: Path):
    """#31 sidecar keys match FileSource (`center_hz`, `sample_rate`,
    `complex64`). #72 `open()` ignores a hand-written `loop` /
    `realtime`. The producer side is the other half: if
    `write_capture` started recording `loop: true` (or
    `realtime: true` from a paced grab), a later "fix" that
    honors those keys on `open()` would rewind a one-shot
    bringup (`loop=False`) and `confirm_hits=2` would light the
    same bird twice, or sleep n/fs per inspect so classify never
    runs. The stand dump must leave constructor loop/realtime
    as the only source of truth.
    """
    p = tmp_path / "stand.cf32"
    iq = np.arange(32, dtype=np.float32).astype(np.complex64)
    write_capture(p, iq, center_hz=5800e6, sample_rate=40e6,
                  gain_db=30.0, note="stand")
    meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    assert "loop" not in meta
    assert "realtime" not in meta
    assert meta["center_hz"] == 5800e6
    assert meta["sample_rate"] == 40e6
    assert meta["format"] == "complex64"

    src = FileSource(p, loop=False, realtime=False)
    src.open()
    try:
        assert src.loop is False
        assert src.realtime is False
        got = src.read(16)
        np.testing.assert_array_equal(got, iq[:16])
        with pytest.raises(EOFError):
            src.read(32)
    finally:
        src.close()
