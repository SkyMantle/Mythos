"""File replay must keep constructor loop/realtime, not sidecar lies."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fpvscan.sdr.file import FileSource, write_capture


def test_file_open_ignores_sidecar_loop_false_and_realtime(tmp_path: Path):
    """#43 factory omits YAML `realtime`; #63/#64 constructor
    `loop=True` is the bringup / factory default. `write_capture`
    does not record those keys, but a hand-written sidecar from
    another tool might. Honoring `loop: false` EOFError's a short
    .cf32 mid-inspect (same ghost as leftover constructor False).
    Honoring `realtime: true` sleeps n/fs per grab — inspect/sweep
    then crawl and the 16.2 kHz line is still in the IQ, just
    never classified before the operator gives up. Constructor
    wins: factory/bringup stay looping and unpaced.
    """
    p = tmp_path / "vtx.cf32"
    iq = (np.arange(256, dtype=np.float32)
          + 1j * np.linspace(-0.5, 0.5, 256)).astype(np.complex64)
    write_capture(p, iq, center_hz=5800e6, sample_rate=40e6,
                  gain_db=30.0, note="stand")
    (tmp_path / "vtx.json").write_text(
        '{"center_hz": 5800000000.0, "sample_rate": 40000000.0, '
        '"loop": false, "realtime": true}',
        encoding="utf-8",
    )
    src = FileSource(p, loop=True, realtime=False)
    src.open()
    try:
        assert src.loop is True
        assert src.realtime is False
        first = src.read(200)
        second = src.read(200)
        np.testing.assert_array_equal(first, iq[:200])
        np.testing.assert_array_equal(second, iq[:200])
        assert src.center_freq == 5800e6
        assert src.sample_rate == 40e6
    finally:
        src.close()


def test_file_open_ignores_sidecar_loop_true_when_constructor_false(tmp_path: Path):
    """#44 `loop=False` is the one-shot finish. A sidecar leftover
    `loop: true` (or a `write_capture` that later grows that key)
    would rewind at EOF, splice tail onto head, and `confirm_hits=2`
    would light the same bird twice on a recording that only ever
    had one pass.
    """
    p = tmp_path / "oneshot.cf32"
    iq = np.arange(16, dtype=np.float32).astype(np.complex64)
    iq.tofile(p)
    (tmp_path / "oneshot.json").write_text(
        '{"center_hz": 1280000000.0, "sample_rate": 40000000.0, '
        '"loop": true, "realtime": false}',
        encoding="utf-8",
    )
    src = FileSource(p, loop=False, realtime=False)
    src.open()
    try:
        assert src.loop is False
        got = src.read(8)
        np.testing.assert_array_equal(got, iq[:8])
        with pytest.raises(EOFError):
            src.read(16)
    finally:
        src.close()
