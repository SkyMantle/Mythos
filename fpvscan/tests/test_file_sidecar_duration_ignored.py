"""File replay must not recompute fs or reread IQ from sidecar lies."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fpvscan.sdr.file import FileSource, write_capture


def test_file_open_ignores_sidecar_duration_samples_format(tmp_path: Path):
    """#65 omitted `sample_rate` is 40 Msps; #70 ignores sidecar
    `gain_db`. `write_capture` still records `duration_s` /
    `samples` / `format` as stand metadata. Honoring a stale
    `duration_s` as `size / duration` would poison fs (256
    samples / 10 s → 25.6 Hz) and classify would miss the
    16.2 kHz line. Honoring `samples` would truncate the .cf32.
    Honoring `format: sc16` would reinterpret complex64 bytes
    and the 16.2 kHz line would look like noise. Replay must
    keep the written rate and the recorded samples.
    """
    p = tmp_path / "vtx.cf32"
    iq = (np.arange(256, dtype=np.float32)
          + 1j * np.linspace(-0.5, 0.5, 256)).astype(np.complex64)
    write_capture(p, iq, center_hz=5800e6, sample_rate=40e6,
                  gain_db=30.0, note="stand")
    sidecar = tmp_path / "vtx.json"
    sidecar.write_text(
        '{"center_hz": 5800000000.0, "sample_rate": 40000000.0, '
        '"gain_db": 30.0, "samples": 4, "duration_s": 10.0, '
        '"format": "sc16", "note": "stale"}',
        encoding="utf-8",
    )
    src = FileSource(p, loop=True)
    src.open()
    try:
        got = src.read(256)
        np.testing.assert_array_equal(got, iq)
        assert src.center_freq == 5800e6
        assert src.sample_rate == 40e6
        assert src.sample_rate != iq.size / 10.0
    finally:
        src.close()


def test_file_open_omitted_rate_ignores_duration_s(tmp_path: Path):
    """#65 only `center_hz` still opens at 40 Msps. A sidecar
    that also carries `duration_s` (write_capture always does)
    must not invent `size / duration` when `sample_rate` is
    missing — leftover 25.6 Hz mis-bins occupancy and the
    16.2 kHz line falls out of classify.
    """
    p = tmp_path / "partial.cf32"
    iq = np.zeros(256, dtype=np.complex64)
    iq.tofile(p)
    (tmp_path / "partial.json").write_text(
        '{"center_hz": 1280000000.0, "duration_s": 10.0, '
        '"samples": 256, "format": "complex64"}',
        encoding="utf-8",
    )
    src = FileSource(p, loop=True)
    src.open()
    try:
        assert src.center_freq == 1280e6
        assert src.sample_rate == 40e6
        assert src.sample_rate != iq.size / 10.0
    finally:
        src.close()
