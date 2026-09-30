"""File replay must not invent IQ when the capture is done."""
from __future__ import annotations

import json

import numpy as np
import pytest

from fpvscan.sdr.file import FileSource


def test_empty_capture_raises_instead_of_yielding_zeros(tmp_path):
    """A 0-byte .cf32 (failed record, full disk) would otherwise
    open as a silent source. Sweep then reports 'no occupancy' and
    the operator raises gain on a capture that never had a VTx.
    """
    p = tmp_path / "empty.cf32"
    p.write_bytes(b"")
    src = FileSource(p)
    with pytest.raises(IOError, match="Порожній"):
        src.open()


def test_noloop_raises_eof_instead_of_rewinding(tmp_path):
    """`loop=False` is how a one-shot replay finishes. Rewinding
    here would splice the tail onto the head and confirm the same
    bird twice — `confirm_hits=2` then lights a ghost detection
    on a recording that only ever had one pass.
    """
    p = tmp_path / "short.cf32"
    np.zeros(16, dtype=np.complex64).tofile(p)
    p.with_suffix(".json").write_text(json.dumps({
        "center_hz": 5800e6,
        "sample_rate": 40e6,
    }), encoding="utf-8")
    src = FileSource(p, loop=False)
    src.open()
    got = src.read(8)
    assert got.shape == (8,)
    with pytest.raises(EOFError):
        src.read(16)
