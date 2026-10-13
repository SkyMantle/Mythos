"""Omitted stream_quality must keep a cheap live WebP, not a photo encode."""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import numpy as np

from tests.helpers import make_engine


class _FullRing:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


@dataclass
class _Field:
    luma: np.ndarray
    line_rate: float = 15625.0
    lines: int = 232
    standard: str = "NTSC"
    locked: bool = True


def test_omitted_stream_quality_is_75():
    """#14 / #41 pinned native-height encode and photo quality 90.
    #29 pinned shipped YAML 60. The `.get` fallback is 75. Defaulting
    to 0 writes an empty WebP on every LOCK frame; a leftover 90 (the
    photo path) blows the Pi 5 idle_ms budget and the USB reader
    overflows look like a dead VTx. helpers.py already omits the key.
    """
    seen: dict = {}

    def fake_encode(_frame, fmt="webp", quality=80, height=480):
        seen["fmt"] = fmt
        seen["quality"] = int(quality)
        seen["height"] = height
        return b"img"

    eng = make_engine(idle_ms=0, capture_ms=5, afc=False)
    assert "stream_quality" not in eng.cfg["video"]
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6
    field = _Field(luma=np.full((232, 64), 80, dtype=np.uint8))
    try:
        with patch("fpvscan.engine.cvbs.decode", return_value=field), \
             patch("fpvscan.engine.cvbs.encode", side_effect=fake_encode):
            eng._do_lock()
    finally:
        eng._stop_reader()
    assert seen["fmt"] == "webp"
    assert seen["quality"] == 75
    assert seen["height"] is None
