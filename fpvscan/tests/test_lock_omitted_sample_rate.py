"""Omitted video.sample_rate must not restart the LOCK reader every field."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from tests.helpers import MockSource, make_engine


class _FullRing:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


def test_omitted_video_sample_rate_is_20msps():
    """Merged #2 tore LOCK down on a fractional-Hz board rate when
    cfg and src were both set. helpers.py sets 20 Msps explicitly;
    shipped YAML is 35 Msps. Dropping the key must keep 20 Msps so a
    20 Msps source (file replay, this mock) does not see
    `|fs − 35e6| > 1` and `_stop_reader` every field — the same
    DecodeState wipe #2 fixed, now from an omitted-key leftover.
    """
    src = MockSource(fs=20e6)
    eng = make_engine(src=src, idle_ms=0, capture_ms=5, afc=False)
    eng.cfg["video"].pop("sample_rate")
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6
    try:
        with patch("fpvscan.engine.cvbs.decode", return_value=None):
            eng._do_lock()
    finally:
        eng._stop_reader()
    assert src.set_sample_rate_calls == 0
    assert src.sample_rate == 20e6
