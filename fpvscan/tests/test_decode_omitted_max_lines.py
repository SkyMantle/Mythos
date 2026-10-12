"""Engine relies on decode()'s one-PAL-field max_lines default."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from fpvscan.dsp import cvbs
from tests.helpers import make_engine


class _FullRing:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


def test_decode_default_max_lines_is_one_pal_field(monkeypatch):
    """#55 / #57 clamped when max_lines=288 was passed explicitly into
    `_attempt` / `_attempt_tracked`. Engine never forwards the kwarg, so
    the signature default is the live LOCK ceiling. Changing it to 576
    paints the next V-sync as a black band on every frame.
    """
    seen: dict = {}

    def spy(_v, _fs, width, max_lines, auto_levels=True, sharpen=0.0):
        seen["max_lines"] = int(max_lines)
        seen["width"] = int(width)
        return 0.0, None, None

    monkeypatch.setattr(cvbs, "_attempt", spy)
    fs = 2e6
    v = np.zeros(int(fs * 0.04), dtype=np.float32)
    assert cvbs.decode(v, fs) is None
    assert seen["max_lines"] == 288
    assert seen["width"] == 640


def test_do_lock_does_not_override_decode_max_lines():
    """Passing 576 (or any two-field cap) from Engine would bypass the
    decode() default that #55 / #57 proved is required. Omitting the
    kwarg is the contract: decode() keeps one PAL field.
    """
    seen: dict = {}

    def _decode(*_a, **kw):
        seen.update(kw)
        return None

    eng = make_engine(idle_ms=0, capture_ms=5, afc=False)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6
    try:
        with patch("fpvscan.engine.cvbs.decode", side_effect=_decode):
            eng._do_lock()
    finally:
        eng._stop_reader()
    assert "max_lines" not in seen
    assert seen.get("max_lines", 288) == 288
