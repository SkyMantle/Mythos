"""LOCK must pass YAML picture knobs into cvbs.decode, not drop them."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from tests.helpers import make_engine


class _FullRing:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


def test_do_lock_forwards_sharpen_auto_levels_and_width():
    """#54 / #18 only pinned the YAML flags. #55 exercised auto_levels
    and motion-average *inside* cvbs / Engine._acc. If `_do_lock`
    stops passing `sharpen=` / `auto_levels=` / `width=`, those
    knobs become dead config and every LOCK frame goes soft / grey
    / 640-wide regardless of what the operator set.
    """
    seen: dict = {}

    def _decode(*_a, **kw):
        seen.update(kw)
        return None

    eng = make_engine(sharpen=0.5, auto_levels=True, width=80,
                      idle_ms=0, capture_ms=5, afc=False)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6
    try:
        with patch("fpvscan.engine.cvbs.decode", side_effect=_decode):
            eng._do_lock()
    finally:
        eng._stop_reader()

    assert seen["sharpen"] == 0.5
    assert seen["auto_levels"] is True
    assert seen["width"] == 80
