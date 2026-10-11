"""Omitted YAML picture knobs must still reach cvbs.decode with shipped defaults."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from tests.helpers import make_engine


class _FullRing:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


def test_do_lock_uses_decode_defaults_when_picture_keys_are_omitted():
    """#56 forwarded explicit sharpen=0.5 / auto_levels=True / width=80.
    The `.get` fallbacks are width=640, auto_levels=True, sharpen=0.0 —
    the decode() signature. Dropping a key and then reading it as a
    required field KeyErrors LOCK; defaulting sharpen to a leftover
    0.5 (or auto_levels to False) makes a partial config silently
    change every frame.
    """
    seen: dict = {}

    def _decode(*_a, **kw):
        seen.update(kw)
        return None

    eng = make_engine(idle_ms=0, capture_ms=5, afc=False)
    for key in ("width", "auto_levels", "sharpen"):
        eng.cfg["video"].pop(key)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6
    try:
        with patch("fpvscan.engine.cvbs.decode", side_effect=_decode):
            eng._do_lock()
    finally:
        eng._stop_reader()

    assert seen["width"] == 640
    assert seen["auto_levels"] is True
    assert seen["sharpen"] == 0.0
