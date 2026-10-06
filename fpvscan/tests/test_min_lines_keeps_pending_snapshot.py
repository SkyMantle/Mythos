"""Photo must still fire after a short-field reject."""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import numpy as np

from tests.helpers import make_engine


@dataclass
class _ShortField:
    luma: np.ndarray
    line_rate: float = 15734.0
    lines: int = 10
    standard: str = "NTSC"
    locked: bool = True


def test_min_lines_reject_does_not_consume_pending_snapshot():
    """`_snap` is consumed only inside `if frame is not None`, after
    `min_lines` may have nulled the field. #9 saved a WebP when a
    frame existed; #49 kept `_snap` across a short-ring miss. A
    232-line NTSC field under a stale 250-line floor (the #4
    picture-quality fix) would swallow the operator's Photo click
    and never retry.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._snap = True
    short = _ShortField(luma=np.zeros((10, 64), dtype=np.uint8))
    with patch("fpvscan.engine.cvbs.decode", return_value=short):
        try:
            eng._do_lock()
            assert eng._snap is True
        finally:
            eng._stop_reader()
