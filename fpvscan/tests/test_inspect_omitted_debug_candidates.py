"""Omitted debug_candidates must not flood the web queue on a reject."""
from __future__ import annotations

from queue import Empty, Queue
from unittest.mock import patch

import numpy as np

from fpvscan.dsp.demod import VideoScore
from fpvscan.dsp.spectrum import Occupancy
from tests.helpers import make_engine

_REJECT = VideoScore(False, 15625.0, "?", 0.2, reason="wifi-like")


def _drain(q: Queue) -> list:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except Empty:
            return out


def test_omitted_debug_candidates_does_not_emit_on_reject():
    """#7 / #9 emitted a candidate with debug_candidates=True.
    #54 swallowed a source error when debug was off. The classify
    reject (Wi-Fi occupier) is the hot sweep path. helpers.py and
    shipped YAML omit the key (falsy). A leftover True emits a
    candidate per blob into the 64-slot web queue and evicts LOCK
    frames while 400 MHz–6 GHz is still walking.
    """
    events = Queue()
    eng = make_engine(events=events)
    assert "debug_candidates" not in eng.cfg["scan"]
    occ = Occupancy(2437e6, 8e6, -25.0, 8.0)
    with patch(
        "fpvscan.engine.demod.classify_video", return_value=_REJECT,
    ):
        eng._inspect(np.zeros(8, dtype=np.complex64), 2440e6, 20e6, occ)
    kinds = [e["type"] for e in _drain(events)]
    assert "candidate" not in kinds
    assert "detection" not in kinds
    assert eng.state.detections == {}
