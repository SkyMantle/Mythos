"""INSPECT USB glitch must not count as an engine-level radio failure."""
from __future__ import annotations

from queue import Empty, Queue

import numpy as np

from fpvscan.dsp.spectrum import Occupancy
from tests.helpers import MockSource, make_engine


def _drain(q: Queue) -> list:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except Empty:
            return out


def test_inspect_source_error_is_swallowed_when_debug_is_off():
    """#7 only checked debug_candidates=True emits a candidate reason.
    The production path is debug off (shipped YAML). Re-raising here
    bubbles into _run's USB fail counter; eight inspect glitches on a
    noisy hop would kill the server instead of skipping that blob.
    """
    class Boom(MockSource):
        def retune_and_read(self, hz, n):
            raise RuntimeError("usb glitch")

    events = Queue()
    eng = make_engine(src=Boom(), events=events)
    assert not eng.cfg["scan"].get("debug_candidates")
    occ = Occupancy(5800e6, 10e6, -20.0, 12.0)
    eng._inspect(np.zeros(8, dtype=np.complex64), 5800e6, 20e6, occ)
    assert _drain(events) == []
    assert eng.state.detections == {}
    assert eng.state.mode == "SWEEP"
