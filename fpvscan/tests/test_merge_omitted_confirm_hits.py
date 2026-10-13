"""Omitted confirm_hits must hide a one-pass noise flash from the console."""
from __future__ import annotations

from queue import Queue

from fpvscan.engine import Detection
from tests.helpers import make_engine


def _drain(q: Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_omitted_confirm_hits_is_two():
    """#14 pinned shipped YAML confirm_hits≥2. helpers.py sets 2
    explicitly; #36 lowered it to 1 so a peek test could fire on the
    first merge. Dropping the key must still require two hits. A
    leftover 1 lists every sweep-dwell flash; a leftover 0 paints the
    whole 400 MHz–6 GHz passband as analog.
    """
    events = Queue()
    eng = make_engine(events=events)
    eng.cfg["scan"].pop("confirm_hits")
    eng.cfg["scan"]["auto_peek"] = False
    det = Detection(
        freq_hz=5800e6, bandwidth_hz=10e6, snr_db=12.0, confidence=0.9,
    )
    eng._merge(det)
    kinds = [e["type"] for e in _drain(events)]
    assert "detection" not in kinds
    assert eng.snapshot()["detections"] == []
    assert eng.state.detections[5800].hits == 1

    eng._merge(Detection(
        freq_hz=5800e6, bandwidth_hz=10e6, snr_db=12.0, confidence=0.9,
    ))
    kinds = [e["type"] for e in _drain(events)]
    assert "detection" in kinds
    listed = eng.snapshot()["detections"]
    assert len(listed) == 1
    assert listed[0]["hits"] == 2
    assert listed[0]["freq_hz"] == 5800e6
