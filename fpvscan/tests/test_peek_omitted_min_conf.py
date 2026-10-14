"""Omitted auto_peek_min_conf must stay stricter than classify."""
from __future__ import annotations

from queue import Queue

from fpvscan.engine import Detection
from tests.helpers import make_engine


def _drain(q: Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_omitted_auto_peek_min_conf_is_06():
    """#36 set 0.6 explicitly so a 0.50 confirm listed but did not LOCK.
    helpers.py always passes 0.6. Dropping the key must keep that gap.
    A leftover 0 peeks every classify accept (0.45) and parks the
    receiver on a weak `?` spur for auto_peek_secs while 400 MHz–6 GHz
    waits. A leftover 1.0 never shows a picture.
    """
    events = Queue()
    eng = make_engine(events=events)
    eng.cfg["scan"]["confirm_hits"] = 1
    eng.cfg["scan"].pop("auto_peek_min_conf")

    eng._merge(Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=11,
        confidence=0.50, standard="?",
    ))
    kinds = [e["type"] for e in _drain(events)]
    assert "detection" in kinds
    assert eng.state.mode == "SWEEP"
    assert eng.state.lock_target is None

    eng._merge(Detection(
        freq_hz=1280e6, bandwidth_hz=12e6, snr_db=14,
        confidence=0.70, standard="PAL",
    ))
    assert eng.state.mode == "LOCK"
    assert eng.state.auto is True
    assert eng.state.lock_target == 1280e6
