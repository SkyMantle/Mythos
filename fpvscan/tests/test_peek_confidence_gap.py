"""auto_peek_min_conf is stricter than classify min_confidence — on purpose."""
from __future__ import annotations

from queue import Queue

from fpvscan.engine import Detection
from tests.helpers import make_engine


def _drain(q: Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_mid_confidence_hit_is_listed_but_does_not_steal_the_sweep():
    """Shipped floors: classify accepts at 0.45, auto-peek waits for 0.6.
    A 0.50 `?` confirm is a real bird the operator should see, but it
    must not auto-LOCK — otherwise a weak spur parks the receiver for
    auto_peek_secs and the rest of 400 MHz–6 GHz waits.
    """
    events = Queue()
    eng = make_engine(events=events)
    eng.cfg["scan"]["confirm_hits"] = 1
    eng.cfg["scan"]["auto_peek"] = True
    eng.cfg["scan"]["auto_peek_min_conf"] = 0.6
    eng._merge(Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=11,
        confidence=0.50, standard="?",
    ))
    kinds = [e["type"] for e in _drain(events)]
    assert "detection" in kinds
    assert eng.state.mode == "SWEEP"
    assert eng.state.lock_target is None
    assert eng.state.auto is False


def test_confident_hit_still_auto_peeks():
    """The 0.6 gate must not be a broken peek path."""
    events = Queue()
    eng = make_engine(events=events)
    eng.cfg["scan"]["confirm_hits"] = 1
    eng.cfg["scan"]["auto_peek"] = True
    eng.cfg["scan"]["auto_peek_min_conf"] = 0.6
    eng._merge(Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=14,
        confidence=0.70, standard="PAL",
    ))
    assert eng.state.mode == "LOCK"
    assert eng.state.auto is True
    assert eng.state.lock_target == 5800e6
