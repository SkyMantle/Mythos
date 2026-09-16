"""Detection payload is what the hit list and click-to-lock read."""
from __future__ import annotations

import inspect
from queue import Empty, Queue

from fpvscan.bands import band_of, nearest_channel
from fpvscan.engine import Detection, Engine
from tests.helpers import make_engine


def _drain(q: Queue) -> list:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except Empty:
            return out


def test_confirmed_detection_has_fields_the_console_renders():
    """renderHits reads freq_hz / channel / band / standard / snr_db /
    bandwidth_hz, then lock() POSTs freq_hz. Dropping any of those
    from asdict() leaves a blank row or sends undefined into /api/lock.
    """
    events = Queue()
    eng = make_engine(events=events)
    eng.cfg["scan"]["confirm_hits"] = 1
    eng.cfg["scan"]["auto_peek"] = False
    det = Detection(
        freq_hz=5800e6,
        bandwidth_hz=12e6,
        snr_db=14.2,
        standard="NTSC",
        confidence=0.8,
        channel=nearest_channel(5800e6),
        band=band_of(5800e6),
    )
    eng._merge(det)
    emitted = [e for e in _drain(events) if e["type"] == "detection"]
    assert len(emitted) == 1
    data = emitted[0]["data"]
    for key in ("freq_hz", "bandwidth_hz", "snr_db", "standard",
                "channel", "band", "confidence", "hits"):
        assert key in data, f"detection payload missing {key}"
    assert data["freq_hz"] == 5800e6
    assert data["channel"] == "F4"
    assert data["band"] == "5G8"
    assert data["standard"] == "NTSC"
    assert data["snr_db"] == 14.2
    assert data["bandwidth_hz"] == 12e6


def test_inspect_labels_hits_with_channel_and_band():
    """_inspect must call nearest_channel / band_of. A hit at 3480 MHz
    with channel accidentally set from the 5.8 grid shows as R-something
    and the operator locks the wrong bird.
    """
    src = inspect.getsource(Engine._inspect)
    assert "nearest_channel(" in src
    assert "band_of(" in src


def test_auto_peek_notice_prints_mhz_not_hz():
    """The notice is the only confirmation that auto-peek went to the
    right bird. Printing 3480000000 МГц looks like a garbage lock.
    """
    events = Queue()
    eng = make_engine(events=events)
    eng.cfg["scan"]["auto_peek"] = True
    eng.cfg["scan"]["auto_peek_min_conf"] = 0.0
    eng._maybe_peek(Detection(
        freq_hz=3480e6, bandwidth_hz=10e6, snr_db=10, confidence=0.9,
    ))
    notices = [e for e in _drain(events) if e["type"] == "notice"]
    assert notices, "auto-peek must emit a notice"
    text = notices[0]["data"]["text"]
    assert "3480" in text
    assert "3480000000" not in text
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 3480e6
