"""Auto-peek must not eat a pending Photo."""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_maybe_peek_keeps_pending_photo():
    """#69 lock / Sweep / Clear / expiry keep `_snap`. #65 snapshot
    in SWEEP sets the flag without a mode==LOCK guard — that is
    exactly when `_maybe_peek` fires (confirm during the walk).
    Clearing `_snap` on the glance would drop a Photo armed on the
    hit: the 8 s LOCK would decode a picture but never write the
    WebP, and the operator would mash Photo after they already
    asked for a still. The flag is consumed only when a LOCK
    frame exists (#9).
    """
    eng = make_engine()
    eng.state.mode = "SWEEP"
    eng._snap = True
    det = Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=16,
        standard="NTSC", confidence=0.9, channel="F4",
        band="5G8", hits=2,
    )
    eng._maybe_peek(det)
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
    assert eng.state.auto is True
    assert eng._snap is True
    assert eng.events.qsize() == 1
