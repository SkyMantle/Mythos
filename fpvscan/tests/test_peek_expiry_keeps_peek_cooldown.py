"""Auto-peek expiry must keep the per-MHz cooldown table."""
from __future__ import annotations

import time

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_peek_expiry_keeps_peeked_so_cooldown_still_holds():
    """#64 Sweep / lock keep `_peeked`. Clear (#37) is the wipe.
    #53 cooldown expires at 60 s on a live table. Expiry (#50)
    already zeros AFC and returns to SWEEP; wiping `_peeked` on
    that timer would forget the bird we just watched, and the
    next confirm_hits cycle would `_maybe_peek` the same MHz
    immediately instead of waiting `auto_peek_cooldown_s`.
    """
    eng = make_engine()
    now = time.time()
    eng._peeked[5800] = now - 10.0
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = True
    eng.state.auto_until = now - 1.0
    try:
        eng._do_lock()
        assert eng.state.mode == "SWEEP"
        assert 5800 in eng._peeked
        assert eng._peeked[5800] == now - 10.0
        det = Detection(
            freq_hz=5800e6, bandwidth_hz=12e6, snr_db=16,
            standard="NTSC", confidence=0.9, channel="F4",
            band="5G8", hits=2,
        )
        eng._maybe_peek(det)
        assert eng.state.mode == "SWEEP"
        assert eng.state.lock_target is None
    finally:
        eng._stop_reader()
