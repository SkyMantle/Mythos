"""Auto-peek expiry must not eat a pending Photo."""
from __future__ import annotations

import time

from tests.helpers import make_engine


def test_peek_expiry_keeps_pending_photo():
    """#65 snapshot in SWEEP only sets `_snap` and emits no error;
    #49 a short-ring miss leaves the flag so `/api/snapshot` is
    not eaten. Expiry (#50) already zeros AFC and (#68) keeps the
    hit list / cooldown. Clearing `_snap` on the 8 s timer would
    drop a Photo the operator pressed during the glance — the
    next peek/lock would not write the WebP, and they would hit
    the button again after the picture had already vanished.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = True
    eng.state.auto_until = time.time() - 1.0
    eng._snap = True
    try:
        eng._do_lock()
        assert eng.state.mode == "SWEEP"
        assert eng._snap is True
    finally:
        eng._stop_reader()
