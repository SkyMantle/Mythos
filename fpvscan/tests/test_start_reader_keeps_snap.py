"""A LOCK hop must not eat a pending Photo."""
from __future__ import annotations

from tests.helpers import make_engine


def test_start_reader_keeps_leftover_snap():
    """#69 Sweep / lock / Clear / expiry keep leftover
    `_snap`; #70 `_maybe_peek` keeps it so a Photo armed
    on the hit survives the glance. #65 snapshot in SWEEP
    sets the flag. The hop is the missing half: `_start_reader`
    joins the old IQ thread and reseeds DecodeState (#50).
    A leftover `_snap = False` on every retune drops a
    still the operator armed before the channel change —
    `_do_lock` only consumes the flag when a frame exists
    (#9). Leftover pending Photo after a hop is current main.
    """
    eng = make_engine()
    eng._snap = True
    try:
        eng._start_reader(5800e6, 20e6, 0.05)
        assert eng._snap is True
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
        assert eng._ring is not None
    finally:
        eng._stop_reader()
