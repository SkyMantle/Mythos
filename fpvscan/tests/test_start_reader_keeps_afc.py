"""A LOCK hop / 1 Hz fs restart must not drop accumulated digital AFC."""
from __future__ import annotations

from tests.helpers import make_engine


def test_start_reader_keeps_leftover_afc():
    """#21 lock() *command* zeros `_afc` on a channel switch so
    F4's residual does not mix onto R5. #50 peek expiry zeros it
    before Sweep. AFC recenter (#46) zeros `_afc` *after* calling
    `_start_reader` (the hop retunes to `f+off+_afc`, then the
    caller clears the digital offset). The 1 Hz fs-mismatch
    restart (#2) calls the hop *without* the command — leftover
    `_afc = 0` there dumps a 400 kHz walk the channelizer already
    paid for, the picture jumps, and SNR drops until AFC climbs
    back. The hop already reseeds DecodeState (#50) and must not
    write the digital offset.
    """
    eng = make_engine()
    eng._afc = 400_000.0
    try:
        eng._start_reader(5800e6, 20e6, 0.05)
        assert eng._afc == 400_000.0
        assert eng._afc != 0.0
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
