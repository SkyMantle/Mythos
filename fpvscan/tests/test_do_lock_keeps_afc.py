"""A LOCK field must not dump accumulated digital AFC."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_lock_keeps_leftover_afc():
    """#21 lock() *command* zeros `_afc` on a channel switch so
    F4's residual does not mix onto R5. #50 peek expiry zeros
    it before Sweep. AFC recenter (#46) zeros `_afc` *after*
    calling `_start_reader`. #80 the hop keeps leftover `_afc`
    on AFC recenter / the 1 Hz fs restart. `_do_lock` is the
    missing half: a field with `afc: False` (and zeros IQ, so
    no err walk) must not `_afc = 0`. A leftover wipe on every
    decode dumps a 400 kHz walk the channelizer already paid
    for, the picture jumps, and SNR drops until AFC climbs
    back. The field already starts the reader when `_ring` is
    None and must not write the digital offset.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._afc = 400_000.0
    try:
        eng._do_lock()
        assert eng._afc == 400_000.0
        assert eng._afc != 0.0
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
