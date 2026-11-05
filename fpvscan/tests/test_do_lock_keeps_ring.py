"""A LOCK field must not replace the IQ ring while the reader is up."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_lock_keeps_leftover_ring_on_later_field():
    """#50 `_start_reader` *does* allocate a new `IQRingBuffer` on a
    channel change (and the 1 Hz fs restart). #74 hop keeps
    `_lock_tuned` so that gate stays closed. The missing half: a
    later field on the same bird must not `_ring = None` or
    `_ring = IQRingBuffer(...)`. A leftover wipe on every decode
    orphans the reader thread (it writes into the old ring, or
    exits when `_ring is None`) and the next `snapshot(n)` is a
    fresh empty buffer — `len(iq) < n`, the field returns before
    decode, and the console freezes on the last picture. First
    field starts the reader (hop *does* allocate); leftover is
    the same object, still filling, after that.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        eng._do_lock()
        leftover = eng._ring
        filled = leftover.filled
        assert leftover is not None
        assert filled > 0
        eng._do_lock()
        assert eng._ring is leftover
        assert leftover.filled >= filled
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
