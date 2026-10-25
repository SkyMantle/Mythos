"""A new LOCK reader must not inherit a leftover USB error."""
from __future__ import annotations

from tests.helpers import make_engine


def test_start_reader_clears_leftover_reader_err():
    """#50 `_start_reader` joins before `retune_and_read` and
    reseeds DecodeState after the hop. #72 disarms leftover
    `_reader_stop`. The third leftover is `_reader_err`:
    `_reader_loop` stores a USB/timeout exception and exits;
    `_stop_reader` (Sweep / expiry) does not clear it. Reusing
    that exception without `= None` makes the next `_do_lock`
    re-raise after a successful re-arm — LOCK looks dead after
    Сканувати even though the new ring already has a prime.
    `_do_lock` still consults leftover err *before* start
    (bugfix track); start itself must disarm what it inherits.
    """
    eng = make_engine()
    leftover = RuntimeError("USB drop")
    eng._reader_err = leftover
    try:
        eng._start_reader(5800e6, 20e6, 0.05)
        assert eng._reader_err is None
        assert eng._reader_thread is not None
        assert eng._lock_tuned == 5800e6
        assert eng._ring is not None
        assert eng._ring.filled > 0
    finally:
        eng._stop_reader()
