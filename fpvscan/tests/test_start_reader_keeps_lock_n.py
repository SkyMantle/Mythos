"""A LOCK hop must not re-arm the first-field waterfall FFT."""
from __future__ import annotations

from tests.helpers import make_engine


def test_start_reader_keeps_leftover_lock_n():
    """#71 lock / Sweep / Clear / peek / expiry keep leftover
    `_lock_n` so commands do not reset cadence. The hop itself
    is the other half: `_start_reader` reseeds DecodeState and
    `_lock_dec` after `retune_and_read` (#50) and clears
    leftover `_reader_err` / `_reader_stop` (#73 / #72). A
    leftover `_lock_n = 0` on every hop makes the next field
    `1 % 8 == 1` and spends idle_ms on a 4096×4 FFT the
    operator did not ask for. Leftover skip after a hop that
    stopped mid-cycle is current main.
    """
    eng = make_engine()
    eng._lock_n = 7
    try:
        eng._start_reader(5800e6, 20e6, 0.05)
        assert eng._lock_n == 7
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
        assert eng._ring is not None
        assert eng._ring.filled > 0
    finally:
        eng._stop_reader()
