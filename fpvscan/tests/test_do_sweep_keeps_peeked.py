"""A sweep hop must not reset auto-peek cooldown."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_sweep_keeps_leftover_peeked():
    """#64 Sweep *command* keeps `_peeked` (Clear is the wipe).
    #70 `_maybe_peek` keeps other MHz; #68 expiry keeps the
    table. The hop is the other half: `_do_sweep` already nulls
    `_lock_tuned` (#68) and keeps `detections` (#76). A leftover
    `_peeked.clear()` on each 8 MHz step forgets that this bird
    already glanced, so the next confirm_hits cycle LOCK-peeks
    the same analog hit instead of waiting `auto_peek_cooldown_s`
    — the 400 MHz–6 GHz walk never resumes. The command/hop
    split is the same as the hit list: Сканувати keeps cooldown;
    only Очистити drops it.
    """
    eng = make_engine()
    eng.src.fixed_freq = True
    eng.src._fc = 5800e6
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    eng._peeked[5800] = 1.0
    eng._peeked[1280] = 2.0
    try:
        eng._do_sweep()
        assert eng.src.retune_calls == 1
        assert eng._lock_tuned is None
        assert eng._peeked[5800] == 1.0
        assert eng._peeked[1280] == 2.0
        assert eng._peeked == {5800: 1.0, 1280: 2.0}
    finally:
        eng._stop_reader()
