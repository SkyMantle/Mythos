"""A sweep hop must not spawn the LOCK IQ reader on the same USB handle."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_do_sweep_keeps_leftover_reader_stopped():
    """#24 Sweep *command* `_stop_reader()` so `_grab` owns the
    radio. #50 `_start_reader` stops the previous thread before
    `retune_and_read`. Peek expiry *does* `_stop_reader()` before
    `_do_sweep` runs. The missing half: an 8 MHz hop must not
    `_start_reader` / allocate `_ring`. A leftover LOCK hop
    inside `_do_sweep` (copy-paste from `_do_lock` after a
    glance) leaves `_reader_thread` racing `sync_rx` with
    `_grab` — the same USB dual-access that drops LOCK after a
    short join — and `_lock_tuned` looks live so the next
    glance skips the retune. `fixed_freq` keeps this to one
    grab; zeros IQ so occupancy/inspect stay out of the path.
    Leftover is `_reader_thread is None` / `_ring is None`
    after the hop, with only the grab's `retune_and_read`.
    """
    src = MockSource()
    src.fixed_freq = True
    src._fc = 5800e6
    eng = make_engine(src)
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    try:
        assert eng._reader_thread is None
        assert eng._ring is None
        eng._do_sweep()
        assert src.retune_calls == 1
        assert eng._reader_thread is None
        assert eng._ring is None
        assert eng._lock_tuned is None
        assert eng.state.mode == "SWEEP"
        assert eng.state.lock_target is None
        assert eng._reader_stop.is_set() is False
    finally:
        eng._stop_reader()
