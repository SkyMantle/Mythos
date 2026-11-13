"""A sweep hop must not replace or raise the IQ-reader stop Event."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_do_sweep_keeps_leftover_reader_stop_identity():
    """#89 a later LOCK field keeps leftover `_reader_stop` identity
    and clear so `_reader_loop` still watches the Event `_stop_reader`
    will set. #90 a hop keeps the reader *stopped* (must not
    `_start_reader` — that races `_grab` on the same USB handle).
    #92 hop keeps engine `_stop` clear. The missing hop analog:
    an 8 MHz hop must not `self._reader_stop = Event()` /
    `self._reader_stop.set()` (copy-paste from `_start_reader` /
    `_stop_reader`). `_do_sweep` never consults that Event;
    `_start_reader` *does* replace it on a later glance, and
    `_stop_reader` *does* `.set()` the live one. A leftover
    replacement still looks like a live walk (`mode=SWEEP`,
    reader down, `retune_calls==1`) while a glance that
    `_start_reader`s after this hop arms a new Event, then
    `_stop_reader` (Sweep / expiry) signals a different Event
    than `_reader_loop` waits on — the USB reader is orphaned
    and the next hop's `_grab` races `src.read`. `fixed_freq`
    keeps this to one grab; zeros IQ so occupancy/inspect stay
    out of the path. Leftover is the same Event, still clear,
    after the hop, and a later `.set()` must land on *that*
    Event (the glance's `_start_reader` is a different hop).
    """
    src = MockSource()
    src.fixed_freq = True
    src._fc = 5800e6
    eng = make_engine(src)
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    leftover = eng._reader_stop
    try:
        assert leftover.is_set() is False
        eng._do_sweep()
        assert src.retune_calls == 1
        assert eng._reader_stop is leftover
        assert leftover.is_set() is False
        assert leftover.is_set() is not True
        leftover.set()
        assert eng._reader_stop.is_set() is True
        leftover.clear()
        assert eng._reader_stop is leftover
        assert leftover.is_set() is False
        assert eng._reader_thread is None
        assert eng._ring is None
        assert eng._lock_tuned is None
        assert eng.state.mode == "SWEEP"
    finally:
        eng._stop_reader()
