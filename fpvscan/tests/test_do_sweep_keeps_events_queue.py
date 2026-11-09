"""A sweep hop must not replace the queue the web pump already holds."""
from __future__ import annotations

from queue import Queue

from tests.helpers import MockSource, make_engine


def test_do_sweep_keeps_leftover_events_queue():
    """`create_app.pump()` *does* capture `engine.events` once
    at startup and blocks on that Queue. Heartbeat / `_emit`
    write through `self.events`. #92 a later LOCK field keeps
    the same Queue so `_emit("frame")` still reaches the
    console. `_do_sweep` *does* `_emit("spectrum")` on every
    hop (and `_emit("detection")` once `confirm_hits` lands).
    The missing half: an 8 MHz hop must not
    `self.events = Queue()` (copy-paste from `__init__`).
    `_emit("spectrum")` / `_emit("detection")` would land on
    a queue nobody pumps, so the waterfall freezes on the
    last hop while `snapshot()` still reports SWEEP, the
    reader stays down, and `retune_calls==1` still looks
    like a live walk. `fixed_freq` keeps this to one grab;
    zeros IQ so occupancy/inspect stay out of the path.
    Leftover is the same Queue object after the hop, and
    the spectrum event must have landed on *that* Queue.
    """
    src = MockSource()
    src.fixed_freq = True
    src._fc = 5800e6
    events = Queue()
    eng = make_engine(src, events=events)
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    try:
        leftover = eng.events
        assert leftover is events
        assert leftover.empty()
        eng._do_sweep()
        assert src.retune_calls == 1
        assert eng.events is leftover
        assert eng.events is events
        assert leftover.empty() is False
        ev = leftover.get_nowait()
        assert ev["type"] == "spectrum"
        assert eng._reader_thread is None
        assert eng._ring is None
        assert eng.state.mode == "SWEEP"
    finally:
        eng._stop_reader()
