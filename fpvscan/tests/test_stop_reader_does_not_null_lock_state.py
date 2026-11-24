"""LOCK stop-reader must drop the ring without wiping genlock."""
from __future__ import annotations

from fpvscan.dsp import cvbs
from tests.helpers import MockSource, make_engine


class _CapturingThread:
    """Records constructor kwargs; start() must not run USB."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.daemon = daemon
        self.started = 0
        self.join_calls: list = []

    def start(self):
        self.started += 1

    def join(self, timeout=None):
        self.join_calls.append(timeout)


def test_stop_reader_does_not_null_lock_state_or_lock_dec(monkeypatch):
    """#50 `_start_reader` *does* reseed `DecodeState()` /
    `_lock_dec = None` *after* the hop. #69 Clear / #81 sweep
    hop *keep* leftover genlock. #116 stop does not null
    `_lock_tuned`. The missing half: `_stop_reader` itself
    must not `_lock_state = None` / `_lock_dec = None`. Sweep
    command / peek-expiry / the >1 Hz fs restart (#2) call
    stop, then the next `_start_reader` reseeds. A leftover
    wipe (copy-paste from start / Clear-the-wrong-way) still
    looks like a live teardown (`thread is None`, ring is
    None) while colliding with hop-vs-command memory: Sweep
    hop is supposed to keep genlock (#81) and let start
    reseed on the next channel. Direct `_stop_reader` after
    planting leftover period/dec; occupancy/decode stay out.
    Freeze the IQ thread. Did not pin join timeout=2 (#10).
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = MockSource(fs=10e6)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        eng._start_reader(5800e6, fs, 0.5)
        leftover = cvbs.DecodeState()
        leftover.period = 512.0
        leftover.standard = "PAL"
        leftover.abs_t0 = 1000.0
        leftover.lost = 0
        eng._lock_state = leftover
        eng._lock_dec = 2
        th = eng._reader_thread
        dump = eng._ring
        ten_ms = max(2048, int(fs * 0.01))
        assert th is not None
        assert dump is not None
        assert dump.filled == ten_ms
        assert eng._lock_tuned == 5800e6
        eng._stop_reader()
        assert eng._lock_state is leftover
        assert eng._lock_state.period == 512.0
        assert eng._lock_state.standard == "PAL"
        assert eng._lock_state.abs_t0 == 1000.0
        assert eng._lock_dec == 2
        assert eng._lock_dec is not None
        assert eng._reader_thread is None
        assert eng._ring is None
        assert eng._lock_tuned == 5800e6
        assert src.retune_calls == 1
    finally:
        eng._stop_reader()
        assert eng._lock_state is leftover
        assert eng._lock_dec == 2
