"""LOCK stop-reader must drop the ring without rewinding waterfall cadence."""
from __future__ import annotations

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


def test_stop_reader_does_not_null_lock_n(monkeypatch):
    """#74 `_start_reader` *does* keep leftover `_lock_n` (hop
    is not a first-field FFT). #71 Sweep / lock / Clear /
    peek / expiry keep it. #88 a later field *increments*
    it once a full capture is in the ring. #116 stop does
    not null `_lock_tuned`. #119 stop does not null
    `_lock_state` / `_lock_dec`. The missing half:
    `_stop_reader` itself must not `_lock_n = 0`. Sweep
    command / peek-expiry / the >1 Hz fs restart (#2) call
    stop, then the next `_start_reader` keeps the leftover
    (#74). A leftover wipe (copy-paste from a "fresh hold")
    still looks like a live teardown (`thread is None`,
    ring is None) while the next field is `1 % 8 == 1`
    (shipped `spectrum_every=8`) and spends `idle_ms` on a
    4096×4 FFT the operator did not ask for. Direct
    `_stop_reader` after planting leftover 7; occupancy
    / decode stay out. Freeze the IQ thread. Did not pin
    join timeout=2 (#10).
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = MockSource(fs=10e6)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        eng._start_reader(5800e6, fs, 0.5)
        eng._lock_n = 7
        th = eng._reader_thread
        dump = eng._ring
        ten_ms = max(2048, int(fs * 0.01))
        assert th is not None
        assert dump is not None
        assert dump.filled == ten_ms
        assert eng._lock_tuned == 5800e6
        eng._stop_reader()
        assert eng._lock_n == 7
        assert eng._lock_n != 0
        assert eng._lock_n != 1
        assert eng._reader_thread is None
        assert eng._ring is None
        assert eng._lock_tuned == 5800e6
        assert src.retune_calls == 1
    finally:
        eng._stop_reader()
        assert eng._lock_n == 7
