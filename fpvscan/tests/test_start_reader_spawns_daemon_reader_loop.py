"""LOCK must spawn a daemon IQ thread targeting _reader_loop(fs)."""
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

    def start(self):
        self.started += 1

    def join(self, timeout=None):
        pass


def test_start_reader_spawns_daemon_thread_targeting_reader_loop(monkeypatch):
    """#105 `_start_reader` constructs a Thread and writes the 10 ms
    dump; FrozenThread only proved `_reader_thread is not None`.
    #91 a later field of the *same* hold keeps that handle. The
    missing half: the Thread is `target=self._reader_loop`,
    `args=(fs,)`, `daemon=True`, and `start()` runs. A leftover
    `Thread(target=self._run)` still looks like a live hop
    (`retune_calls==1`, thread is not None) while a second `_run`
    races `_grab` on the same handle. A leftover
    `args=(self.src.sample_rate,)` at a board that returned a
    neighbour rate (or a test fs that is not the source clock)
    makes `_reader_loop` stamp 5 ms chunks in the wrong time
    base — DecodeState `abs_t0` jumps, tracking never shrinks,
    every field grabs full `capture_ms`. A leftover construct
    without `start()` leaves the ring at the 10 ms prime;
    shipped `capture_ms=80` then hits `len(iq) < n` and sleeps
    `idle_ms` for the whole glance. Direct `_start_reader`;
    occupancy/decode stay out. Freeze the IQ thread so USB fill
    is not the path.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = MockSource(fs=10e6)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        eng._start_reader(5800e6, fs, 0.5)
        th = eng._reader_thread
        assert th is not None
        assert isinstance(th, _CapturingThread)
        assert th.target == eng._reader_loop
        assert th.target != eng._run
        assert th.args == (fs,)
        assert th.args != (src.sample_rate,)
        assert th.args != (10e6,)
        assert th.daemon is True
        assert th.started == 1
        assert src.retune_calls == 1
        assert eng._lock_tuned == 5800e6
        ten_ms = max(2048, int(fs * 0.01))
        assert eng._ring is not None
        assert eng._ring.filled == ten_ms
    finally:
        eng._stop_reader()
