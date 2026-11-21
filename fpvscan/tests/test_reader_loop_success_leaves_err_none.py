"""LOCK USB reader must not stash _reader_err after a successful chunk."""
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


def test_reader_loop_successful_write_leaves_reader_err_none(monkeypatch):
    """#110 writes the 5 ms chunk onto the dump (`filled==dump+
    chunk`, snapshot tail is the chunk) and never looks at
    `_reader_err`. #112 stashes a USB *exception* into
    `_reader_err` and returns without writing. #113 ring-None
    / already-set Event exit *without* setting err — those
    are the stop/failure paths. The missing half: a
    successful dump+chunk still leaves `_reader_err is None`.
    `_do_lock` is `if self._reader_err is not None: raise`.
    A leftover `self._reader_err = e` / sentinel after every
    `ring.write` still looks like a live hop (`retune_calls
    ==1`, filled grew) while the next field re-raises —
    LOCK looks dead with a primed dump, Sweep never retakes
    the board. Direct `_reader_loop` after `_start_reader`;
    occupancy/decode stay out. Freeze the IQ thread. Did
    not copy #84 (`_do_lock` still consults leftover err
    *before* start). File pacing (#102) is not pinned.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = MockSource(fs=10e6)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        eng._start_reader(5800e6, fs, 0.5)
        dump = eng._ring
        ten_ms = max(2048, int(fs * 0.01))
        five_ms = max(1024, int(fs * 0.005))
        assert dump is not None
        assert dump.filled == ten_ms
        assert eng._reader_err is None
        ns: list[int] = []
        orig = src.read

        def _read(n):
            ns.append(int(n))
            eng._reader_stop.set()
            return orig(n)

        src.read = _read
        eng._reader_loop(fs)
        assert ns == [five_ms]
        assert eng._reader_err is None
        assert eng._ring is dump
        assert dump.filled == ten_ms + five_ms
        assert src.retune_calls == 1
    finally:
        eng._stop_reader()
