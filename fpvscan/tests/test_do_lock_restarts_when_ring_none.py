"""LOCK must restart the IQ reader when the ring is gone, even if tune matches."""
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


def test_do_lock_restarts_reader_when_ring_none_and_lock_tuned_matches(monkeypatch):
    """#91 a later field keeps leftover `_lock_tuned` *and* the
    live ring, so `_do_lock` skips `_start_reader`. #50 hop
    *does* write `want`. #115 short-ring miss keeps the live
    reader (`retune_calls` stays 1). The missing half: after
    `_stop_reader` the ring is gone while leftover tune still
    matches `want`. `_do_lock` is `if self._ring is None or
    self._lock_tuned != want: _start_reader`. Sweep command /
    peek-expiry / the >1 Hz fs restart (#2) stop, then the
    next field must retune. A leftover drop of `_ring is None`
    (only `_lock_tuned != want`) still looks like a hold
    (`lock_tuned == 5800e6`) while `snapshot` AttributeErrors
    on None — LOCK dies after Сканувати / a 0.4 Hz fs blip
    that already passed the 1 Hz gate. Direct `_do_lock` after
    `_start_reader` + `_stop_reader`; occupancy/decode stay
    out. Freeze the IQ thread so a USB chunk cannot hide a
    skip. Src fs matches `video.sample_rate` so the 1 Hz
    restart (#2) does not fire. File pacing (#102) is not
    pinned.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = MockSource(fs=fs)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5, idle_ms=0)
    try:
        eng.state.mode = "LOCK"
        eng.state.lock_target = 5800e6
        eng._start_reader(5800e6, fs, 0.5)
        first = eng._reader_thread
        dump = eng._ring
        ten_ms = max(2048, int(fs * 0.01))
        field = int(fs * 0.080)
        assert first is not None
        assert dump is not None
        assert dump.filled == ten_ms
        assert dump.filled < field
        assert eng._lock_tuned == 5800e6
        assert src.retune_calls == 1
        eng._stop_reader()
        assert eng._ring is None
        assert eng._reader_thread is None
        assert eng._lock_tuned == 5800e6
        assert src.retune_calls == 1
        assert eng._lock_n == 0
        eng._do_lock()
        second = eng._reader_thread
        new_dump = eng._ring
        assert second is not None
        assert second is not first
        assert second.started == 1
        assert second.join_calls == []
        assert new_dump is not None
        assert new_dump is not dump
        assert new_dump.filled == ten_ms
        assert new_dump.filled < field
        assert src.retune_calls == 2
        assert src._fc == 5800e6
        assert eng._lock_tuned == 5800e6
        assert eng._lock_n == 0
        assert eng._reader_err is None
        assert not eng._reader_stop.is_set()
    finally:
        eng._stop_reader()
