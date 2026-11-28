"""LOCK after stop must restart the ring without rewriting the sample clock."""
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


def test_do_lock_after_stop_does_not_set_sample_rate(monkeypatch):
    """#2 / `_do_lock` *does* `set_sample_rate(fs_want)` when
    `abs(src.sample_rate - fs_want) > 1` (the 1 Hz gate; exact
    `!=` used to tear the IQ thread every field). #50 `_run`
    *does* `set_sample_rate` once after open. #116
    `_start_reader` does not `set_sample_rate`. #123 after
    stop, `_do_lock` restarts because `_ring is None` and does
    not `set_gain` / `set_bias_tee`. The missing half: that
    ring-None restart must not `src.set_sample_rate` next to
    `_start_reader` *inside* `_do_lock` when fs already
    matches. The 1 Hz gate is the clock owner, not the hop. A
    leftover `self.src.set_sample_rate(fs_want)` (copy-paste
    from `_run` / the >1 Hz branch) beside the restart would
    pass #116 (start never hits it) and still restart the
    bladeRF RX stream on every Sweep→LOCK / peek-expiry while
    fs already matches — PLL dump + `_config_stream`,
    DecodeState reseeds via start, tracking never shrinks, the
    new dump still looks primed (`retune_calls==2`, new
    thread). Direct `_do_lock` after `_start_reader` +
    `_stop_reader`; occupancy / decode stay out. Freeze the IQ
    thread. Src fs matches `video.sample_rate` so the 1 Hz
    restart (#2) does not fire. File pacing (#102) is not
    pinned.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = MockSource(fs=fs)
    src.gain = 15.0
    src.bias = True
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
        assert src.set_sample_rate_calls == 0
        assert src.sample_rate == fs
        assert src.retune_calls == 1
        eng._stop_reader()
        assert eng._ring is None
        assert first.join_calls != []
        assert src.set_sample_rate_calls == 0
        assert src.sample_rate == fs
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
        assert src.set_sample_rate_calls == 0
        assert src.sample_rate == fs
        assert src.gain == 15.0
        assert src.bias is True
        assert eng._lock_tuned == 5800e6
        assert eng._lock_n == 0
        assert eng._reader_err is None
    finally:
        eng._stop_reader()
