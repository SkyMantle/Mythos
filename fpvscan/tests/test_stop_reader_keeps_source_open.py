"""LOCK stop-reader must drop the IQ thread without closing the radio."""
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


class _Src(MockSource):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.close_calls = 0
        self.open_calls = 0

    def open(self):
        self.open_calls += 1
        super().open()

    def close(self):
        self.close_calls += 1
        super().close()


def test_stop_reader_does_not_close_the_radio(monkeypatch):
    """#19 / #50 `_run` *does* `src.close()` in `finally` after
    `_stop_reader` / `_rec_stop` — process teardown. #91 a later
    `_do_lock` / `_do_sweep` hop keeps the handle open. #24 Sweep
    *command* joins the reader so `_grab` owns the radio. The
    missing half: `_stop_reader` itself must not close. Sweep /
    peek-expiry / AFC hop call stop, then the next `_grab` /
    `retune_and_read` on the same handle. A leftover
    `self.src.close()` (copy-paste from `_run` finally) zeros
    the USB handle while the worker is still alive: the next
    sweep hop AttributeErrors / IOErrors, `confirm_hits=2`
    never lights the bird, and the next glance's `_start_reader`
    retunes a closed board (NIOS hang if the orphan is still in
    `sync_rx`). Direct `_stop_reader` after `_start_reader`;
    occupancy/decode stay out. Freeze the IQ thread. Did not
    pin join timeout=2 (#10).
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = _Src(fs=10e6)
    src.open()
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        assert src.opened is True
        assert src.close_calls == 0
        eng._start_reader(5800e6, fs, 0.5)
        leftover = eng._reader_thread
        dump = eng._ring
        ev = eng._reader_stop
        ten_ms = max(2048, int(fs * 0.01))
        assert leftover is not None
        assert dump is not None
        assert dump.filled == ten_ms
        assert src.opened is True
        assert src.close_calls == 0
        assert src.open_calls == 1
        eng._stop_reader()
        assert src.opened is True
        assert src.close_calls == 0
        assert src.open_calls == 1
        assert eng._reader_thread is None
        assert eng._ring is None
        assert eng._ring is not dump
        assert leftover.join_calls != []
        assert ev.is_set()
        assert src.retune_calls == 1
    finally:
        eng._stop_reader()
        assert src.close_calls == 0
        assert src.opened is True
