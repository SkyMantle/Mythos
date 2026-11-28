"""LOCK USB reader must fill the ring without a settled retune dump."""
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
        self.retune_n: list[int] = []
        self.retune_hz: list[float] = []

    def retune_and_read(self, hz: float, n: int):
        self.retune_n.append(int(n))
        self.retune_hz.append(float(hz))
        return super().retune_and_read(hz, n)


def test_reader_loop_does_not_retune_and_read(monkeypatch):
    """#101 / #119 `_start_reader` *does* the PLL dump via
    `retune_and_read` (not `_grab` / fast, not `src.read`).
    #108 / #110 / #114 the USB producer *does* `src.read` 5 ms
    chunks after that dump. The missing half: `_reader_loop`
    itself must not `src.retune_and_read`. The USB producer is
    not the LO owner; start already settled the hop. A leftover
    `iq = self.src.retune_and_read(self._lock_tuned, chunk)`
    (copy-paste from start) instead of / in addition to `read`
    would pass #119 (start never hits the loop) and #108 (chunk
    size still looks right if retune returns `n` samples) and
    still hop+dump every 5 ms — extra `set_center_freq` +
    `settle_us` skip races the live `sync_rx` (#27 ERR_IO /
    NIOS), DecodeState reseeds onto hopping PLL, tracking never
    shrinks, every field grabs full `capture_ms`. The dump
    still looks primed (`filled` grew). Direct `_reader_loop`
    after `_start_reader`; occupancy / decode stay out. Freeze
    the IQ thread. Wrap both `read` and `retune_and_read` so a
    leftover-instead-of-read cannot hang the loop. File pacing
    (#102) is not pinned.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = _Src(fs=10e6)
    src.gain = 15.0
    src.bias = True
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        eng._start_reader(5800e6, fs, 0.5)
        leftover = eng._reader_thread
        dump = eng._ring
        ten_ms = max(2048, int(fs * 0.01))
        five_ms = max(1024, int(fs * 0.005))
        assert leftover is not None
        assert dump is not None
        assert dump.filled == ten_ms
        assert src.retune_calls == 1
        assert src.retune_n == [ten_ms]
        assert src.retune_hz == [5800e6]
        assert src._fc == 5800e6
        ns: list[int] = []
        orig_read = src.read
        orig_retune = src.retune_and_read

        def _read(n):
            ns.append(int(n))
            eng._reader_stop.set()
            return orig_read(n)

        def _retune(hz, n):
            eng._reader_stop.set()
            return orig_retune(hz, n)

        src.read = _read
        src.retune_and_read = _retune
        eng._reader_loop(fs)
        assert ns == [five_ms]
        assert src.retune_calls == 1
        assert src.retune_n == [ten_ms]
        assert src.retune_hz == [5800e6]
        assert src._fc == 5800e6
        assert src.gain == 15.0
        assert src.bias is True
        assert eng._reader_err is None
        assert eng._ring is dump
        assert dump.filled == ten_ms + five_ms
        assert leftover.started == 1
        assert leftover.join_calls == []
        assert src.set_sample_rate_calls == 0
        assert eng._lock_tuned == 5800e6
    finally:
        eng._stop_reader()
