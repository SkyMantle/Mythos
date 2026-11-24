"""LOCK start-reader dump must come from retune_and_read, not src.read."""
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


class _Src(MockSource):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.read_calls = 0
        self.read_n: list[int] = []

    def read(self, n: int):
        self.read_calls += 1
        self.read_n.append(int(n))
        return super().read(n)


def test_start_reader_dump_is_retune_and_read_not_src_read(monkeypatch):
    """#101 start *does* `retune_and_read` (not the sweep fast
    path). #108 `_reader_loop` *does* `src.read` 5 ms chunks
    after the dump. BladeRF `retune_and_read` dumps
    `max(2048, fs * settle_us * 1e-6)` then returns `read(n)`;
    MockSource retune fills `_iq` without calling `read()`.
    The missing half: `_start_reader` itself must not
    `src.read` for the PLL dump. A leftover
    `first = self.src.read(max(2048, int(fs * 0.01)))` after
    a bare LO write (or instead of retune) skips the settle
    dump — DecodeState locks onto 400 µs of hopping PLL, the
    tracking window never shrinks, every field grabs full
    `capture_ms`. Direct `_start_reader`; occupancy/decode
    stay out. Freeze the IQ thread so a USB chunk cannot hide
    a dump-via-read. File pacing (#102) is not pinned.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = _Src(fs=10e6)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        assert src.read_calls == 0
        eng._start_reader(5800e6, fs, 0.5)
        ten_ms = max(2048, int(fs * 0.01))
        dump = eng._ring
        th = eng._reader_thread
        assert dump is not None
        assert dump.filled == ten_ms
        assert src.retune_calls == 1
        assert src._fc == 5800e6
        assert src.read_calls == 0
        assert src.read_n == []
        assert th is not None
        assert th.started == 1
        assert th.target is not None
    finally:
        eng._stop_reader()
