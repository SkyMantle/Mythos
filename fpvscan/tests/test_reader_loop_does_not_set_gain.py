"""LOCK USB reader must fill the ring without rewriting VGA gain."""
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
        self.set_gain_calls = 0
        self.set_gain_db: list[float] = []

    def set_gain(self, db):
        self.set_gain_calls += 1
        self.set_gain_db.append(float(db))
        super().set_gain(db)


def test_reader_loop_does_not_set_gain(monkeypatch):
    """#50 `_run` *does* `set_gain(gain_db)` once after open,
    then `_apply_bias_tee_gain`. The `bias_tee` command *does*
    both on a click. #120 start does not `set_gain`. #121 stop
    / later field do not `set_gain`. #108 / #110 / #114 the
    USB producer *does* `src.read` 5 ms chunks onto the dump.
    The missing half: `_reader_loop` itself must not
    `src.set_gain`. The USB producer is not the radio-knob
    owner. A leftover `self.src.set_gain(self.cfg["sdr"]
    ["gain_db"])` or `_apply_bias_tee_gain(False)` at the top
    of the loop (copy-paste from `_run`) would pass #120 /
    #121 (those freeze the IQ thread so the loop never runs)
    and still rewrite 15 → 30 on every chunk while the LNA is
    still on — ADC clips, CVBS paints bars, or the VGA write
    races the live `sync_rx` (#27 ERR_IO / NIOS). The dump
    still looks primed (`retune_calls==1`, filled grew).
    Direct `_reader_loop` after `_start_reader` with leftover
    compensated VGA; occupancy / decode stay out. Freeze the
    IQ thread. File pacing (#102) is not pinned.
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
        assert src.set_gain_calls == 0
        assert src.gain == 15.0
        assert src.retune_calls == 1
        ns: list[int] = []
        orig = src.read

        def _read(n):
            ns.append(int(n))
            eng._reader_stop.set()
            return orig(n)

        src.read = _read
        eng._reader_loop(fs)
        assert ns == [five_ms]
        assert src.set_gain_calls == 0
        assert src.set_gain_db == []
        assert src.gain == 15.0
        assert src.gain != 30.0
        assert src.bias is True
        assert eng._reader_err is None
        assert eng._ring is dump
        assert dump.filled == ten_ms + five_ms
        assert leftover.started == 1
        assert leftover.join_calls == []
        assert src.retune_calls == 1
        assert src.set_sample_rate_calls == 0
        assert eng._lock_tuned == 5800e6
    finally:
        eng._stop_reader()
