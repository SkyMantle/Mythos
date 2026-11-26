"""LOCK later field must not rewrite VGA gain on a live reader."""
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


def test_do_lock_later_field_does_not_set_gain(monkeypatch):
    """#50 `_run` *does* `set_gain(gain_db)` once after open,
    then `_apply_bias_tee_gain` (YAML 30 − 15 LNA offset → 15
    dB while the tee is on). The `bias_tee` command *does*
    both on a click. #120 `_start_reader` does not `set_gain`.
    #91 / #115 a later field of the *same* hold keeps the live
    ring (skips start) and a short snapshot (`filled==10 ms`
    vs shipped `capture_ms=80`) returns without `_lock_n++`.
    The missing half: that later-field miss must not
    `src.set_gain`. VGA belongs to `_run` / the bias-tee
    command, not the field. A leftover
    `self.src.set_gain(self.cfg["sdr"]["gain_db"])` or
    `_apply_bias_tee_gain(False)` (copy-paste from `_run`) at
    the top of `_do_lock` would pass #120 (start never hits
    it) and still rewrite 15 → 30 on every field while the
    LNA is still on — ADC clips, CVBS paints bars, the ring
    still looks primed (`retune_calls==1`, same thread).
    Direct `_do_lock` after `_start_reader` with leftover
    compensated VGA; occupancy/decode stay out. Freeze the IQ
    thread. Src fs matches `video.sample_rate` so the 1 Hz
    restart (#2) does not fire. File pacing (#102) is not
    pinned.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = _Src(fs=fs)
    src.gain = 15.0
    src.bias = True
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5, idle_ms=0)
    try:
        eng.state.mode = "LOCK"
        eng.state.lock_target = 5800e6
        eng._start_reader(5800e6, fs, 0.5)
        leftover = eng._reader_thread
        dump = eng._ring
        ten_ms = max(2048, int(fs * 0.01))
        field = int(fs * 0.080)
        assert leftover is not None
        assert dump is not None
        assert dump.filled == ten_ms
        assert dump.filled < field
        assert src.set_gain_calls == 0
        assert src.gain == 15.0
        assert src.retune_calls == 1
        eng._do_lock()
        assert eng._reader_thread is leftover
        assert eng._ring is dump
        assert dump.filled == ten_ms
        assert leftover.started == 1
        assert leftover.join_calls == []
        assert src.retune_calls == 1
        assert src.set_gain_calls == 0
        assert src.set_gain_db == []
        assert src.gain == 15.0
        assert src.gain != 30.0
        assert src.set_sample_rate_calls == 0
        assert src.bias is True
        assert eng._lock_tuned == 5800e6
        assert eng._lock_n == 0
        assert eng._reader_err is None
    finally:
        eng._stop_reader()
