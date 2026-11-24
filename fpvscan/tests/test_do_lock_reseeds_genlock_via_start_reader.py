"""LOCK after stop must reseed genlock in start-reader, not in stop."""
from __future__ import annotations

from fpvscan.dsp import cvbs
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


def test_do_lock_after_stop_reseeds_genlock_via_start_reader(monkeypatch):
    """#50 start *does* `DecodeState()` / `_lock_dec = None`
    after the hop (source-order). #85 a later field of the
    *same* hold keeps genlock. #116 after stop, `_do_lock`
    restarts because `_ring is None`. The missing half:
    leftover period/dec that stop kept must be *replaced* by
    start on that restart — new DecodeState, period/abs_t0
    None, `_lock_dec is None`. A leftover keep-old-period
    across Sweep→LOCK (dropping the reseed, or reseeding in
    stop then restoring) tracks the new VTx with the old
    line length; tracking never shrinks, every field grabs
    full `capture_ms`. Direct `_do_lock` after `_start_reader`
    + planted leftover + `_stop_reader`; occupancy/decode
    stay out. Freeze the IQ thread. Src fs matches
    `video.sample_rate` so the 1 Hz restart (#2) does not
    fire. File pacing (#102) is not pinned.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = MockSource(fs=fs)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5, idle_ms=0)
    try:
        eng.state.mode = "LOCK"
        eng.state.lock_target = 5800e6
        eng._start_reader(5800e6, fs, 0.5)
        leftover = cvbs.DecodeState()
        leftover.period = 512.0
        leftover.standard = "PAL"
        leftover.abs_t0 = 1000.0
        leftover.lost = 0
        eng._lock_state = leftover
        eng._lock_dec = 2
        first = eng._reader_thread
        dump = eng._ring
        ten_ms = max(2048, int(fs * 0.01))
        field = int(fs * 0.080)
        assert first is not None
        assert dump is not None
        assert dump.filled == ten_ms
        assert dump.filled < field
        assert src.retune_calls == 1
        eng._stop_reader()
        assert eng._ring is None
        assert eng._lock_state is leftover
        assert eng._lock_dec == 2
        assert eng._lock_n == 0
        eng._do_lock()
        second = eng._reader_thread
        new_state = eng._lock_state
        new_dump = eng._ring
        assert second is not None
        assert second is not first
        assert second.started == 1
        assert new_dump is not None
        assert new_dump is not dump
        assert new_dump.filled == ten_ms
        assert src.retune_calls == 2
        assert src._fc == 5800e6
        assert new_state is not leftover
        assert new_state is not None
        assert new_state.period is None
        assert new_state.abs_t0 is None
        assert new_state.lost == 0
        assert eng._lock_dec is None
        assert eng._lock_tuned == 5800e6
        assert eng._lock_n == 0
        assert eng._reader_err is None
    finally:
        eng._stop_reader()
