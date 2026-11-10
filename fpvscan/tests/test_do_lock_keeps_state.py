"""A LOCK field must not replace the EngineState the console already reads."""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import MockSource, make_engine


def test_do_lock_keeps_leftover_state_on_later_field():
    """`__init__` *does* `self.state = EngineState()`. Heartbeat /
    `/api/state` *do* `engine.snapshot()` which reads
    `self.state` live. #90 a later field keeps `mode` /
    `lock_target` so `_run` stays on the bird. #85 keeps
    `auto` / `auto_until` and the occupancy Detection.
    #93 keeps `cfg` / `_thread` so YAML and the worker
    handle stay the objects `_run` already holds. The
    missing half: a later field on the same bird must not
    `self.state = EngineState()` / `self.state =
    EngineState(mode=self.state.mode,
    lock_target=self.state.lock_target)` (copy-paste from
    `__init__` or a "fresh hold"). `_do_lock` never
    consults the EngineState object identity; snapshot
    always goes through `self.state`. A leftover
    reconstruction still looks like a live hold
    (`mode=LOCK`, `lock_target`, `_ring`, `_lock_tuned`,
    reader alive, `retune_calls==1`, the 0.4 Hz
    fs-tolerance case still passes) while `detections` /
    `auto` / `sweeps_done` go back to whoever owns the
    old object and the heartbeat occupancy / glance
    timer / walk counter flash to boot. First field
    starts the reader; leftover is the same EngineState
    *and* the same detections dict after that.
    """
    src = MockSource()
    eng = make_engine(src)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = True
    eng.state.auto_until = 4_000_000_000.0
    eng.state.sweeps_done = 3
    leftover_det = Detection(
        freq_hz=5800e6, bandwidth_hz=10e6, snr_db=18.0,
        standard="NTSC", confidence=0.80, channel="F4", band="5G8",
        first_seen=11.0, last_seen=22.0, hits=2,
    )
    eng.state.detections[5800] = leftover_det
    try:
        leftover = eng.state
        leftover_dets = eng.state.detections
        eng._do_lock()
        assert src.retune_calls == 1
        assert eng.state is leftover
        eng._do_lock()
        assert eng.state is leftover
        assert eng.state.detections is leftover_dets
        assert leftover_dets[5800] is leftover_det
        assert leftover.mode == "LOCK"
        assert leftover.lock_target == 5800e6
        assert leftover.auto is True
        assert leftover.auto_until == 4_000_000_000.0
        assert leftover.sweeps_done == 3
        assert eng._reader_thread is not None
        assert eng._lock_tuned == 5800e6
        assert src.retune_calls == 1
        snap = eng.snapshot()
        assert snap["mode"] == "LOCK"
        assert snap["auto"] is True
        assert snap["sweeps_done"] == 3
    finally:
        eng._stop_reader()
