"""A sweep hop must not replace the EngineState the console already reads."""
from __future__ import annotations

from fpvscan.engine import Detection
from tests.helpers import MockSource, make_engine


def test_do_sweep_keeps_leftover_state():
    """`__init__` *does* `self.state = EngineState()`. Heartbeat /
    `/api/state` *do* `engine.snapshot()` which reads
    `self.state` live. #96 a later LOCK field keeps the
    EngineState *and* the detections dict so occupancy /
    glance timer / walk counter do not flash to boot. #90
    hop keeps the LOCK reader stopped; #97 hop keeps cfg.
    The missing half: an 8 MHz hop must not
    `self.state = EngineState()` /
    `self.state = EngineState(mode="SWEEP")` (copy-paste
    from `__init__` or a "fresh walk"). `_do_sweep` *does*
    write `tuned_hz` / `sweep_pos_hz` (and `sweeps_done`
    on plan wrap) into that object; it never consults
    EngineState identity. A leftover reconstruction still
    looks like a live walk (`mode=SWEEP`, reader down,
    `retune_calls==1`) while `detections` / `auto` /
    `sweeps_done` go back to whoever owns the old object
    and the heartbeat occupancy / glance timer / walk
    counter flash to boot — `confirm_hits=2` already
    lit the bird, then the ribbon looks empty. `fixed_freq`
    keeps this to one grab (no wrap, so `sweeps_done`
    stays); zeros IQ so occupancy/inspect stay out of
    the path. Leftover is the same EngineState *and* the
    same detections dict after the hop, and the hop's
    `tuned_hz` write must land on *that* object.
    """
    src = MockSource()
    src.fixed_freq = True
    src._fc = 5800e6
    eng = make_engine(src)
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    eng.state.mode = "SWEEP"
    eng.state.lock_target = None
    eng.state.auto = True
    eng.state.auto_until = 4_000_000_000.0
    eng.state.sweeps_done = 3
    leftover_det = Detection(
        freq_hz=5800e6, bandwidth_hz=10e6, snr_db=18.0,
        standard="NTSC", confidence=0.80, channel="F4", band="5G8",
        first_seen=11.0, last_seen=22.0, hits=2,
    )
    eng.state.detections[5800] = leftover_det
    leftover = eng.state
    leftover_dets = eng.state.detections
    try:
        eng._do_sweep()
        assert src.retune_calls == 1
        assert eng.state is leftover
        assert eng.state.detections is leftover_dets
        assert leftover_dets[5800] is leftover_det
        assert leftover.mode == "SWEEP"
        assert leftover.lock_target is None
        assert leftover.auto is True
        assert leftover.auto_until == 4_000_000_000.0
        assert leftover.sweeps_done == 3
        assert leftover.tuned_hz == 5800e6
        assert leftover.sweep_pos_hz == 5800e6
        assert leftover.tuned_hz != 0.0
        assert eng._reader_thread is None
        assert eng._ring is None
        snap = eng.snapshot()
        assert snap["mode"] == "SWEEP"
        assert snap["auto"] is True
        assert snap["sweeps_done"] == 3
        assert snap["lock_target"] is None
        assert snap["tuned_hz"] == 5800e6
        assert any(d["freq_hz"] == 5800e6 for d in snap["detections"])
    finally:
        eng._stop_reader()
