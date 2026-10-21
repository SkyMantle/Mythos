"""Clear must not reseed LOCK genlock / decimation memory."""
from __future__ import annotations

from fpvscan.dsp.cvbs import DecodeState
from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_clear_keeps_lock_dec_and_decode_state():
    """#63 kept `_lock_tuned` so Очистити does not join the reader.
    #64 kept `_ring`. `_lock_dec` / `_lock_state` are the genlock
    half of the same button: `_do_lock` reseeds DecodeState whenever
    `dec` disagrees with the last call, and a None leftover skips
    the tracking snapshot and falls back to a blind `_attempt`.
    Zeroing them on Clear rolls the live field even though the
    operator only wanted the hit list emptied. #48 is the occupancy
    BW collapse after Clear; this is the TBC side.
    """
    eng = make_engine()
    state = DecodeState(sign=-1.0, period=1274.0, standard="NTSC",
                        abs_t0=8000.0, lost=0)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = False
    eng._lock_tuned = 5800e6
    eng._lock_dec = 2
    eng._lock_state = state
    eng.state.detections[5800] = Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=14.0, hits=2,
    )
    eng.command("clear")
    eng._drain_commands()
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
    assert eng._lock_tuned == 5800e6
    assert eng._lock_dec == 2
    assert eng._lock_state is state
    assert state.period == 1274.0
    assert state.standard == "NTSC"
    assert eng.state.detections == {}
