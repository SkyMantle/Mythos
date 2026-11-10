"""A LOCK field must not replace the SDR YAML dict `_run` already applied."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_do_lock_keeps_leftover_sdr_cfg_on_later_field():
    """`_run` *does* read `self.cfg["sdr"]` for `set_gain` /
    bias-tee at boot. Bias-tee *does* `_apply_bias_tee_gain`
    from that same nested dict (`gain_db` minus
    `bias_tee_gain_offset_db`, shipped 15 not the engine
    fallback 18). #93 a later field keeps the top cfg dict
    and the nested video/scan dicts so the 1 Hz fs-tolerance
    restart still sees the board rate `_run` wrote.
    The missing half: a later field on the same bird must
    not `self.cfg["sdr"] = {}` / `self.cfg["sdr"] =
    dict(self.cfg["sdr"])` / `self.cfg["sdr"] =
    {"gain_db": 30}` (copy-paste from a "fresh SDR view").
    `_do_lock` does not re-read sdr, so the hold still
    looks live (`mode=LOCK`, `lock_target`, `_ring`,
    `_lock_tuned`, reader alive, `retune_calls==1`, the
    0.4 Hz case still passes). A leftover replacement
    sends the next bias-tee click to a copy nobody else
    holds — LNA offset falls to 18 dB, the ADC clips, and
    CVBS paints bars instead of a picture. First field
    starts the reader; leftover is the same nested sdr
    dict after that, and `_apply_bias_tee_gain(True)`
    still subtracts 15 from that dict.
    """
    src = MockSource()
    eng = make_engine(src)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        leftover = eng.cfg
        leftover_sdr = eng.cfg["sdr"]
        assert leftover_sdr["gain_db"] == 30
        assert leftover_sdr["bias_tee_gain_offset_db"] == 15
        eng._do_lock()
        assert src.retune_calls == 1
        assert eng.cfg is leftover
        assert eng.cfg["sdr"] is leftover_sdr
        eng._do_lock()
        assert eng.cfg is leftover
        assert eng.cfg["sdr"] is leftover_sdr
        assert leftover_sdr["gain_db"] == 30
        assert leftover_sdr["bias_tee_gain_offset_db"] == 15
        eng._apply_bias_tee_gain(True)
        assert src.gain == 15.0
        assert src.gain != 12.0
        assert eng._reader_thread is not None
        assert eng._lock_tuned == 5800e6
        assert src.retune_calls == 1
        assert eng.state.mode == "LOCK"
    finally:
        eng._stop_reader()
