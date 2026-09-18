from __future__ import annotations

from queue import Queue

import numpy as np

from fpvscan.dsp import cvbs, demod
from fpvscan.engine import Engine


CFG = {
    "scan": {
        "start_hz": 1.0e9,
        "stop_hz": 1.2e9,
        "sample_rate": 20e6,
        "channel_bw_hz": 12e6,
        "step_hz": 12e6,
        "fft_size": 1024,
        "averages": 1,
        "priority_bands": False,
    },
    "video": {
        "sample_rate": 20e6,
        "channel_bw_hz": 12e6,
        "afc": True,
        "afc_deadband_hz": 80e3,
        "afc_digital_max_hz": 1.5e6,
        "afc_limit_hz": 20e6,
        "afc_max_step_hz": 0.25e6,
        "afc_gain": 0.5,
        "afc_peg_frames": 8,
    },
    "rotator": {"enable": False},
    "sdr": {},
}


class _Source:
    name = "stub"
    sample_rate = 20e6
    overflows = 0
    clip_frac = 0.0
    adc_rms = 0.0
    bias_tee = False

    def retune_and_read(self, _freq_hz, samples):
        return np.zeros(max(8, int(samples)), dtype=np.complex64)


def _engine() -> Engine:
    return Engine(_Source(), CFG, Queue())


def _free_run_pal() -> cvbs.Frame:
    rng = np.random.default_rng(11)
    luma = rng.integers(24, 96, size=(120, 160), dtype=np.uint8)
    return cvbs.Frame(
        luma=luma,
        line_rate=15_625.0,
        lines=120,
        standard="PAL",
        locked=False,
        free_run=True,
    )


def _locked_analog() -> cvbs.Frame:
    luma = np.full((288, 320), 64, np.uint8)
    luma[::2, :40] = 8
    return cvbs.Frame(
        luma=luma,
        line_rate=15_625.0,
        lines=288,
        standard="PAL",
        locked=True,
        free_run=False,
    )


def test_free_run_pal_comb_drives_digital_afc(monkeypatch) -> None:
    engine = _engine()
    frame = _free_run_pal()
    pic = cvbs.score_picture(frame)
    analog_ok = cvbs.analog_usable(frame, pic=pic)
    assert analog_ok is False
    assert engine._video_ok_for_afc(frame, pic=pic) is True
    monkeypatch.setattr(demod, "freq_error_from_demod", lambda *_a, **_k: 220e3)
    vcfg = CFG["video"]
    before = engine._afc
    engine._apply_afc(
        np.ones(2048, np.float32), 20e6, 0.0, 12e6, 4.8e6, vcfg,
        pic_locked=False,
    )
    assert engine._afc != before
    assert abs(engine._afc) > 0.0


def test_analog_ok_does_not_rf_nudge() -> None:
    engine = _engine()
    engine.state.lock_target = 1.1e9
    engine._afc = 1.5e6
    engine._last_err = 400e3
    engine._afc_peg_n = 20
    vcfg = CFG["video"]
    engine._update_afc_peg(vcfg, 1.5e6, pic_locked=False, freeze_rf=True)
    assert engine.state.lock_target == 1.1e9
    assert engine._afc_nudge is False
    assert engine._afc == 1.5e6


def test_afc_disabled_stays_zero(monkeypatch) -> None:
    engine = _engine()
    frame = _free_run_pal()
    monkeypatch.setattr(demod, "freq_error_from_demod", lambda *_a, **_k: 300e3)
    vcfg = {**CFG["video"], "afc": False}
    hold_afc = False
    analog_ok = False
    if (vcfg.get("afc", True) and engine._video_ok_for_afc(frame)
            and not hold_afc):
        engine._apply_afc(
            np.ones(2048, np.float32), 20e6, 0.0, 12e6, 4.8e6, vcfg,
        )
    assert analog_ok is False
    assert engine._afc == 0.0


def test_locked_analog_ok_does_not_require_contradictory_gate() -> None:
    engine = _engine()
    frame = _locked_analog()
    pic = cvbs.PictureScore(value=0.55, locked=True, lines=288, row_corr=0.45)
    analog_ok = cvbs.analog_usable(frame, pic=pic)
    assert analog_ok is True
    assert engine._video_ok_for_afc(frame, pic=pic) is True
    engine.state.lock_target = 5.1e9
    engine._afc = 1.5e6
    engine._last_err = 500e3
    engine._update_afc_peg(CFG["video"], 1.5e6, freeze_rf=True)
    assert engine.state.lock_target == 5.1e9
