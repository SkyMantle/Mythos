from __future__ import annotations

from pathlib import Path
from queue import Queue

from fpvscan.auto_mgc import (
    MAX_DB,
    MIN_DB,
    MgcSample,
    MgcState,
    due,
    next_gain_db,
    picture_is_good,
    picture_is_usable,
    picture_is_weak,
)


def _sample(**kw) -> MgcSample:
    base = dict(
        gain_db=35.0,
        pic_locked=False,
        pic_score=0.0,
        pic_lines=0,
        row_corr=0.0,
        clip_frac=0.0,
    )
    base.update(kw)
    return MgcSample(**base)


def test_weak_no_clip_steps_up() -> None:
    nxt = next_gain_db(_sample(gain_db=35, pic_locked=False, pic_score=0.05, clip_frac=0.0))
    assert nxt == 38.0
    assert picture_is_weak(_sample(pic_locked=False, pic_score=0.05, pic_lines=0))


def test_clip_steps_down() -> None:
    nxt = next_gain_db(_sample(
        gain_db=50, pic_locked=False, pic_score=0.0, clip_frac=0.05,
    ))
    assert nxt == 47.0


def test_clip_below_old_one_percent_still_downs() -> None:
    nxt = next_gain_db(_sample(
        gain_db=50, pic_locked=False, pic_score=0.0, clip_frac=0.003,
    ))
    assert nxt == 47.0


def test_high_rms_steps_down_without_clip() -> None:
    nxt = next_gain_db(_sample(
        gain_db=50, pic_locked=False, pic_score=0.0, clip_frac=0.0, adc_rms=0.55,
    ))
    assert nxt == 47.0


def test_luma_crush_steps_down_without_clip() -> None:
    nxt = next_gain_db(_sample(
        gain_db=50, pic_locked=False, pic_score=0.0, clip_frac=0.0, sat_frac=0.25,
    ))
    assert nxt == 47.0


def test_clip_wins_over_weak_picture() -> None:
    nxt = next_gain_db(_sample(
        gain_db=40, pic_locked=False, pic_score=0.0, pic_lines=0, clip_frac=0.02,
    ))
    assert nxt < 40


def test_locked_good_picture_holds() -> None:
    sample = _sample(
        gain_db=42, pic_locked=True, pic_score=0.55, pic_lines=250, row_corr=0.4,
        clip_frac=0.0,
    )
    assert picture_is_good(sample)
    assert next_gain_db(sample) == 42.0


def test_locked_usable_does_not_raise_to_brighten() -> None:
    sample = _sample(
        gain_db=30, pic_locked=True, pic_score=0.22, pic_lines=220, row_corr=0.2,
        clip_frac=0.0,
    )
    assert picture_is_usable(sample)
    assert next_gain_db(sample) == 30.0


def test_operator_hold_does_not_yank() -> None:
    nxt = next_gain_db(_sample(
        gain_db=20, pic_locked=False, pic_score=0.0, clip_frac=0.0, operator_hold=True,
    ))
    assert nxt == 20.0


def test_disabled_holds() -> None:
    nxt = next_gain_db(_sample(
        gain_db=20, pic_locked=False, pic_score=0.0, enabled=False,
    ))
    assert nxt == 20.0


def test_caps_at_max_60() -> None:
    nxt = next_gain_db(_sample(
        gain_db=MAX_DB, pic_locked=False, pic_score=0.0, clip_frac=0.0,
    ))
    assert nxt == MAX_DB


def test_floor_at_min_on_clip() -> None:
    nxt = next_gain_db(_sample(
        gain_db=MIN_DB, pic_locked=True, pic_score=0.8, pic_lines=250,
        row_corr=0.5, clip_frac=0.2,
    ))
    assert nxt == MIN_DB


def test_no_bias_tee_term_in_step() -> None:
    """Loop must not take Bias-T as a fixed +15; only clip + picture."""
    assert "bias" not in MgcSample.__dataclass_fields__
    a = next_gain_db(_sample(gain_db=35, pic_locked=False, pic_score=0.0, clip_frac=0.0))
    b = next_gain_db(_sample(gain_db=35, pic_locked=False, pic_score=0.0, clip_frac=0.0))
    assert a == b == 38.0


def test_interval_not_every_frame() -> None:
    assert not due(0.016, 0.0, 0.40)
    assert due(0.40, 0.0, 0.40)
    assert due(1.0, 0.5, 0.40)


def test_up_without_improve_reverts_and_stops() -> None:
    state = MgcState()
    up = next_gain_db(_sample(
        gain_db=35, pic_locked=False, pic_score=0.05, clip_frac=0.0, now_s=0.0,
    ), state)
    assert up == 38.0
    hold = next_gain_db(_sample(
        gain_db=38, pic_locked=False, pic_score=0.05, clip_frac=0.0, now_s=0.3,
    ), state)
    assert hold == 38.0
    reverted = next_gain_db(_sample(
        gain_db=38, pic_locked=False, pic_score=0.04, clip_frac=0.0, now_s=0.8,
    ), state)
    assert reverted == 35.0
    stuck = next_gain_db(_sample(
        gain_db=35, pic_locked=False, pic_score=0.04, clip_frac=0.0, now_s=2.0,
    ), state)
    assert stuck == 35.0


def test_up_with_improve_can_climb_again() -> None:
    state = MgcState()
    assert next_gain_db(_sample(
        gain_db=35, pic_locked=False, pic_score=0.05, clip_frac=0.0, now_s=0.0,
    ), state) == 38.0
    assert next_gain_db(_sample(
        gain_db=38, pic_locked=False, pic_score=0.12, clip_frac=0.0, now_s=0.8,
    ), state) == 38.0
    assert next_gain_db(_sample(
        gain_db=38, pic_locked=False, pic_score=0.12, clip_frac=0.0, now_s=1.3,
    ), state) == 41.0


def test_frozen_identical_frames_do_not_climb_to_54() -> None:
    state = MgcState()
    sig = tuple([40.0] * (12 * 12))
    gain = 48.0
    for i in range(12):
        gain = next_gain_db(_sample(
            gain_db=gain, pic_locked=False, pic_score=0.04, clip_frac=0.0,
            frame_sig=sig, now_s=i * 0.4,
        ), state)
        assert gain < 54.0
    assert gain <= 48.0


def test_good_still_scene_does_not_step_down() -> None:
    state = MgcState()
    sig = tuple(float(i % 17) for i in range(12 * 12))
    gain = 42.0
    for i in range(5):
        gain = next_gain_db(_sample(
            gain_db=42, pic_locked=True, pic_score=0.55, pic_lines=250,
            row_corr=0.4, clip_frac=0.0, frame_sig=sig, now_s=i * 0.4,
        ), state)
    assert gain == 42.0


class _GainSrc:
    name = "stub"
    sample_rate = 20e6
    overflows = 0
    clip_frac = 0.0
    adc_rms = 0.0
    bias_tee = False
    gain_db = 35.0

    def set_gain(self, db: float) -> None:
        self.gain_db = float(db)


def _engine(gain_db: float = 35, auto_gain: bool = True, bias_tee: bool = False):
    from fpvscan.engine import Engine

    cfg = {
        "scan": {},
        "video": {},
        "sdr": {
            "gain_db": gain_db,
            "auto_gain": auto_gain,
            "bias_tee": bias_tee,
            "auto_gain_interval_s": 0,
        },
    }
    src = _GainSrc()
    src.gain_db = float(gain_db)
    eng = Engine(src, cfg, Queue())
    return eng, cfg, src


def test_engine_steps_catalog_gain_and_respects_hold() -> None:
    eng, cfg, src = _engine()
    eng._mgc_last_mono = 0.0
    eng._maybe_auto_mgc(None)
    assert cfg["sdr"]["gain_db"] == 38
    assert src.gain_db == 38.0
    eng.hold_auto_mgc(30)
    eng._mgc_last_mono = 0.0
    eng._maybe_auto_mgc(None)
    assert cfg["sdr"]["gain_db"] == 38


def test_engine_bias_tee_uses_existing_offset_not_plus_15() -> None:
    from fpvscan.engine import Engine

    cfg = {
        "scan": {},
        "video": {},
        "sdr": {
            "gain_db": 35,
            "auto_gain": True,
            "bias_tee": True,
            "bias_tee_gain_offset_db": 15,
            "auto_gain_interval_s": 0,
        },
    }
    src = _GainSrc()
    eng = Engine(src, cfg, Queue())
    eng._mgc_last_mono = 0.0
    eng._maybe_auto_mgc(None)
    assert cfg["sdr"]["gain_db"] == 38
    assert src.gain_db == 23.0


def test_operator_gain_write_disables_auto(tmp_path: Path) -> None:
    from fpvscan.app.adapters.engine_adapter import EngineAdapter

    eng, cfg, src = _engine(gain_db=54, auto_gain=True, bias_tee=False)
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("sdr:\n  gain_db: 54\n", encoding="utf-8")
    adapter = EngineAdapter(eng, yaml_path)
    commands: list[tuple] = []
    eng.command = lambda name, **kw: commands.append((name, kw))  # type: ignore[method-assign]

    adapter.apply_values({"sdr.gain_db": 30})
    assert cfg["sdr"]["auto_gain"] is False
    assert cfg["sdr"]["gain_db"] == 30
    assert cfg["sdr"]["bias_tee"] is False
    assert src.gain_db == 30.0
    assert commands == []

    for _ in range(6):
        eng._mgc_last_mono = 0.0
        eng._maybe_auto_mgc(None)
    assert cfg["sdr"]["gain_db"] == 30
    assert cfg["sdr"]["auto_gain"] is False


def test_operator_auto_resume_allows_climb(tmp_path: Path) -> None:
    from fpvscan.app.adapters.engine_adapter import EngineAdapter

    eng, cfg, src = _engine(gain_db=30, auto_gain=False)
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("sdr:\n  gain_db: 30\n", encoding="utf-8")
    adapter = EngineAdapter(eng, yaml_path)
    adapter.apply_values({"sdr.auto_gain": True})
    assert cfg["sdr"]["auto_gain"] is True
    eng._mgc_last_mono = 0.0
    eng._maybe_auto_mgc(None)
    assert cfg["sdr"]["gain_db"] == 33
    assert src.gain_db == 33.0


def test_gain_write_with_auto_true_keeps_auto(tmp_path: Path) -> None:
    from fpvscan.app.adapters.engine_adapter import EngineAdapter

    eng, cfg, _src = _engine(gain_db=35, auto_gain=True)
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("sdr:\n  gain_db: 35\n", encoding="utf-8")
    adapter = EngineAdapter(eng, yaml_path)
    adapter.apply_values({"sdr.gain_db": 40, "sdr.auto_gain": True})
    assert cfg["sdr"]["auto_gain"] is True
    assert cfg["sdr"]["gain_db"] == 40
