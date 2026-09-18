from __future__ import annotations

import numpy as np

from fpvscan.dsp.cvbs import (
    CROP_BOTTOM_LINES,
    CROP_LEFT_FRAC,
    DecodeState,
    Frame,
    USABLE_ROW_CORR,
    _h_crop,
    _h_unwrap,
    adopt_analog_picture,
    analog_usable,
    choose_lock_raster,
    decode,
    drop_dead_sticky,
    edge_col_stable,
    estimate_line_hz,
    free_run,
    h_blank_col,
    h_phase_col,
    h_phase_manual_px,
    h_phase_should_nudge,
    h_pit_in_mid_half,
    h_pit_unsafe,
    h_roll_step,
    h_sync_edge_col,
    hold_visible_lock,
    luma_mean,
    needs_lock_fallback,
    raster_is_black,
    row_correlation,
    should_save_still,
    complete_field_geometry,
    field_is_framed,
)
from fpvscan.dsp.demod import blob_offset_hz
from fpvscan.scan_view import (
    RF_SNAP_COOLDOWN_S,
    afc_is_pegged,
    afc_should_nudge,
    freeze_afc_hunt,
    freeze_lock_afc,
    hunt_span_hz,
    lock_channel_bw,
    lock_decimation,
    prelock_mix_hz,
    rf_snap_due,
)


def _mid_blank(w: int = 80) -> np.ndarray:
    img = np.full((40, w), 180, dtype=np.uint8)
    img[:, 36:44] = 30
    return img


def test_h_blank_finds_midframe_pit() -> None:
    col = h_blank_col(_mid_blank())
    assert 28 <= col <= 44


def test_h_unwrap_auto_parks_left_not_mid() -> None:
    img = _mid_blank()
    out = _h_unwrap(img, None, h_phase_frac=0.0)
    assert float(out[:, 36:44].mean()) > 150
    pit = h_blank_col(out)
    assert not h_pit_in_mid_half(pit if pit else 0, 80)
    quiet = _h_unwrap(img, None, h_phase_frac=-0.006)
    assert np.array_equal(out, quiet)


def test_h_phase_deadband_and_safe_manual() -> None:
    assert h_phase_manual_px(0.0, 640) == 0
    assert h_phase_manual_px(-0.006, 640) == 0
    assert h_phase_manual_px(-0.06, 640) != 0
    img = _mid_blank()
    auto = _h_unwrap(img, None, h_phase_frac=0.0)
    dumped = _h_unwrap(img, None, h_phase_frac=-0.06)
    pit = h_blank_col(dumped)
    dest = pit if pit else 0
    assert not h_pit_unsafe(dest, 80)
    assert float(dumped[:, 36:44].mean()) > 150
    # 0.25 would land in the mid-half → dropped, framing stays auto.
    rejected = _h_unwrap(img, None, h_phase_frac=0.25)
    assert np.array_equal(auto, rejected)
    # Small extra roll after park, still on the left porch.
    shifted = _h_unwrap(img, None, h_phase_frac=0.05)
    assert not np.array_equal(auto, shifted)
    assert not h_pit_unsafe(h_blank_col(shifted) or 0, 80)
    # Already parked: extra negative must not wrap blank onto the right.
    wrapped = _h_unwrap(auto, None, h_phase_frac=-0.06)
    assert np.array_equal(auto, wrapped)


def test_crop_after_park_does_not_wrap_mid() -> None:
    img = _mid_blank()
    parked = _h_unwrap(img, None, h_phase_frac=0.0)
    cropped = _h_crop(parked, CROP_LEFT_FRAC, CROP_BOTTOM_LINES)
    assert cropped.shape[1] < parked.shape[1]
    assert cropped.shape[0] == parked.shape[0] - CROP_BOTTOM_LINES
    # Slice, not a roll: mid-half stays picture, not the parked porch.
    mid = cropped[:, int(cropped.shape[1] * 0.25):int(cropped.shape[1] * 0.75)]
    assert float(mid.mean()) > 150
    assert float(mid.min()) > 100
    left = int(round(parked.shape[1] * CROP_LEFT_FRAC))
    assert np.array_equal(cropped, parked[:-CROP_BOTTOM_LINES, left:])


def test_h_sync_prefers_edge_over_dark_wardrobe() -> None:
    w, h = 80, 48
    img = np.full((h, w), 180, dtype=np.uint8)
    img[8:36, 36:50] = 8
    img[:, 22:27] = 55
    pit = h_blank_col(img)
    assert 28 <= pit <= 50
    edge = h_sync_edge_col(img)
    assert 18 <= edge <= 26
    assert h_phase_col(img) == edge
    out = _h_unwrap(img, None, h_phase_frac=0.0)
    assert h_phase_col(out) == 0
    assert not h_pit_unsafe(h_sync_edge_col(out) or 0, w)


def test_lock_channel_bw_keeps_9mhz_at_13msps() -> None:
    ch = lock_channel_bw(13e6, 9e6)
    assert abs(ch - 9e6) < 1.0
    assert lock_decimation(13e6, ch) == 1


def test_lock_channel_bw_keeps_9mhz_dec2_at_20msps() -> None:
    ch = lock_channel_bw(20e6, 9e6)
    assert abs(ch - 9e6) < 1.0
    assert lock_decimation(20e6, ch) == 2
    assert ch != 20e6


def test_lock_channel_bw_12mhz_at_20msps_is_dec1() -> None:
    ch = lock_channel_bw(20e6, 12e6)
    assert abs(ch - 12e6) < 1.0
    assert lock_decimation(20e6, ch) == 1


def test_lock_channel_bw_12mhz_at_30msps_is_dec2() -> None:
    ch = lock_channel_bw(30e6, 12e6, off_hz=0.0)
    assert abs(ch - 12e6) < 1.0
    assert lock_decimation(30e6, ch) == 2
    assert lock_decimation(30e6, 12e6) == 2


def test_afc_pegged_independent_of_freq_err() -> None:
    assert afc_is_pegged(-1.5e6, 1.5e6)
    assert afc_is_pegged(-1.5e6, 1.5e6, freq_err_hz=-100e3)
    assert not afc_is_pegged(-0.4e6, 1.5e6)
    assert not afc_should_nudge(-1.5e6, -100e3, 1.5e6)
    assert afc_should_nudge(-1.5e6, -2.3e6, 1.5e6)
    assert hunt_span_hz([0.25], pegged=False) == 0.25e6
    assert hunt_span_hz([0.25], pegged=True) == 2e6


def test_freeze_afc_hunt_when_pic_locked() -> None:
    assert freeze_afc_hunt(pic_locked=True, pic_score=0.55)
    assert not freeze_afc_hunt(pic_locked=True, pic_score=0.10)
    assert not freeze_afc_hunt(pic_locked=False, pic_score=0.80)
    assert freeze_lock_afc(analog_ok=True, pic_locked=False, pic_score=0.1)
    assert freeze_lock_afc(
        analog_ok=False, pic_locked=True, pic_score=0.20, row_corr=0.40)
    assert not freeze_lock_afc(
        analog_ok=False, pic_locked=False, pic_score=0.80, row_corr=0.05)
    assert not freeze_lock_afc(
        analog_ok=True, pic_locked=True, pic_score=0.55, row_corr=0.40,
        luma_mean=2.0)
    assert freeze_lock_afc(
        analog_ok=True, pic_locked=False, pic_score=0.1, luma_mean=80.0)
    assert not afc_should_nudge(-1.5e6, -2.3e6, 1.5e6, pic_locked=True)
    assert afc_should_nudge(-1.5e6, -2.3e6, 1.5e6, pic_locked=False)
    assert hunt_span_hz([0.25], pegged=False) == 0.25e6


def test_rf_snap_due_pegged_locked_not_unlocked() -> None:
    cap = 1.5e6
    pegged = -0.95 * cap
    assert rf_snap_due(
        pic_locked=True, pic_score=0.55, afc_hz=pegged, digital_max_hz=cap,
        last_snap_mono=0.0, now_mono=10.0)
    assert not rf_snap_due(
        pic_locked=False, pic_score=0.80, afc_hz=pegged, digital_max_hz=cap,
        last_snap_mono=0.0, now_mono=10.0)
    assert not rf_snap_due(
        pic_locked=True, pic_score=0.10, afc_hz=pegged, digital_max_hz=cap,
        last_snap_mono=0.0, now_mono=10.0)
    assert not rf_snap_due(
        pic_locked=True, pic_score=0.55, afc_hz=-0.4e6, digital_max_hz=cap,
        last_snap_mono=0.0, now_mono=10.0)


def test_rf_snap_due_cooldown_blocks_spam() -> None:
    cap = 1.5e6
    kw = dict(pic_locked=True, pic_score=0.55, afc_hz=-cap, digital_max_hz=cap)
    assert not rf_snap_due(
        **kw, last_snap_mono=1.0, now_mono=1.0 + RF_SNAP_COOLDOWN_S - 0.1)
    assert rf_snap_due(
        **kw, last_snap_mono=1.0, now_mono=1.0 + RF_SNAP_COOLDOWN_S + 0.1)


def test_h_phase_deadzone_freeze_and_stable_edge() -> None:
    # ±2° of a 64 µs line ≈ 0.0056 — inside the 0.01 dead zone.
    assert not h_phase_should_nudge(0.0056)
    assert not h_phase_should_nudge(0.009)
    assert h_phase_should_nudge(0.02)
    assert not h_phase_should_nudge(0.05, pic_locked=True)
    assert not h_phase_should_nudge(0.05, edge_stable=False)
    assert not edge_col_stable([10, 10])
    assert edge_col_stable([10, 10, 11])
    assert not edge_col_stable([10, 18, 4])
    assert h_roll_step(None, 40, 80) == 40
    assert h_roll_step(40, 41, 80) == 40
    assert h_roll_step(40, 0, 80) == 0
    assert h_roll_step(0, 50, 80, pic_locked=True) == 0
    assert h_roll_step(0, 50, 80, edge_stable=False) == 0
    stepped = h_roll_step(40, 50, 80, edge_stable=True)
    assert stepped != 40
    assert abs(stepped - 40) < abs(50 - 40)


def test_h_unwrap_state_does_not_chase_one_px() -> None:
    img = _mid_blank()
    st = DecodeState()
    out = _h_unwrap(img, st, h_phase_frac=0.0)
    parked = int(st.h_roll or 0)
    assert parked != 0
    jitter = np.roll(out, 1, axis=1)
    _h_unwrap(jitter, st, h_phase_frac=0.0)
    # Parked raster + 1 px must not keep walking h_roll.
    assert st.h_roll in (0, parked)
    locked = DecodeState()
    _h_unwrap(img, locked, h_phase_frac=0.0)
    first = locked.h_roll
    _h_unwrap(img, locked, h_phase_frac=0.0, pic_locked=True)
    assert locked.h_roll == first


def test_prelock_mix_follows_blob_until_tracking() -> None:
    assert prelock_mix_hz(2.0e6, tracking=True, fs=20e6, ch_bw=12e6) == 0.0
    assert prelock_mix_hz(40e3, tracking=False, fs=20e6, ch_bw=12e6) == 0.0
    pulled = prelock_mix_hz(-2.0e6, tracking=False, fs=20e6, ch_bw=12e6)
    assert pulled == -2.0e6
    capped = prelock_mix_hz(8e6, tracking=False, fs=20e6, ch_bw=12e6, max_hz=2.5e6)
    assert capped == 2.5e6


def test_blob_offset_finds_shifted_tone() -> None:
    fs = 10e6
    n = 4096
    f0 = 1.2e6
    t = np.arange(n, dtype=np.float64) / fs
    iq = np.exp(2j * np.pi * f0 * t).astype(np.complex64)
    assert abs(blob_offset_hz(iq, fs) - f0) < 150e3


def _pal_lines(fs: float = 8e6, n_lines: int = 400) -> np.ndarray:
    period = int(round(fs / 15625.0))
    line = np.linspace(0.35, 1.0, period, dtype=np.float32)
    line[: max(4, int(fs * 4.7e-6))] = 0.0
    return np.tile(line, n_lines)


def _corr_luma(h: int = 80, w: int = 64, noise: float = 0.0) -> np.ndarray:
    row = np.linspace(20, 220, w, dtype=np.float32)
    luma = np.tile(row, (h, 1))
    if noise:
        rng = np.random.default_rng(0)
        luma = luma + rng.normal(0.0, noise, luma.shape)
    return np.clip(luma, 0, 255).astype(np.uint8)


def test_adopt_analog_picture_promotes_high_corr_pal_free_run() -> None:
    """3290 Rabbit: corr~0.65, line~15618, V-sync missing → still a lock."""
    fr = Frame(
        luma=_corr_luma(), line_rate=15618.1, lines=288,
        standard="?", locked=False, free_run=True,
    )
    assert analog_usable(fr)
    assert adopt_analog_picture(fr) is True
    assert fr.locked and not fr.free_run
    assert fr.standard == "PAL"


def test_should_save_still_locked_analog_even_if_free_run_flag() -> None:
    """5473 sweep NOTE lied: locked+corr=0.938 is a still, not snow."""
    fr = Frame(
        luma=_corr_luma(), line_rate=15623.0, lines=288,
        standard="PAL", locked=True, free_run=True,
    )
    assert should_save_still(fr) is True
    snow = Frame(
        luma=np.random.default_rng(1).integers(0, 256, size=(80, 64), dtype=np.uint8),
        line_rate=15625.0, lines=288,
        standard="?", locked=False, free_run=True,
    )
    assert should_save_still(snow) is False


def test_free_run_skips_line_hz_hunt_with_period_hint(monkeypatch) -> None:
    """Known analog period must not pay for a second estimate_line_hz."""
    import fpvscan.dsp.cvbs as cvbs
    hunts = {"n": 0}
    real = cvbs.estimate_line_hz

    def counted(*a, **k):
        hunts["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(cvbs, "estimate_line_hz", counted)
    fs = 8e6
    base = _pal_lines(fs)
    hint = fs / 15625.0
    fr = cvbs.free_run(base, fs, width=160, max_lines=80, period_hint=hint)
    assert fr is not None
    assert hunts["n"] == 0
    assert abs(fr.line_rate - 15625.0) < 1.0


def test_choose_lock_raster_rejects_black_tracked() -> None:
    """Sticky predicted t0 must not ship 288 black lines as the only raster."""
    black = Frame(
        luma=np.zeros((288, 80), dtype=np.uint8),
        line_rate=15625.0, lines=288, standard="PAL", locked=True,
    )
    good = Frame(
        luma=_corr_luma(), line_rate=15625.0, lines=80,
        standard="PAL", locked=False, free_run=True,
    )
    assert raster_is_black(black)
    assert not analog_usable(black)
    assert needs_lock_fallback(black)
    assert choose_lock_raster(black, good) is good
    assert choose_lock_raster(black, None) is None
    assert choose_lock_raster(None, good) is good
    assert choose_lock_raster(good, black) is good
    weak = Frame(
        luma=_corr_luma(noise=40.0), line_rate=15625.0, lines=80,
        standard="?", locked=True,
    )
    # Low-corr but not black: keep it if free_run is missing.
    if not analog_usable(weak):
        assert choose_lock_raster(weak, None) is weak
        assert needs_lock_fallback(weak)


def test_engine_never_skips_fallback_on_black_non_none() -> None:
    """Engine LOCK path: non-None black decode must still run free_run."""
    black = Frame(
        luma=np.zeros((282, 80), dtype=np.uint8),
        line_rate=15625.0, lines=282, standard="PAL", locked=True,
    )
    fallback = Frame(
        luma=_corr_luma(), line_rate=15625.0, lines=80,
        standard="PAL", locked=False, free_run=True, field_t0=12.0,
    )
    assert black is not None
    assert needs_lock_fallback(black)
    analog_ok = analog_usable(black)
    assert not analog_ok
    show = black
    if needs_lock_fallback(show):
        show = choose_lock_raster(show, fallback)
    assert show is fallback
    assert analog_usable(show)
    assert not freeze_lock_afc(
        analog_ok=analog_ok,
        pic_locked=True,
        pic_score=0.9,
        row_corr=0.9,
        luma_mean=luma_mean(black),
    )
    st = DecodeState(period=512.0, sticky=True, abs_t0=8.0)
    drop_dead_sticky(st)
    assert not st.sticky
    assert st.abs_t0 is None
    assert st.period == 512.0
    hold_visible_lock(st, fallback, fs=8e6, abs_start=100.0)
    assert st.sticky
    assert abs(float(st.abs_t0) - 112.0) < 0.5
    assert not needs_lock_fallback(fallback)


def test_field_hold_ok_visible_sheared_analog() -> None:
    """Sheared-but-visible analog must hold sticky (mid-frame trough is ok)."""
    from fpvscan.dsp.cvbs import _v_blank_row, field_hold_ok, field_needs_resnap
    luma = _corr_luma(h=160)
    luma[70:95] = 40
    fr = Frame(
        luma=luma, line_rate=15625.0, lines=160,
        standard="PAL", locked=True,
    )
    assert analog_usable(fr)
    assert _v_blank_row(fr.luma, 25) is not None
    assert field_hold_ok(fr)
    assert not needs_lock_fallback(fr)
    # 160 lines is a short pack — resnap; a full field with an early OSD is not.
    assert field_needs_resnap(160, luma=luma)
    full = _corr_luma(h=288)
    full[60:82] = 40
    assert not field_needs_resnap(288, luma=full)


def test_analog_usable_rejects_snow_and_off_band() -> None:
    snow = Frame(
        luma=np.random.default_rng(1).integers(0, 256, size=(80, 64), dtype=np.uint8),
        line_rate=15625.0, lines=288,
        standard="?", locked=False, free_run=True,
    )
    assert row_correlation(snow.luma) < USABLE_ROW_CORR
    assert not analog_usable(snow)
    assert adopt_analog_picture(snow) is False
    assert snow.free_run and not snow.locked
    off = Frame(
        luma=_corr_luma(), line_rate=12000.0, lines=288,
        standard="?", locked=False, free_run=True,
    )
    assert not analog_usable(off)


def test_free_run_pal_lines_show_picture_not_black() -> None:
    fs = 8e6
    fr = free_run(_pal_lines(fs), fs, width=160, max_lines=80)
    assert fr is not None
    assert fr.free_run and not fr.locked
    assert fr.luma.shape[0] >= 32
    assert float(fr.luma.mean()) > 20
    assert row_correlation(fr.luma) > 0.4


def _lines_at(fs: float, line_hz: float, n_lines: int = 400) -> np.ndarray:
    period = int(round(fs / line_hz))
    line = np.linspace(0.35, 1.0, period, dtype=np.float32)
    line[: max(4, int(fs * 4.7e-6))] = 0.0
    return np.tile(line, n_lines)


def test_estimate_line_hz_finds_off_pal() -> None:
    fs = 8e6
    true_hz = 15420.0
    hz = estimate_line_hz(_lines_at(fs, true_hz), fs)
    assert hz is not None
    assert abs(hz - true_hz) < 40.0


def test_free_run_hunts_off_pal_instead_of_15625() -> None:
    """Hardcoded 15625 free-run shears a camera that is ~200 Hz off PAL."""
    fs = 8e6
    true_hz = 15420.0
    base = _lines_at(fs, true_hz)
    fr = free_run(base, fs, width=160, max_lines=80)
    assert fr is not None
    assert abs(fr.line_rate - true_hz) < 40.0
    assert abs(fr.line_rate - 15625.0) > 80.0
    assert row_correlation(fr.luma) > 0.35


def test_decode_keeps_off_pal_raster() -> None:
    fs = 8e6
    true_hz = 15420.0
    fr = decode(_lines_at(fs, true_hz), fs, width=160, max_lines=80, state=None)
    assert fr is not None
    assert fr.luma.size > 0
    assert abs(fr.line_rate - true_hz) < 80.0
    assert float(fr.luma.mean()) > 8


def test_decode_keeps_jittery_raster_instead_of_none() -> None:
    """Messy H-sync used to score ≤0.5 and return None (black LOCK pane)."""
    fs = 8e6
    period = int(round(fs / 15625.0))
    chunks = []
    rng = np.random.default_rng(0)
    for i in range(400):
        n = period + int(rng.integers(-3, 4))
        n = max(period - 4, n)
        line = np.linspace(0.35, 1.0, n, dtype=np.float32)
        line[: max(4, int(fs * 4.7e-6))] = 0.0
        if i % 23 == 0:
            line[: max(8, n // 8)] = 0.0
        chunks.append(line)
    base = np.concatenate(chunks)
    fr = decode(base, fs, width=160, max_lines=80, state=None)
    assert fr is not None
    assert fr.luma.size > 0
    assert float(fr.luma.mean()) > 8


def test_field_is_framed_rejects_short_ntsc_shear() -> None:
    torn = Frame(
        luma=np.full((234, 80), 90, np.uint8),
        line_rate=15734.0, lines=234, standard="NTSC", locked=True,
    )
    assert complete_field_geometry(torn)
    assert not field_is_framed(torn)
    full = Frame(
        luma=np.full((240, 80), 90, np.uint8),
        line_rate=15734.0, lines=240, standard="NTSC", locked=True,
    )
    assert field_is_framed(full)


def test_h_crop_does_not_eat_ntsc_active_field() -> None:
    ntsc = np.ones((240, 120), np.uint8) * 80
    cropped = _h_crop(ntsc, 0.0, 6)
    assert cropped.shape[0] == 240
    pal = np.ones((288, 120), np.uint8) * 80
    pal_out = _h_crop(pal, 0.0, 6)
    assert pal_out.shape[0] == 282
