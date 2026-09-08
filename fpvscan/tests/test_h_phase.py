from __future__ import annotations

import numpy as np

from fpvscan.dsp.cvbs import (
    CROP_BOTTOM_LINES,
    CROP_LEFT_FRAC,
    DecodeState,
    _h_crop,
    _h_unwrap,
    edge_col_stable,
    h_blank_col,
    h_phase_col,
    h_phase_manual_px,
    h_phase_should_nudge,
    h_pit_in_mid_half,
    h_pit_unsafe,
    h_roll_step,
    h_sync_edge_col,
)
from fpvscan.scan_view import (
    RF_SNAP_COOLDOWN_S,
    afc_is_pegged,
    afc_should_nudge,
    freeze_afc_hunt,
    hunt_span_hz,
    lock_channel_bw,
    lock_decimation,
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


def test_lock_dec_stays_one_at_30msps() -> None:
    ch = lock_channel_bw(30e6, 12e6, off_hz=0.0)
    assert lock_decimation(30e6, ch) == 1
    assert int(30e6 / ch) == 1
    ch20 = lock_channel_bw(20e6, 12e6)
    assert lock_decimation(20e6, ch20) == 1
    assert lock_decimation(30e6, 12e6) == 1


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
