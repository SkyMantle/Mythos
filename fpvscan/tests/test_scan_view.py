from __future__ import annotations

import numpy as np

from fpvscan.dsp.spectrum import find_occupied
from fpvscan.scan_view import (
    affect_of,
    cluster_step_hz,
    coarse_sweep_len,
    densify_around,
    extras_for_hit,
    grid_snapshot,
    lock_spectrum_due,
    nearest_sweep_hz,
    needs_lock_refresh,
    pending_for,
    spectrum_snapshot,
    sweep_centers,
    sweep_step_hz,
)


CFG = {
    "scan": {
        "start_hz": 400e6,
        "stop_hz": 6000e6,
        "sample_rate": 35e6,
        "channel_bw_hz": 10e6,
        "step_hz": 0,
        "fft_size": 8192,
        "priority_bands": False,
    },
    "video": {
        "sample_rate": 20e6,
        "channel_bw_hz": 12e6,
        "lo_offset_hz": 0.0,
    },
}


def test_sweep_step_matches_engine_formula() -> None:
    step = sweep_step_hz(CFG["scan"])
    fs, ch = 35e6, 10e6
    assert step == max(fs * 0.25, fs * 0.9 - ch / 2)
    cluster = sweep_step_hz(CFG["scan"], cluster=True)
    assert cluster == 8.0e6
    assert cluster < step
    assert cluster_step_hz({**CFG["scan"], "cluster_step_mhz": "off"}) is None
    assert sweep_step_hz({**CFG["scan"], "cluster_step_mhz": "4"}, cluster=True) == 4.0e6


def test_sweep_plan_visits_near_3489() -> None:
    """Coarse tile stays far; densify-on-hit puts 3489 on a 4 MHz extra."""
    scan = {
        "start_hz": 3400e6,
        "stop_hz": 3600e6,
        "sample_rate": 35e6,
        "channel_bw_hz": 10e6,
        "step_hz": 0,
    }
    centers = sweep_centers(scan, priority_bands=None)
    near = nearest_sweep_hz(centers, 3489e6)
    assert near is not None
    assert abs(near - 3489e6) <= 15.0e6
    full_scan = {**scan, "start_hz": 400e6, "stop_hz": 6000e6}
    full = sweep_centers(full_scan)
    assert abs(nearest_sweep_hz(full, 3489e6) - 3489e6) <= 15.0e6
    assert len(full) < 3 * coarse_sweep_len(full_scan)
    assert abs(len(full) - coarse_sweep_len(full_scan)) <= 2
    assert sweep_step_hz(scan) > 20e6
    extras = extras_for_hit(near, 3489e6, full_scan)
    dense = densify_around(3489e6, full_scan)
    assert nearest_sweep_hz(dense, 3489e6) is not None
    assert abs(nearest_sweep_hz(dense, 3489e6) - 3489e6) <= 2.0e6
    assert extras
    assert abs(nearest_sweep_hz(extras, 3489e6) - 3489e6) <= 2.0e6
    assert len(extras) < 20
    assert extras_for_hit(near, float(near) + 25e6, full_scan) == []
    assert extras_for_hit(2412e6, 2412e6, full_scan) == []


def test_cluster_step_8_vs_4_plan_size() -> None:
    scan = {
        "start_hz": 400e6,
        "stop_hz": 6000e6,
        "sample_rate": 35e6,
        "channel_bw_hz": 10e6,
        "step_hz": 0,
        "cluster_step_mhz": "8",
    }
    coarse = coarse_sweep_len(scan)
    assert abs(len(sweep_centers(scan)) - coarse) <= 2
    e8 = extras_for_hit(3497e6, 3489e6, scan)
    e4 = extras_for_hit(3497e6, 3489e6, {**scan, "cluster_step_mhz": "4"})
    e12 = extras_for_hit(3497e6, 3489e6, {**scan, "cluster_step_mhz": "12"})
    assert e8 and e4 and e12
    assert len(e8) < len(e4)
    assert len(e12) <= len(e8)
    assert extras_for_hit(3497e6, 3489e6, {**scan, "cluster_step_mhz": "off"}) == []
    assert abs(nearest_sweep_hz(e8, 3489e6) - 3489e6) <= 2.0e6


def test_sweep_plan_visits_near_1280() -> None:
    scan = {
        "start_hz": 1100e6,
        "stop_hz": 1400e6,
        "sample_rate": 35e6,
        "channel_bw_hz": 10e6,
        "step_hz": 0,
    }
    centers = sweep_centers(scan, priority_bands=None)
    near = nearest_sweep_hz(centers, 1280e6)
    assert near is not None
    assert abs(near - 1280e6) <= 15.0e6
    full = sweep_centers({**scan, "start_hz": 400e6, "stop_hz": 6000e6})
    assert abs(nearest_sweep_hz(full, 1280e6) - 1280e6) <= 15.0e6


def test_occupied_peak_interpolates_off_dwell() -> None:
    nfft = 4096
    fs = 35e6
    center = 3491.5e6
    psd = np.full(nfft, -42.0)
    bin_hz = fs / nfft
    peak = nfft // 2 + int(-2.0e6 / bin_hz)
    half = int(3.0e6 / bin_hz)
    xs = np.arange(-half, half)
    psd[peak - half: peak + half] = -18.0 - (xs.astype(np.float64) ** 2) * 1.2e-4
    occ = find_occupied(psd, center, fs, threshold_db=5.0, min_bw_hz=2e6)
    assert occ
    assert abs(occ[0].center_hz - (center - 2.0e6)) <= 1.0e6
    assert abs(occ[0].center_hz - center) <= 8.0e6


def test_spectrum_lock_follows_target_and_cfg() -> None:
    last = {"bins": [-30.0, -10.0], "floor_db": -30.0, "nfft": 2048, "center_hz": 3e9}
    view = spectrum_snapshot(
        CFG, mode="LOCK", lock_target=4988e6, tuned_hz=3e9, last=last,
    )
    assert view["cursor_hz"] == 4988e6
    assert view["center_hz"] == 4988e6
    assert view["bw_hz"] == 12e6
    assert view["span_hz"] == 20e6
    assert view["peak_db"] == -10.0
    assert view["bins"] == [-30.0, -10.0]
    assert isinstance(view["t_mono_ms"], int)
    assert view["t_mono_ms"] > 0

    moved = spectrum_snapshot(
        CFG, mode="LOCK", lock_target=4993e6, tuned_hz=4988e6, last=last,
    )
    assert moved["cursor_hz"] == 4993e6
    assert moved["center_hz"] == 4993e6

    cfg = {**CFG, "video": {**CFG["video"], "channel_bw_hz": 8e6, "lo_offset_hz": 2e6}}
    off = spectrum_snapshot(
        cfg, mode="LOCK", lock_target=4988e6, tuned_hz=4988e6, last=last,
    )
    assert off["bw_hz"] == 8e6
    assert off["center_hz"] == 4988e6 + 2e6
    assert off["cursor_hz"] == 4988e6
    tracked = spectrum_snapshot(
        CFG, mode="LOCK", lock_target=4988e6, tuned_hz=4988e6, last=last,
        afc_hz=120e3,
    )
    assert tracked["cursor_hz"] == 4988e6 + 120e3
    assert tracked["center_hz"] == 4988e6


def test_grid_from_real_scan_bounds() -> None:
    grid = grid_snapshot(
        CFG, mode="SWEEP", sweep_i=10, sweeps_done=2,
        sweep_pos_hz=1200e6, tuned_hz=1200e6, lock_target=None,
        visiting_hz=[1180e6, 1200e6],
    )
    assert grid["start_hz"] == 400e6
    assert grid["stop_hz"] == 6000e6
    assert grid["step_hz"] == sweep_step_hz(CFG["scan"])
    assert grid["pass_index"] == 10
    assert grid["pass_count"] > 10
    assert 0 < grid["progress_01"] < 1
    assert grid["current_hz"] == 1200e6
    assert grid["visiting_hz"] == [1180e6, 1200e6]
    assert grid["passes_done"] == 2
    locked = grid_snapshot(
        CFG, mode="LOCK", sweep_i=10, sweeps_done=2,
        sweep_pos_hz=1200e6, tuned_hz=1200e6, lock_target=4988e6,
        afc_hz=-80e3,
    )
    assert locked["current_hz"] == 4988e6 - 80e3
    assert isinstance(grid["t_mono_ms"], int)
    hopped = grid_snapshot(
        CFG, mode="SWEEP", sweep_i=11, sweeps_done=2,
        sweep_pos_hz=1200e6, tuned_hz=1200e6, lock_target=None,
        next_hz=1235e6, dwell_ms=42.0, t_mono_ms=9001,
    )
    assert hopped["next_hz"] == 1235e6
    assert hopped["dwell_ms"] == 42.0
    assert hopped["t_mono_ms"] == 9001


def test_lock_spectrum_due_every_iq_when_every_is_one() -> None:
    assert lock_spectrum_due(1, 1)
    assert lock_spectrum_due(2, 1)
    assert lock_spectrum_due(8, 1)
    assert lock_spectrum_due(1, 16)
    assert not lock_spectrum_due(2, 16)
    assert lock_spectrum_due(17, 16)
    assert not lock_spectrum_due(0, 1)


def test_pending_keys_only_retune_or_next_cycle() -> None:
    keys, reasons = pending_for(["sdr.gain_db", "video.sample_rate", "scan.threshold_db"])
    assert keys == ["video.sample_rate"]
    assert reasons["video.sample_rate"] == "next lock retune"
    cleared, _ = pending_for(
        ["video.sample_rate", "scan.start_hz"], lock_refreshed=True,
    )
    assert cleared == ["scan.start_hz"]
    assert affect_of("video.average") == "picture"
    assert affect_of("scan.start_hz") == "grid"
    assert affect_of("scan.cluster_step_mhz") == "grid"
    assert affect_of("scan.hit_filter") == "detections"
    assert needs_lock_refresh(["video.sharpen"])
    assert not needs_lock_refresh(["scan.start_hz"])
