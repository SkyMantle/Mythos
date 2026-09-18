from __future__ import annotations

from queue import Empty, Queue
from types import SimpleNamespace
import time

import numpy as np

from fpvscan.dsp.spectrum import find_occupied
from fpvscan.engine import Detection, Engine
from fpvscan.scan_hits import LOCK_HOLD_HZ, LOCK_SPURIOUS_HZ, MERGE_LOCK_HZ
from fpvscan.scan_view import (
    affect_of,
    cluster_step_hz,
    coarse_sweep_len,
    densify_around,
    extras_for_hit,
    grid_snapshot,
    lock_spectrum_due,
    lock_spectrum_every,
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
    assert step == 12e6
    cluster = sweep_step_hz(CFG["scan"], cluster=True)
    assert cluster == 4.0e6
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
    assert sweep_step_hz(scan) == 12e6
    extras = extras_for_hit(near, 3489e6, full_scan)
    dense = densify_around(3489e6, full_scan)
    assert nearest_sweep_hz(dense, 3489e6) is not None
    assert abs(nearest_sweep_hz(dense, 3489e6) - 3489e6) <= 2.0e6
    assert extras
    covered = [near, *extras]
    assert abs(nearest_sweep_hz(covered, 3489e6) - 3489e6) <= 2.0e6
    assert len(extras) < 20
    assert extras_for_hit(near, float(near) + 35e6, full_scan) == []
    assert extras_for_hit(2412e6, 2412e6, full_scan)


def test_cluster_step_4_survives_coarse_step_hz() -> None:
    """Live config has step_hz 12e6; that must not flatten 4 MHz extras."""
    scan = {
        "start_hz": 400e6,
        "stop_hz": 6000e6,
        "sample_rate": 35e6,
        "channel_bw_hz": 16e6,
        "step_hz": 12e6,
        "cluster_step_mhz": "4",
    }
    assert sweep_step_hz(scan) == 12e6
    assert sweep_step_hz(scan, cluster=True) == 4e6
    extras = extras_for_hit(4988e6, 4988e6, scan)
    assert extras
    assert min(extras) >= 4960e6
    assert max(extras) <= 5016e6
    assert not any(hz < 4920e6 for hz in extras)
    assert abs(min(extras) - 4968e6) <= 4e6
    assert any(abs(hz - 4992e6) <= 2e6 for hz in extras)


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
    assert tracked["cursor_hz"] == 4988e6
    assert tracked["center_hz"] == 4988e6
    unpinned = spectrum_snapshot(
        {**CFG, "video": {**CFG["video"], "spectrum_pin_center": False}},
        mode="LOCK", lock_target=4988e6, tuned_hz=4988e6, last=last,
        afc_hz=120e3,
    )
    assert unpinned["cursor_hz"] == 4988e6 + 120e3
    assert unpinned["center_hz"] == 4988e6


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


def test_lock_spectrum_every_reads_live_catalog_flag() -> None:
    assert lock_spectrum_every(None) == 1
    assert lock_spectrum_every({}) == 1
    assert lock_spectrum_every({"spectrum_every": 1}) == 1
    assert lock_spectrum_every({"spectrum_every": 16}) == 16
    assert lock_spectrum_every({"spectrum_every_4": False, "spectrum_every": 1}) == 1
    assert lock_spectrum_every({"spectrum_every_4": True, "spectrum_every": 1}) == 4
    assert lock_spectrum_every({"spectrum_every_4": True, "spectrum_every": 16}) == 4
    assert affect_of("video.spectrum_every_4") == "spectrum"


def test_pending_keys_only_retune_or_next_cycle() -> None:
    keys, reasons = pending_for(["sdr.gain_db", "video.sample_rate", "scan.threshold_db"])
    assert keys == ["video.sample_rate"]
    assert reasons["video.sample_rate"] == (
        "request recorded; session source rate remains shared"
    )
    cleared, _ = pending_for(
        ["video.sample_rate", "scan.start_hz"], lock_refreshed=True,
    )
    assert cleared == ["scan.start_hz"]
    assert affect_of("video.average") == "picture"
    assert affect_of("scan.start_hz") == "grid"
    assert affect_of("scan.cluster_step_mhz") == "grid"
    assert affect_of("scan.hit_filter") == "detections"
    assert not needs_lock_refresh([
        "video.deviation_hz", "video.capture_ms", "video.sharpen",
        "video.average", "video.h_pll", "video.width",
    ])
    assert not needs_lock_refresh(["video.sample_rate"])
    assert needs_lock_refresh(["video.lo_offset_hz"])
    assert needs_lock_refresh(["video.h_pll"]) is False
    assert not needs_lock_refresh(["scan.start_hz"])


def test_lock_metrics_separate_processed_iteration_and_ws() -> None:
    eng = Engine(_StubSrc(), CFG, Queue())
    eng._fps_ema = 7.5
    eng._iteration_fps_ema = 9.25
    eng._frame_seq = 12
    eng._frame_mono = time.monotonic()
    eng._ws_frame_ts = time.perf_counter() - 0.1
    eng.note_ws_frame_emitted(10)
    eng._timings.update({"line_hunt": 3.2, "fallback": 1.4, "lock_total": 8.8})

    snap = eng.snapshot()
    assert snap["processed_fps"] == 7.5
    assert snap["iteration_fps"] == 9.25
    assert snap["emitted_fps"] > 0.0
    assert snap["frame_seq"] == 12
    assert snap["emitted_frame_seq"] == 10
    assert snap["frame_age_ms"] is not None
    assert snap["timings_ms"]["line_hunt"] == 3.2
    assert snap["timings_ms"]["fallback"] == 1.4


def test_stale_good_lock_is_not_held_indefinitely() -> None:
    eng = Engine(_StubSrc(), CFG, Queue())
    eng._lock_good = object()
    eng._lock_good_at = time.monotonic() - 2.0
    assert eng._fresh_lock_good(max_age_s=0.25) is None
    assert eng._lock_good is None


def _analog_lock_frame(*, free_run: bool = False, locked: bool = True):
    from fpvscan.dsp import cvbs

    luma = np.full((288, 48), 90, dtype=np.uint8)
    luma[:, 12:28] = 170
    luma[1::2, :] = np.clip(luma[1::2, :].astype(np.int16) + 18, 0, 255).astype(
        np.uint8,
    )
    return cvbs.Frame(
        luma=luma,
        line_rate=15_625.0,
        lines=288,
        standard="PAL",
        locked=locked,
        free_run=free_run,
    )


class _HoldAssembler:
    period = 85.0
    standard = "PAL"

    def __init__(self, *, comb: bool = True, snow=None) -> None:
        self._comb = comb
        self.snow_calls = 0
        self._snow = snow

    def line_comb_present(self, *, field_span: float = 1.2) -> bool:
        return bool(self._comb)

    def latest_free_run(self, *, max_lines: int = 288):
        self.snow_calls += 1
        return self._snow


def test_lock_hold_does_not_publish_snow_over_analog() -> None:
    from fpvscan.dsp import cvbs

    eng = Engine(_StubSrc(), CFG, Queue())
    good = _analog_lock_frame()
    assert cvbs.analog_usable(good)
    eng._remember_good_frame(good)
    snow = _analog_lock_frame(free_run=True, locked=False)
    snow.luma = np.random.default_rng(3).integers(0, 255, snow.luma.shape, dtype=np.uint8)
    assembler = _HoldAssembler(comb=True, snow=snow)
    eng._snow_frame_at = 0.0
    shown = eng._choose_lock_display(
        None, assembler, {"snow_interval_ms": 1.0, "lock_good_hold_s": 5.0},
    )
    assert shown is not None
    assert shown.locked and not shown.free_run
    assert assembler.snow_calls == 0
    eng._lock_good_at = time.monotonic() - 2.0
    shown2 = eng._choose_lock_display(
        None, assembler, {"snow_interval_ms": 1.0, "lock_good_hold_s": 5.0},
    )
    assert shown2 is not None
    assert shown2.locked and not shown2.free_run


def test_true_snow_without_analog_still_publishes_unlocked_snow() -> None:
    eng = Engine(_StubSrc(), CFG, Queue())
    snow = _analog_lock_frame(free_run=True, locked=False)
    assembler = _HoldAssembler(comb=False, snow=snow)
    eng._snow_frame_at = 0.0
    shown = eng._choose_lock_display(
        None, assembler, {"snow_interval_ms": 1.0, "lock_good_hold_s": 5.0},
    )
    assert shown is not None
    assert shown.free_run
    assert not shown.locked
    assert assembler.snow_calls == 1


def test_lock_hold_expires_without_h_comb_then_allows_snow() -> None:
    eng = Engine(_StubSrc(), CFG, Queue())
    eng._remember_good_frame(_analog_lock_frame())
    eng._lock_good_at = time.monotonic() - 1.0
    snow = _analog_lock_frame(free_run=True, locked=False)
    assembler = _HoldAssembler(comb=False, snow=snow)
    eng._snow_frame_at = 0.0
    shown = eng._choose_lock_display(
        None, assembler, {"snow_interval_ms": 1.0, "lock_good_hold_s": 5.0},
    )
    assert eng._lock_good is None
    assert shown is not None and shown.free_run


class _StubSrc:
    name = "stub"
    sample_rate = 20e6
    overflows = 0
    clip_frac = 0.0
    bias_tee = False


def test_publish_spectrum_refreshes_cached_grid() -> None:
    events = Queue()
    eng = Engine(_StubSrc(), CFG, events)
    eng.state.mode = "SWEEP"
    eng._sweep_i = 20
    eng.state.sweep_pos_hz = 1200e6
    eng.state.tuned_hz = 1200e6
    assert eng.snapshot()["grid"]["pass_index"] == 0
    eng._publish_spectrum(1200e6, 35e6, [-40.0, -20.0], -50.0, 8192)
    snap = eng.snapshot()
    assert snap["grid"]["pass_index"] == 20
    assert snap["grid"]["current_hz"] == 1200e6
    assert snap["sweep_pos_hz"] == 1200e6
    found = []
    while True:
        try:
            found.append(events.get_nowait())
        except Empty:
            break
    spec = next(ev["data"] for ev in found if ev["type"] == "spectrum")
    assert spec["grid"]["pass_index"] == 20
    assert spec["grid"]["current_hz"] == 1200e6


def test_auto_peek_expires_when_ring_empty() -> None:
    eng = Engine(_StubSrc(), CFG, Queue())
    eng.state.mode = "LOCK"
    eng.state.auto = True
    eng.state.lock_target = 1200e6
    eng.state.auto_until = 0.0
    assert eng._end_auto_peek_if_due() is True
    assert eng.state.mode == "SWEEP"
    assert eng.state.lock_target is None
    assert eng.state.auto is False


def test_lock_command_sets_target_and_emits_state() -> None:
    events = Queue()
    eng = Engine(_StubSrc(), CFG, events)
    eng._handle_command("lock", {"freq_hz": 4988e6})
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 4988e6
    assert eng.state.auto is False
    found = []
    while True:
        try:
            found.append(events.get_nowait())
        except Empty:
            break
    kinds = [ev["type"] for ev in found]
    assert "spectrum" in kinds
    assert "state" in kinds
    snap = next(ev["data"] for ev in found if ev["type"] == "state")
    assert snap["mode"] == "LOCK"
    assert snap["lock_target"] == 4988e6


def test_inspect_skips_when_lock_queued() -> None:
    class _SpySrc:
        name = "stub"
        sample_rate = 20e6
        overflows = 0
        clip_frac = 0.0
        bias_tee = False

        def __init__(self) -> None:
            self.retunes: list[float] = []

        def retune_and_read(self, f, n):
            self.retunes.append(float(f))
            return np.zeros(8, np.complex64)

    src = _SpySrc()
    eng = Engine(src, CFG, Queue())
    eng.command("lock", freq_hz=4988e6)
    occ = SimpleNamespace(center_hz=1200e6, bandwidth_hz=8e6, snr_db=20.0)
    eng._inspect(None, 1200e6, 20e6, occ)
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 4988e6
    assert src.retunes == []


def _inspect(freq_hz: float, **kw) -> Detection:
    now = float(kw.get("seen", 1.0))
    return Detection(
        freq_hz=freq_hz,
        bandwidth_hz=8e6,
        snr_db=float(kw.get("snr", 20.0)),
        standard=kw.get("standard", "PAL"),
        confidence=float(kw.get("confidence", 0.7)),
        band="3G3",
        first_seen=now,
        last_seen=now,
        pic_score=float(kw.get("pic_score", 0.25)),
        row_corr=float(kw.get("row_corr", 0.2)),
        pic_locked=bool(kw.get("pic_locked", False)),
    )


def test_close_inspects_merge_to_one_hit() -> None:
    events = Queue()
    cfg = {**CFG, "scan": {**CFG["scan"], "confirm_hits": 2, "hit_filter": "all"}}
    eng = Engine(_StubSrc(), cfg, events)
    base = 3597e6
    eng._merge(_inspect(base, seen=10.0, snr=18.0, pic_score=0.22))
    eng._merge(_inspect(base + 1, seen=11.0, snr=21.0, pic_score=0.31))
    assert len(eng.state.detections) == 1
    det = next(iter(eng.state.detections.values()))
    assert det.freq_hz == base
    assert det.first_seen == 10.0
    assert det.snr_db == 21.0
    assert det.pic_score == 0.31
    assert det.hits == 2
    published = eng._published_detections()
    assert len(published) == 1
    assert published[0]["freq_hz"] == base

    eng2 = Engine(_StubSrc(), cfg, Queue())
    eng2._merge(_inspect(base, seen=1.0))
    eng2._merge(_inspect(base + 10e3, seen=2.0, snr=22.0, pic_score=0.4))
    assert len(eng2.state.detections) == 1
    merged = next(iter(eng2.state.detections.values()))
    assert merged.freq_hz == base
    assert merged.first_seen == 1.0
    assert merged.hits == 2


def test_lock_skips_tiny_delta() -> None:
    events = Queue()
    eng = Engine(_StubSrc(), CFG, events)
    eng._handle_command("lock", {"freq_hz": 3597e6})
    gen = eng._lock_gen
    target = eng.state.lock_target
    tuned = eng._lock_tuned
    while True:
        try:
            events.get_nowait()
        except Empty:
            break
    eng._handle_command("lock", {"freq_hz": 3597e6 + 1})
    assert eng.state.lock_target == target
    assert eng._lock_gen == gen
    assert eng._lock_tuned is tuned
    leftover = []
    while True:
        try:
            leftover.append(events.get_nowait())
        except Empty:
            break
    assert leftover == []
    eng._handle_command("lock", {"freq_hz": 3597e6 + 10e3})
    assert eng.state.lock_target == target
    assert eng._lock_gen == gen
    eng._handle_command("lock", {"freq_hz": 3597e6 + LOCK_HOLD_HZ + 1e3})
    assert eng._lock_gen == gen + 1
    assert eng.state.lock_target == 3597e6 + LOCK_HOLD_HZ + 1e3
    eng._handle_command("lock", {"freq_hz": eng.state.lock_target, "force": True})
    assert eng._lock_gen == gen + 2


def test_lock_nudge_01_mhz_retunes_despite_nearby_hit() -> None:
    events = Queue()
    cfg = {**CFG, "scan": {**CFG["scan"], "confirm_hits": 2, "hit_filter": "all"}}
    eng = Engine(_StubSrc(), cfg, events)
    base = 3597e6
    eng._merge(_inspect(base, seen=1.0))
    eng._merge(_inspect(base + 1, seen=2.0))
    eng._handle_command("lock", {"freq_hz": base})
    gen = eng._lock_gen
    assert len(eng.state.detections) == 1
    eng._handle_command("lock", {"freq_hz": base + 100e3})
    assert eng._lock_gen == gen + 1
    assert eng.state.lock_target == base + 100e3
    published = eng._published_detections()
    assert len(published) == 1
    assert published[0]["freq_hz"] == base + 100e3
    eng._handle_command("lock", {"freq_hz": base + 100e3 + 1})
    assert eng._lock_gen == gen + 1
    assert eng.state.lock_target == base + 100e3
    eng._handle_command("lock", {"freq_hz": base + 200e3, "force": True})
    assert eng._lock_gen == gen + 2
    assert eng.state.lock_target == base + 200e3
    assert LOCK_SPURIOUS_HZ == 20e3
    assert LOCK_HOLD_HZ == LOCK_SPURIOUS_HZ


def test_locked_row_ignores_khz_interp() -> None:
    events = Queue()
    cfg = {**CFG, "scan": {**CFG["scan"], "confirm_hits": 2, "hit_filter": "all"}}
    eng = Engine(_StubSrc(), cfg, events)
    base = 3597e6
    eng._merge(_inspect(base, seen=1.0))
    eng._merge(_inspect(base + 1, seen=2.0))
    eng._handle_command("lock", {"freq_hz": base})
    eng._merge(_inspect(base + 1500, seen=3.0, snr=24.0))
    assert len(eng.state.detections) == 1
    det = next(iter(eng.state.detections.values()))
    assert det.freq_hz == base
    assert det.first_seen == 1.0
    assert eng._published_detections()[0]["freq_hz"] == base


def test_locked_row_pins_afc_residual_to_lock_target() -> None:
    events = Queue()
    cfg = {**CFG, "scan": {**CFG["scan"], "confirm_hits": 2, "hit_filter": "all"}}
    eng = Engine(_StubSrc(), cfg, events)
    lock_hz = 4990.5e6
    inspect_hz = 4989e6
    neighbor_hz = lock_hz + 10e6
    eng._merge(_inspect(inspect_hz, seen=1.0))
    eng._merge(_inspect(inspect_hz, seen=2.0))
    first = next(iter(eng.state.detections.values())).first_seen
    eng._handle_command("lock", {"freq_hz": lock_hz})
    eng._merge(_inspect(inspect_hz, seen=3.0, snr=24.0))
    assert len(eng.state.detections) == 1
    det = next(iter(eng.state.detections.values()))
    assert det.freq_hz == lock_hz
    assert det.first_seen == first
    published = eng._published_detections()
    assert len(published) == 1
    assert published[0]["freq_hz"] == lock_hz
    gen = eng._lock_gen
    eng._handle_command("lock", {"freq_hz": lock_hz + 100e3})
    assert eng._lock_gen == gen + 1
    assert eng.state.lock_target == lock_hz + 100e3
    eng._merge(_inspect(neighbor_hz, seen=4.0))
    eng._merge(_inspect(neighbor_hz, seen=5.0))
    freqs = sorted(d["freq_hz"] for d in eng._published_detections())
    assert freqs == [lock_hz + 100e3, neighbor_hz]
    assert MERGE_LOCK_HZ == 2e6


def test_rf_snap_once_when_pegged_and_locked() -> None:
    events = Queue()
    cfg = {**CFG, "scan": {**CFG["scan"], "confirm_hits": 2, "hit_filter": "all"}}
    eng = Engine(_StubSrc(), cfg, events)
    lock = 4988e6
    cap = 1.5e6
    eng._merge(_inspect(lock, seen=1.0, pic_locked=True, pic_score=0.5))
    eng._merge(_inspect(lock, seen=2.0, pic_locked=True, pic_score=0.5))
    eng._handle_command("lock", {"freq_hz": lock})
    eng._afc = -cap
    gen = eng._lock_gen
    assert eng._maybe_rf_snap(pic_locked=True, pic_score=0.55, digital_max_hz=cap)
    assert eng.state.lock_target == lock - cap
    assert eng.state.tuned_hz == lock - cap
    assert eng._afc == 0.0
    assert eng._lock_tuned is None
    assert eng._lock_gen == gen + 1
    assert eng._published_detections()[0]["freq_hz"] == lock - cap
    eng._afc = -cap
    assert not eng._maybe_rf_snap(pic_locked=True, pic_score=0.55, digital_max_hz=cap)
    assert eng.state.lock_target == lock - cap
    assert eng._afc == -cap


def test_rf_snap_skipped_without_picture() -> None:
    eng = Engine(_StubSrc(), CFG, Queue())
    lock = 4988e6
    cap = 1.5e6
    eng._handle_command("lock", {"freq_hz": lock})
    eng._afc = cap
    gen = eng._lock_gen
    assert not eng._maybe_rf_snap(pic_locked=False, pic_score=0.0, digital_max_hz=cap)
    assert not eng._maybe_rf_snap(pic_locked=True, pic_score=0.10, digital_max_hz=cap)
    assert eng.state.lock_target == lock
    assert eng._afc == cap
    assert eng._lock_gen == gen
