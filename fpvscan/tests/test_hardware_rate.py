from __future__ import annotations

from queue import Queue

import numpy as np
import pytest

from fpvscan import hardware_rate, scan_view
from fpvscan.dsp import adaptive_if
from fpvscan.engine import Engine


def _cfg(*, step: float = 12e6, cap: float = 12e6,
         override: float | None = None) -> dict:
    sdr = {"gain_db": 30, "quick_tune": False}
    if override is not None:
        sdr["sample_rate"] = override
    return {
        "scan": {
            "start_hz": 1.1e9,
            "stop_hz": 1.3e9,
            "sample_rate": 35e6,
            "channel_bw_hz": cap,
            "step_hz": step,
            "fft_size": 256,
            "averages": 1,
            "priority_bands": False,
        },
        "video": {
            "sample_rate": 20e6,
            "channel_bw_hz": cap,
            "capture_ms": 20,
        },
        "sdr": sdr,
        "rotator": {"enable": False},
    }


class _Source:
    name = "fake"
    fixed_freq = False
    center_freq = 1.1e9
    overflows = 0
    clip_frac = 0.0
    adc_rms = 0.0
    bias_tee = False

    def __init__(self) -> None:
        self.sample_rate = 0.0
        self.rate_calls: list[float] = []

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def set_sample_rate(self, hz: float) -> None:
        self.rate_calls.append(float(hz))
        self.sample_rate = float(hz)

    def set_gain(self, _db: float) -> None:
        pass


def test_rate_derivation_is_frequency_agnostic_and_explicit() -> None:
    cfg = _cfg()
    plan = hardware_rate.derive_hardware_rate(cfg)

    assert plan.hardware_sample_rate_hz == 13.4e6
    assert plan.usable_span_hz == pytest.approx(12.06e6)
    assert plan.effective_sweep_step_hz == 12e6
    assert plan.requested_scan_rate_hz == 35e6
    assert plan.requested_video_rate_hz == 20e6
    assert "max(analogue channel cap, inspect bandwidth)" in plan.reason
    assert plan.estimated_sc16_mb_s == pytest.approx(53.6)
    assert hardware_rate.sc16_throughput_mb_s(35e6) == 140.0

    # RF center is not an input: arbitrary scan bounds cannot change the plan.
    moved = _cfg()
    moved["scan"].update(start_hz=4.4e9, stop_hz=5.1e9)
    assert hardware_rate.derive_hardware_rate(moved) == plan
    wider_cap = hardware_rate.derive_hardware_rate(_cfg(cap=16e6))
    assert wider_cap.hardware_sample_rate_hz == 17.8e6
    assert wider_cap.usable_span_hz == pytest.approx(16.02e6)


def test_rate_follows_analogue_skirt_not_sweep_step() -> None:
    cfg = _cfg(step=12e6, cap=16e6)
    cfg["scan"]["inspect_bw_hz"] = 18e6
    plan = hardware_rate.derive_hardware_rate(cfg)
    assert plan.hardware_sample_rate_hz == 20e6
    assert plan.usable_span_hz == pytest.approx(18e6)
    assert plan.effective_sweep_step_hz == 12e6
    assert plan.requested_sweep_step_hz == 12e6
    assert plan.requested_channel_cap_hz == 16e6
    moved = _cfg(step=12e6, cap=16e6)
    moved["scan"]["inspect_bw_hz"] = 18e6
    moved["scan"].update(start_hz=5.4e9, stop_hz=5.9e9)
    assert hardware_rate.derive_hardware_rate(moved) == plan


def test_gap_free_sweep_clamps_step_instead_of_raising_rate() -> None:
    cfg = _cfg(step=20e6, cap=8e6, override=10e6)
    plan = hardware_rate.derive_hardware_rate(cfg)
    hardware_rate.normalize_runtime_config(cfg, plan)
    centers = scan_view.sweep_centers(cfg["scan"])

    assert plan.hardware_sample_rate_hz == 10e6
    assert plan.usable_span_hz == 9e6
    assert plan.effective_sweep_step_hz == 9e6
    assert scan_view.sweep_step_hz(cfg["scan"]) == 9e6
    assert centers
    assert centers[0] - plan.usable_span_hz / 2 <= cfg["scan"]["start_hz"]
    assert centers[-1] + plan.usable_span_hz / 2 >= cfg["scan"]["stop_hz"]
    assert max(np.diff(centers), default=0.0) <= plan.usable_span_hz


def test_sweep_lock_transitions_never_change_shared_source_rate() -> None:
    source = _Source()
    engine = Engine(source, _cfg(), Queue())
    engine._configure_open_source()
    assert source.rate_calls == [13.4e6]

    engine._handle_command("lock", {"freq_hz": 1.234567e9, "force": True})
    engine._handle_command("sweep", {"auto_lock": False})

    assert source.rate_calls == [13.4e6]
    assert engine.cfg["scan"]["sample_rate"] == 13.4e6
    assert engine.cfg["video"]["sample_rate"] == 13.4e6
    snap = engine.snapshot()
    assert snap["hardware_sample_rate"] == 13.4e6
    assert snap["requested_scan_sample_rate"] == 35e6
    assert snap["requested_video_sample_rate"] == 20e6


def test_explicit_override_is_one_serialized_full_reconfigure() -> None:
    source = _Source()
    engine = Engine(source, _cfg(), Queue())
    engine._configure_open_source()
    target = hardware_rate.derive_hardware_rate(
        _cfg(override=15e6),
    )

    engine._reconfigure_hardware_rate_now(target)
    engine._reconfigure_hardware_rate_now(target)

    assert source.rate_calls == [13.4e6, 15e6]
    assert engine.cfg["scan"]["sample_rate"] == 15e6
    assert engine.cfg["video"]["sample_rate"] == 15e6
    assert engine.snapshot()["hardware_rate_plan"]["explicit_override"] is True


def test_adaptive_if_has_derived_rate_candidates_without_fixed_winner() -> None:
    plan = hardware_rate.derive_hardware_rate(_cfg())
    candidates = adaptive_if.generate_candidates(
        plan.hardware_sample_rate_hz,
        channel_cap_hz=plan.requested_channel_cap_hz,
    )

    assert [item.decimation for item in candidates] == [1, 2, 3]
    assert candidates[0].effective_bw_hz == pytest.approx(
        plan.hardware_sample_rate_hz * adaptive_if.FILTER_USABLE_FRAC,
    )
    assert candidates[0].cutoff_hz == candidates[0].effective_bw_hz / 2
    assert all(item.effective_bw_hz <= plan.usable_span_hz for item in candidates)


def test_stale_auto_tile_step_falls_back_to_12mhz_overlap() -> None:
    cfg = _cfg(step=23.802e6, cap=16e6)
    cfg["scan"]["inspect_bw_hz"] = 18e6
    plan = hardware_rate.derive_hardware_rate(cfg)
    assert plan.hardware_sample_rate_hz == 20e6
    assert plan.usable_span_hz == pytest.approx(18e6)
    assert plan.requested_sweep_step_hz == 12e6
    assert plan.effective_sweep_step_hz == 12e6
    hardware_rate.normalize_runtime_config(cfg, plan)
    assert scan_view.sweep_step_hz(cfg["scan"]) == 12e6
    assert cfg["scan"]["step_hz"] == 23.802e6


def test_missing_step_does_not_become_channel_bw_or_auto_tile() -> None:
    cfg = _cfg(step=12e6, cap=16e6)
    cfg["scan"]["step_hz"] = 0
    cfg["scan"]["inspect_bw_hz"] = 18e6
    plan = hardware_rate.derive_hardware_rate(cfg)
    assert plan.requested_sweep_step_hz == 12e6
    assert plan.effective_sweep_step_hz == 12e6
    assert scan_view.sweep_step_hz(cfg["scan"]) == 12e6
