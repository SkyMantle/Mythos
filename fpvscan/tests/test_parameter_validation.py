from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from fpvscan.app.domain.exceptions import ValidationError
from fpvscan.app.services.catalog import build_catalog, coerce_param
from fpvscan.app.services.catalog_tasks import LOCK_TASKS, SWEEP_TASKS
from fpvscan.app.services.parameter_service import ParameterService


def test_catalog_from_real_config_has_sweep_and_lock() -> None:
    root = Path(__file__).resolve().parents[1]
    from fpvscan import config
    cfg = config.load(root / "config.yaml")
    yaml_text = (root / "config.yaml").read_text(encoding="utf-8")
    specs = build_catalog(cfg, cfg, yaml_text)
    groups = {s.group for s in specs}
    keys = {s.key for s in specs}
    assert "sweep" in groups and "lock" in groups and "shared" in groups
    assert "scan.threshold_db" in keys
    assert "video.afc" in keys
    assert "sdr.gain_db" in keys
    assert "sdr.auto_gain" in keys
    assert "sdr.driver" not in keys
    thresh = next(s for s in specs if s.key == "scan.threshold_db")
    assert thresh.type == "float"
    assert thresh.unit == "dB"
    assert thresh.min is not None and thresh.max is not None
    assert "підлог" in thresh.description
    assert thresh.task is None
    assert thresh.affects == "spectrum"
    pin = next(s for s in specs if s.key == "video.spectrum_pin_center")
    assert pin.type == "bool" and pin.default is True and pin.affects == "spectrum"
    assert pin.task is None
    ema = next(s for s in specs if s.key == "video.spectrum_ema")
    assert ema.type == "bool" and ema.default is True and ema.affects == "spectrum"
    sm3 = next(s for s in specs if s.key == "video.spectrum_smooth3")
    assert sm3.type == "bool" and sm3.default is True and sm3.affects == "spectrum"
    every4 = next(s for s in specs if s.key == "video.spectrum_every_4")
    assert every4.type == "bool" and every4.default is False and every4.affects == "spectrum"
    start = next(s for s in specs if s.key == "scan.start_hz")
    assert start.task == "scan_width"
    assert start.modes == ["sweep"]
    assert start.affects == "grid"
    grid = next(s for s in specs if s.key == "scan.cluster_step_mhz")
    assert grid.task == "scan_grid"
    assert grid.type == "enum"
    assert grid.affects == "grid"
    assert grid.default == "8"
    filt = next(s for s in specs if s.key == "scan.hit_filter")
    assert filt.task == "hit_filter"
    assert filt.affects == "detections"
    assert filt.default == "hide_weak"
    afc = next(s for s in specs if s.key == "video.afc")
    assert afc.task == "picture_jump"
    assert afc.modes == ["lock"]
    assert afc.affects == "picture"
    margin = next(s for s in specs if s.key == "video.track_window_margin")
    assert margin.task == "phase_tear_h"
    avg = next(s for s in specs if s.key == "video.average")
    assert avg.task == "phase_tear_v"
    assert SWEEP_TASKS.keys() <= keys
    assert LOCK_TASKS.keys() <= keys
    assert "video.sharpen" in keys
    assert LOCK_TASKS["video.h_phase_frac"] == "phase_tear_h"
    assert LOCK_TASKS["video.h_pll"] == "pll"
    assert LOCK_TASKS["sdr.bias_tee"] == "picture_jump"
    assert "video.sample_rate" not in LOCK_TASKS
    assert "video.crop_left_frac" not in LOCK_TASKS
    assert "video.crop_bottom_lines" not in LOCK_TASKS
    assert "video.tbc_search_frac" not in LOCK_TASKS
    hp = next(s for s in specs if s.key == "video.h_phase_frac")
    assert hp.task == "phase_tear_h"
    assert hp.min == -0.5 and hp.max == 0.5
    pll = next(s for s in specs if s.key == "video.h_pll")
    assert pll.type == "bool"
    assert pll.default is False
    assert pll.task == "pll"
    assert pll.affects == "picture"
    gain = next(s for s in specs if s.key == "sdr.gain_db")
    assert gain.task == "picture_jump"
    auto_g = next(s for s in specs if s.key == "sdr.auto_gain")
    assert auto_g.type == "bool"
    assert auto_g.default is True
    assert auto_g.task is None
    bias = next(s for s in specs if s.key == "sdr.bias_tee")
    assert bias.type == "bool"
    assert "gain" in bias.description.lower() or "Bias" in bias.description
    assert "Bias-T" in gain.description or "bias" in gain.description.lower()
    dead = next(s for s in specs if s.key == "video.afc_deadband_hz")
    assert dead.max == 500_000
    dmax = next(s for s in specs if s.key == "video.afc_digital_max_hz")
    assert dmax.max == 3_000_000
    step = next(s for s in specs if s.key == "video.afc_max_step_hz")
    assert step.max == 500_000
    with pytest.raises(ValidationError, match="max"):
        coerce_param(dmax, 50e6)


def test_list_parameters_mode_returns_curated_tasks(engine, store) -> None:
    svc = ParameterService(engine, store)

    async def run() -> None:
        full = await svc.list_parameters()
        sweep = await svc.list_parameters("sweep")
        lock = await svc.list_parameters("lock")
        assert {s.key for s in sweep} == set(SWEEP_TASKS)
        by_sweep = {s.key: s.task for s in sweep}
        assert by_sweep["scan.start_hz"] == "scan_width"
        assert by_sweep["scan.cluster_step_mhz"] == "scan_grid"
        assert by_sweep["scan.hit_filter"] == "hit_filter"
        step = next(s for s in sweep if s.key == "scan.cluster_step_mhz")
        assert step.type == "enum"
        assert step.enum_values == ["off", "12", "8", "4"]
        assert step.enum_labels["8"] == "8 МГц"
        filt = next(s for s in sweep if s.key == "scan.hit_filter")
        assert filt.enum_values == ["all", "hide_weak", "hide_no_video", "hide_near_dup"]
        assert filt.enum_labels["hide_weak"] == "ховати слабкі"
        assert {s.key for s in lock} == set(LOCK_TASKS)
        assert "sdr.auto_gain" not in {s.key for s in lock}
        by_task = {s.key: s.task for s in lock}
        assert by_task["video.afc"] == "picture_jump"
        assert by_task["video.track_window_margin"] == "phase_tear_h"
        assert by_task["video.h_phase_frac"] == "phase_tear_h"
        assert by_task["video.average"] == "phase_tear_v"
        assert by_task["video.h_pll"] == "pll"
        assert by_task["sdr.bias_tee"] == "picture_jump"
        pll = next(s for s in lock if s.key == "video.h_pll")
        assert pll.type == "bool"
        assert pll.default is False
        assert {s.key for s in full} > {s.key for s in sweep} | {s.key for s in lock}
        with pytest.raises(ValidationError, match="sweep or lock"):
            await svc.list_parameters("waterfall")

    import asyncio
    asyncio.run(run())


def test_coerce_rejects_out_of_range(engine) -> None:
    spec = next(s for s in build_catalog(engine.cfg, engine.cfg, "") if s.key == "scan.threshold_db")
    with pytest.raises(ValidationError, match="max"):
        coerce_param(spec, 999.0)
    with pytest.raises(ValidationError, match="min"):
        coerce_param(spec, -100.0)


def test_coerce_int_rejects_fraction(engine) -> None:
    spec = next(s for s in build_catalog(engine.cfg, engine.cfg, "") if s.key == "scan.confirm_hits")
    with pytest.raises(ValidationError, match="expected int"):
        coerce_param(spec, 2.5)
    assert coerce_param(spec, 3) == 3


def test_apply_unknown_key_rejected(engine, store) -> None:
    svc = ParameterService(engine, store)

    async def run() -> None:
        with pytest.raises(ValidationError, match="unknown"):
            await svc.apply_parameters(uuid4(), {"scan.not_a_real_param": 1})

    import asyncio
    asyncio.run(run())


def test_apply_cluster_step_is_pending_plan(engine, store) -> None:
    svc = ParameterService(engine, store)

    async def run() -> None:
        result = await svc.apply_parameters(uuid4(), {"scan.cluster_step_mhz": "4"})
        assert result.values["scan.cluster_step_mhz"] == "4"
        assert result.affects["scan.cluster_step_mhz"] == "grid"
        assert "scan.cluster_step_mhz" in result.pending_keys
        assert result.pending_reasons["scan.cluster_step_mhz"] == "next sweep plan"
        filt = await svc.apply_parameters(uuid4(), {"scan.hit_filter": "hide_near_dup"})
        assert filt.values["scan.hit_filter"] == "hide_near_dup"
        assert filt.affects["scan.hit_filter"] == "detections"
        assert "scan.hit_filter" not in filt.pending_keys

    import asyncio
    asyncio.run(run())


def test_apply_subset_updates_current(engine, store) -> None:
    svc = ParameterService(engine, store)

    async def run() -> None:
        result = await svc.apply_parameters(uuid4(), {"scan.threshold_db": 7.5})
        assert result.values["scan.threshold_db"] == 7.5
        assert "scan.threshold_db" in result.applied_keys
        assert result.values["video.afc"] is True
        assert result.affects["scan.threshold_db"] == "spectrum"
        assert ("refresh_lock", {}) not in engine.commands

    import asyncio
    asyncio.run(run())


def test_lock_apply_refreshes_picture_knobs(engine, store) -> None:
    svc = ParameterService(engine, store)

    async def run() -> None:
        result = await svc.apply_parameters(uuid4(), {
            "video.sample_rate": 25e6,
            "video.sharpen": 0.8,
            "scan.start_hz": 500e6,
        })
        assert "video.sample_rate" in result.applied_keys
        assert "video.sample_rate" not in result.pending_keys
        assert "scan.start_hz" in result.pending_keys
        assert result.pending_reasons["scan.start_hz"] == "next sweep plan"
        assert result.affects["video.sharpen"] == "picture"
        assert result.affects["scan.start_hz"] == "grid"
        assert ("refresh_lock", {}) in engine.commands
        engine.commands.clear()
        engine._snap["mode"] = "SWEEP"
        swept = await svc.apply_parameters(uuid4(), {"video.sample_rate": 20e6})
        assert "video.sample_rate" in swept.pending_keys
        assert swept.pending_reasons["video.sample_rate"] == "next lock retune"
        assert ("refresh_lock", {}) not in engine.commands

    import asyncio
    asyncio.run(run())


def test_apply_h_pll_live_on_lock(engine, store) -> None:
    svc = ParameterService(engine, store)

    async def run() -> None:
        result = await svc.apply_parameters(uuid4(), {"video.h_pll": True})
        assert result.values["video.h_pll"] is True
        assert result.affects["video.h_pll"] == "picture"
        assert "video.h_pll" not in result.pending_keys
        assert ("refresh_lock", {}) not in engine.commands

    import asyncio
    asyncio.run(run())


def test_apply_spectrum_every_4_live_on_lock(engine, store) -> None:
    svc = ParameterService(engine, store)

    async def run() -> None:
        result = await svc.apply_parameters(uuid4(), {"video.spectrum_every_4": True})
        assert result.values["video.spectrum_every_4"] is True
        assert result.affects["video.spectrum_every_4"] == "spectrum"
        assert "video.spectrum_every_4" not in result.pending_keys
        assert ("refresh_lock", {}) not in engine.commands

    import asyncio
    asyncio.run(run())
