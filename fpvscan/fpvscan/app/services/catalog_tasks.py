"""Curated test-panel knobs: scan width (SWEEP) and jump/tear tools (LOCK).

Only keys the engine actually reads on the live path. PUT still accepts
the full catalog; GET ?mode= filters to these subsets.
"""
from __future__ import annotations

from fpvscan.scan_view import affect_of

# «Смуга сканування частот»: start/stop are the span; channel_bw sets step.
SWEEP_TASKS: dict[str, str] = {
    "scan.start_hz": "scan_width",
    "scan.stop_hz": "scan_width",
    "scan.channel_bw_hz": "scan_width",
    "scan.cluster_step_mhz": "scan_grid",
    "scan.hit_filter": "hit_filter",
}

# Picture jumps = AFC/hunt wander, clip, retune settle, short capture.
# video.sample_rate is not a jump tool: LOCK must stay dec=1.
# H tears = short tracked field (207-line crop). V tears = same-field blend.
# crop_left_frac / crop_bottom_lines are always-on defaults, not LOCK-wall knobs.
# TBC window / smooth are engine constants — not operator sliders.
LOCK_TASKS: dict[str, str] = {
    "video.afc": "picture_jump",
    "video.afc_gain": "picture_jump",
    "video.afc_deadband_hz": "picture_jump",
    "video.afc_max_step_hz": "picture_jump",
    "video.afc_digital_max_hz": "picture_jump",
    "video.hunt": "picture_jump",
    "video.hunt_every": "picture_jump",
    "video.hunt_drop": "picture_jump",
    "sdr.gain_db": "picture_jump",
    "sdr.settle_us": "picture_jump",
    "video.capture_ms": "picture_jump",
    "video.track_window_margin": "phase_tear_h",
    "video.h_phase_frac": "phase_tear_h",
    "video.average": "phase_tear_v",
    "video.motion_thresh": "phase_tear_v",
}

CURATED_BY_MODE: dict[str, frozenset[str]] = {
    "sweep": frozenset(SWEEP_TASKS),
    "lock": frozenset(LOCK_TASKS),
}


def catalog_task(key: str) -> str | None:
    return SWEEP_TASKS.get(key) or LOCK_TASKS.get(key)


def catalog_modes(key: str, applies_to: list[str]) -> list[str]:
    modes: list[str] = []
    if key in SWEEP_TASKS:
        modes.append("sweep")
    if key in LOCK_TASKS:
        modes.append("lock")
    return modes or list(applies_to)


def catalog_affects(key: str) -> str:
    return affect_of(key)


def curated_keys(mode: str) -> frozenset[str]:
    return CURATED_BY_MODE[mode]
