"""Frequency-agnostic hardware sample-rate planning.

The BladeRF source rate is a transport property, not a SWEEP/LOCK DSP knob.
One rate is selected at service start and both modes consume that source.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any


USABLE_SPAN_FRACTION = 0.90
RATE_QUANTUM_HZ = 100_000.0
SC16_BYTES_PER_COMPLEX_SAMPLE = 4
# Overlapping coarse dwells. Never persist the old Nyquist auto-tile (~23.8 MHz).
DEFAULT_SWEEP_STEP_HZ = 12e6


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0.0 else None


@dataclass(frozen=True)
class HardwareRatePlan:
    hardware_sample_rate_hz: float
    requested_scan_rate_hz: float | None
    requested_video_rate_hz: float | None
    requested_sweep_step_hz: float
    requested_channel_cap_hz: float
    effective_sweep_step_hz: float
    usable_span_hz: float
    usable_span_fraction: float
    estimated_sc16_mb_s: float
    reason: str
    explicit_override: bool

    def diagnostics(self) -> dict[str, Any]:
        return asdict(self)


def requested_sweep_step_hz(
    scan: dict[str, Any] | None,
    *,
    usable_span_hz: float | None = None,
) -> float:
    """Operator step, or 12 MHz when missing / wider than the usable tile.

    A stale auto-tile (``step_hz`` > usable span) used to clamp to the
    whole tile and skip overlap.  Zero / absent step must not fall back
    to ``channel_bw`` or a Nyquist formula.
    """
    step = _positive((scan or {}).get("step_hz"))
    usable = _positive(usable_span_hz)
    if step is None or (usable is not None and step > usable):
        return DEFAULT_SWEEP_STEP_HZ
    return step


def sc16_throughput_mb_s(sample_rate_hz: float) -> float:
    """USB payload estimate for one SC16 Q11 complex RX stream."""
    return float(sample_rate_hz) * SC16_BYTES_PER_COMPLEX_SAMPLE / 1_000_000.0


def derive_hardware_rate(
    cfg: dict[str, Any],
    *,
    usable_fraction: float = USABLE_SPAN_FRACTION,
    rate_quantum_hz: float = RATE_QUANTUM_HZ,
) -> HardwareRatePlan:
    """Select one source rate without using RF center or band tables.

    The analogue FM skirt is the transport width: ``max(channel_cap,
    inspect_bw)``.  Sweep ``step_hz`` only sets dwell spacing; if the step
    exceeds the usable tile it is clamped for overlap.  The rate is rounded
    upward (never down) to a hardware-friendly 100 kHz quantum.

    ``sdr.sample_rate`` is the sole explicit source-rate override.  Legacy
    ``scan.sample_rate`` and ``video.sample_rate`` are recorded as requests,
    but do not cause mode-specific hardware mutations.
    """
    scan = cfg.get("scan") or {}
    video = cfg.get("video") or {}
    sdr = cfg.get("sdr") or {}
    fraction = min(0.98, max(0.50, float(usable_fraction)))
    quantum = max(1.0, float(rate_quantum_hz))

    requested_scan = _positive(scan.get("sample_rate"))
    requested_video = _positive(video.get("sample_rate"))
    channel_cap = (
        _positive(video.get("channel_bw_hz"))
        or _positive(scan.get("channel_bw_hz"))
        or 1_000_000.0
    )
    override = _positive(sdr.get("sample_rate"))
    inspect_bw = _positive(scan.get("inspect_bw_hz"))
    required_span = channel_cap
    if inspect_bw is not None:
        required_span = max(required_span, inspect_bw)

    if override is not None:
        hardware = override
        reason = (
            "explicit sdr.sample_rate override; SWEEP and LOCK share it"
        )
        explicit = True
    else:
        raw = required_span / fraction
        hardware = math.ceil(raw / quantum) * quantum
        reason = (
            "derived from max(analogue channel cap, inspect bandwidth) "
            f"/ {fraction:.2f} usable-span fraction"
        )
        explicit = False

    usable = hardware * fraction
    requested_step = requested_sweep_step_hz(scan, usable_span_hz=usable)
    effective_step = min(requested_step, usable)
    if effective_step < requested_step:
        reason += "; sweep step clamped to usable span for gap-free coverage"
    return HardwareRatePlan(
        hardware_sample_rate_hz=float(hardware),
        requested_scan_rate_hz=requested_scan,
        requested_video_rate_hz=requested_video,
        requested_sweep_step_hz=float(requested_step),
        requested_channel_cap_hz=float(channel_cap),
        effective_sweep_step_hz=float(effective_step),
        usable_span_hz=float(usable),
        usable_span_fraction=float(fraction),
        estimated_sc16_mb_s=sc16_throughput_mb_s(hardware),
        reason=reason,
        explicit_override=explicit,
    )


def with_actual_rate(
    plan: HardwareRatePlan,
    actual_rate_hz: float,
    *,
    reason_suffix: str = "hardware-reported actual rate",
) -> HardwareRatePlan:
    """Update diagnostics after the driver reports its quantized actual rate."""
    actual = float(actual_rate_hz)
    usable = actual * plan.usable_span_fraction
    effective_step = min(plan.requested_sweep_step_hz, usable)
    reason = plan.reason
    if abs(actual - plan.hardware_sample_rate_hz) > 1.0:
        reason = f"{reason}; {reason_suffix}"
    return replace(
        plan,
        hardware_sample_rate_hz=actual,
        usable_span_hz=usable,
        effective_sweep_step_hz=effective_step,
        estimated_sc16_mb_s=sc16_throughput_mb_s(actual),
        reason=reason,
    )


def with_runtime_requests(
    plan: HardwareRatePlan,
    cfg: dict[str, Any],
    values: dict[str, Any] | None = None,
) -> HardwareRatePlan:
    """Refresh request diagnostics without changing the session source rate."""
    values = values or {}
    scan = cfg.get("scan") or {}
    video = cfg.get("video") or {}
    requested_scan = (
        _positive(values.get("scan.sample_rate"))
        if "scan.sample_rate" in values
        else plan.requested_scan_rate_hz
    )
    requested_video = (
        _positive(values.get("video.sample_rate"))
        if "video.sample_rate" in values
        else plan.requested_video_rate_hz
    )
    channel_cap = (
        _positive(video.get("channel_bw_hz"))
        or _positive(scan.get("channel_bw_hz"))
        or plan.requested_channel_cap_hz
    )
    usable = plan.hardware_sample_rate_hz * plan.usable_span_fraction
    requested_step = requested_sweep_step_hz(scan, usable_span_hz=usable)
    effective = min(requested_step, usable)
    reason = plan.reason
    clamp_note = "; sweep step clamped to usable span for gap-free coverage"
    reason = reason.replace(clamp_note, "")
    if effective < requested_step:
        reason += clamp_note
    return replace(
        plan,
        requested_scan_rate_hz=requested_scan,
        requested_video_rate_hz=requested_video,
        requested_sweep_step_hz=float(requested_step),
        requested_channel_cap_hz=float(channel_cap),
        effective_sweep_step_hz=float(effective),
        usable_span_hz=float(usable),
        estimated_sc16_mb_s=sc16_throughput_mb_s(
            plan.hardware_sample_rate_hz
        ),
        reason=reason,
    )


def normalize_runtime_config(
    cfg: dict[str, Any],
    plan: HardwareRatePlan,
) -> None:
    """Make legacy mode rates truthfully reflect the shared source rate."""
    for section in ("scan", "video"):
        block = cfg.get(section)
        if isinstance(block, dict):
            block["sample_rate"] = plan.hardware_sample_rate_hz
