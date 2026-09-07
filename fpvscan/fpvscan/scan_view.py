"""Pure views of the current FFT snapshot and sweep grid.

Used by the engine snapshot and the test-panel live API so spectrum
brackets and grid progress always read the live config, not stale
defaults baked into the last IQ capture.
"""
from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np


def mono_ms() -> int:
    return int(time.monotonic() * 1000)


def lock_spectrum_due(lock_n: int, every: int) -> bool:
    """First lock IQ and every Nth after. every=1 → every IQ (n % 1 is always 0)."""
    n = max(1, int(every))
    return lock_n >= 1 and (lock_n - 1) % n == 0


def afc_is_pegged(afc_hz: float, digital_max_hz: float,
                 freq_err_hz: float | None = None) -> bool:
    """True when digital AFC sits on its stop (|afc| ≥ 0.95 × cap).

    Independent of freq_err. The unused freq_err_hz kw keeps older
    call sites valid.
    """
    cap = abs(float(digital_max_hz))
    if cap < 1.0:
        return False
    return abs(float(afc_hz)) >= 0.95 * cap


def afc_should_nudge(afc_hz: float, freq_err_hz: float, digital_max_hz: float,
                     min_err_hz: float = 250e3) -> bool:
    """Wide hunt / lock step only if pegged and residual error is still large."""
    if not afc_is_pegged(afc_hz, digital_max_hz):
        return False
    return abs(float(freq_err_hz)) >= float(min_err_hz)


def lock_channel_bw(fs: float, want_bw: float, *,
                    off_hz: float = 0.0,
                    headroom_hz: float = 1.5e6,
                    min_bw: float = 12e6) -> float:
    """LOCK channel width with dec=1. dec=2 breaks PAL (30 Msps + 12 MHz IF)."""
    fs = float(fs)
    if fs < 1.0:
        return float(want_bw)
    ch = min(float(want_bw), fs * 0.9)
    ch = max(ch, min(float(min_bw), fs * 0.9))
    max_bw = 2.0 * max(6e6, fs * 0.45 - abs(float(off_hz)) - float(headroom_hz))
    ch = min(ch, max_bw)
    ch = max(ch, min(float(min_bw), fs * 0.9))
    # Forbid dec>=2: raise BW to match fs so channelize stays dec=1.
    if int(fs / max(ch, 1.0)) >= 2:
        ch = fs
    return ch


def lock_decimation(fs: float, ch_bw: float) -> int:
    """LOCK decimation. dec=2 is forbidden — never return 2+."""
    dec = max(1, int(float(fs) / max(float(ch_bw), 1.0)))
    return 1 if dec >= 2 else dec


def hunt_span_hz(offsets_mhz: list | None, *, pegged: bool, wide_hz: float = 2e6) -> float:
    offs = [abs(float(m)) * 1e6 for m in (offsets_mhz or [0.25])]
    span = max(offs) if offs else 0.25e6
    if pegged:
        return max(span, float(wide_hz))
    return span


PENDING_REASONS: dict[str, str] = {
    "scan.sample_rate": "next sweep retune",
    "video.sample_rate": "next lock retune",
    "video.lo_offset_hz": "next lock retune",
    "sdr.settle_us": "next retune",
    "scan.fft_size": "next sweep FFT",
    "scan.averages": "next sweep FFT",
    "scan.channel_bw_hz": "next sweep step/FFT",
    "scan.step_hz": "next sweep plan",
    "scan.start_hz": "next sweep plan",
    "scan.stop_hz": "next sweep plan",
    "scan.cluster_step_mhz": "next sweep plan",
    "video.channel_bw_hz": "next lock channelize",
    "video.capture_ms": "next lock capture",
    "video.hunt": "next hunt cycle",
    "video.hunt_offsets_mhz": "next hunt cycle",
    "video.hunt_every": "next hunt cycle",
    "video.deviation_hz": "next lock demod",
}

# Keys that a LOCK refresh (restart reader / decode, keep rec) makes live.
LOCK_REFRESH_KEYS = frozenset(k for k in PENDING_REASONS if not k.startswith("scan."))


def affect_of(key: str) -> str:
    """Primary surface a knob changes: picture | spectrum | grid | next_sweep."""
    if key in {
        "scan.start_hz", "scan.stop_hz", "scan.step_hz", "scan.priority_bands",
        "scan.cluster_step_mhz",
    }:
        return "grid"
    if key == "scan.hit_filter":
        return "detections"
    if key in {
        "scan.fft_size", "scan.averages", "scan.threshold_db",
        "scan.dc_notch_hz", "video.spectrum_every",
    }:
        return "spectrum"
    if key.startswith("scan."):
        return "next_sweep"
    return "picture"


def pending_for(
    keys: Iterable[str],
    *,
    lock_refreshed: bool = False,
) -> tuple[list[str], dict[str, str]]:
    pending = [k for k in keys if k in PENDING_REASONS]
    if lock_refreshed:
        pending = [k for k in pending if k not in LOCK_REFRESH_KEYS]
    return pending, {k: PENDING_REASONS[k] for k in pending}


def needs_lock_refresh(keys: Iterable[str]) -> bool:
    return any(affect_of(k) == "picture" for k in keys)


# Coarse Nyquist tile (~26 MHz at 35 Msps). Cluster extras are
# injected only around a coarse hit. Default spacing is 8 MHz.
SWEEP_CLUSTER_STEP_DEFAULT_MHZ = 8.0
SWEEP_HIT_TOL_HZ = 8.0e6
SWEEP_HIT_SEE_HZ = 20.0e6
SWEEP_DENSE_RADIUS_HZ = 20.0e6
SWEEP_DENSE_ALIGN_HZ = 2.0e6


def cluster_step_hz(scan: dict[str, Any]) -> float | None:
    """Operator cluster extra spacing. None = off (coarse + inspect only)."""
    raw = scan.get("cluster_step_mhz", SWEEP_CLUSTER_STEP_DEFAULT_MHZ)
    if raw in (None, "", False, "off", "Off", "OFF"):
        return None
    try:
        mhz = float(raw)
    except (TypeError, ValueError):
        return None
    if mhz <= 0:
        return None
    return mhz * 1e6


def sweep_step_hz(scan: dict[str, Any], *, cluster: bool = False) -> float:
    fs = float(scan.get("sample_rate") or 0)
    ch_bw = float(scan.get("channel_bw_hz") or 20e6)
    step = float(scan.get("step_hz") or 0)
    if step > 0:
        return step
    if fs <= 0:
        return 0.0
    tile = max(fs * 0.25, fs * 0.9 - ch_bw / 2)
    if not cluster:
        return tile
    wanted = cluster_step_hz(scan)
    if wanted is None:
        return tile
    return min(tile, wanted)


def _unique_hz(points: Iterable[float], tol_hz: float = 1.0e6) -> list[float]:
    ordered = sorted(float(p) for p in points)
    out: list[float] = []
    for hz in ordered:
        if not out or abs(hz - out[-1]) >= tol_hz:
            out.append(hz)
    return out


def sweep_cluster_bands(priority_bands: Iterable[Any] | None) -> list[Any]:
    """Dense ~4 MHz dwells: 433 / 900 / 1.2 / 3.3 / 5.8. Not the whole 400–2 GHz."""
    from fpvscan.bands import PRIORITY_BANDS
    always = {b.name for b in PRIORITY_BANDS if b.name in {
        "433", "900", "1G2", "3G3", "5G8",
    }}
    extra = list(priority_bands or [])
    names = {getattr(b, "name", "") for b in extra}
    out = list(extra)
    for b in PRIORITY_BANDS:
        if b.name in always and b.name not in names:
            out.append(b)
    return out


def cluster_containing(
    freq_hz: float,
    priority_bands: Iterable[Any] | None = None,
) -> Any | None:
    """433 / 900 / 1.2 / 3.3 / 5.8 band that holds freq, or None."""
    hz = float(freq_hz)
    for band in sweep_cluster_bands(priority_bands):
        if float(band.start_hz) <= hz <= float(band.stop_hz):
            return band
    return None


def densify_around(
    peak_hz: float,
    scan: dict[str, Any],
    *,
    radius_hz: float | None = None,
    band: Any | None = None,
) -> list[float]:
    """Cluster extras aligned on the peak, clipped to the cluster / scan."""
    step = sweep_step_hz(scan, cluster=True)
    if step <= 0:
        return []
    start = float(scan.get("start_hz") or 0)
    stop = float(scan.get("stop_hz") or 0)
    peak = float(peak_hz)
    rad = float(radius_hz if radius_hz is not None else SWEEP_DENSE_RADIUS_HZ)
    lo = peak - rad
    hi = peak + rad
    if band is not None:
        lo = max(lo, float(band.start_hz))
        hi = min(hi, float(band.stop_hz))
    if start:
        lo = max(lo, start)
    if stop:
        hi = min(hi, stop)
    if hi <= lo:
        return []
    k0 = int(np.ceil((lo - peak) / step))
    k1 = int(np.floor((hi - peak) / step))
    pts = [peak + k * step for k in range(k0, k1 + 1)]
    if pts and min(abs(p - peak) for p in pts) > SWEEP_DENSE_ALIGN_HZ:
        pts.append(peak)
    return _unique_hz(pts)


def extras_for_hit(
    dwell_hz: float,
    peak_hz: float,
    scan: dict[str, Any],
    *,
    priority_bands: Iterable[Any] | None = None,
) -> list[float]:
    """Cluster extras when a coarse dwell sees a nearby blob. off → []."""
    if cluster_step_hz(scan) is None:
        return []
    see = float(scan.get("cluster_see_hz") or SWEEP_HIT_SEE_HZ)
    if abs(float(peak_hz) - float(dwell_hz)) > see:
        return []
    band = cluster_containing(peak_hz, priority_bands)
    if band is None:
        return []
    extras = densify_around(peak_hz, scan, band=band)
    return [hz for hz in extras if abs(hz - float(dwell_hz)) >= 1.0e6]


def sweep_centers(
    scan: dict[str, Any],
    priority_bands: Iterable[Any] | None = None,
    fixed_freq: float | None = None,
    pass_index: int = 0,
) -> list[float]:
    """Coarse 400 MHz–6 GHz tile. Cluster 4 MHz extras are on-hit, not here.

    `pass_index` is kept for callers; it no longer sprinkles a carpet.
    """
    del pass_index, priority_bands
    if fixed_freq is not None:
        return [float(fixed_freq)]
    fs = float(scan.get("sample_rate") or 0)
    start = float(scan.get("start_hz") or 0)
    stop = float(scan.get("stop_hz") or 0)
    step = sweep_step_hz(scan)
    if fs <= 0 or step <= 0 or stop <= start:
        return []
    pts = [float(p) for p in np.arange(start + fs / 2, stop, step)]
    return _unique_hz(pts)


def coarse_sweep_len(scan: dict[str, Any]) -> int:
    fs = float(scan.get("sample_rate") or 0)
    start = float(scan.get("start_hz") or 0)
    stop = float(scan.get("stop_hz") or 0)
    step = sweep_step_hz(scan)
    if fs <= 0 or step <= 0 or stop <= start:
        return 0
    return int(np.arange(start + fs / 2, stop, step).size)


def nearest_sweep_hz(centers: Iterable[float], freq_hz: float) -> float | None:
    best = None
    best_d = 1e18
    for hz in centers:
        d = abs(float(hz) - float(freq_hz))
        if d < best_d:
            best, best_d = float(hz), d
    return best


def spectrum_snapshot(
    cfg: dict[str, Any],
    *,
    mode: str,
    lock_target: float | None,
    tuned_hz: float,
    last: dict[str, Any] | None,
    afc_hz: float = 0.0,
    next_hz: float | None = None,
    dwell_ms: float | None = None,
    t_mono_ms: int | None = None,
) -> dict[str, Any]:
    scan = cfg.get("scan") or {}
    video = cfg.get("video") or {}
    locked = str(mode).upper() == "LOCK" and lock_target
    last = last or {}
    bins = last.get("bins")
    floor = last.get("floor_db")
    peak = last.get("peak_db")
    if peak is None and bins:
        peak = round(float(max(bins)), 1)
    if locked:
        off = float(video.get("lo_offset_hz") or 0)
        span = float(video.get("sample_rate") or 0) or None
        bw = float(video.get("channel_bw_hz") or 0) or None
        cursor = float(lock_target) + float(afc_hz or 0)
        center = float(lock_target) + off
        nfft = int(last.get("nfft") or 2048)
    else:
        span = float(scan.get("sample_rate") or 0) or None
        bw = float(scan.get("channel_bw_hz") or 0) or None
        center = float(last.get("center_hz") or tuned_hz or 0) or None
        cursor = center
        nfft = int(last.get("nfft") or scan.get("fft_size") or 4096)
    rate = last.get("rate_hz")
    stamp = int(t_mono_ms if t_mono_ms is not None else last.get("t_mono_ms") or mono_ms())
    return {
        "bins": [float(v) for v in bins] if bins is not None else None,
        "center_hz": None if center is None else float(center),
        "bw_hz": None if bw is None else float(bw),
        "span_hz": None if span is None else float(span),
        "cursor_hz": None if cursor is None else float(cursor),
        "floor_db": None if floor is None else float(floor),
        "peak_db": None if peak is None else float(peak),
        "nfft": nfft,
        "rate_hz": None if rate is None else float(rate),
        "t_mono_ms": stamp,
        "next_hz": None if next_hz is None else float(next_hz),
        "dwell_ms": None if dwell_ms is None else float(dwell_ms),
        "afc_pegged": bool(last.get("afc_pegged", False)),
        "freq_err_hz": last.get("freq_err_hz"),
        "afc_hz": last.get("afc_hz"),
        "hunt_span_hz": last.get("hunt_span_hz"),
    }


def grid_snapshot(
    cfg: dict[str, Any],
    *,
    mode: str,
    sweep_i: int,
    sweeps_done: int,
    sweep_pos_hz: float,
    tuned_hz: float,
    lock_target: float | None,
    visiting_hz: list[float] | None = None,
    priority_bands: Iterable[Any] | None = None,
    plan_len: int | None = None,
    afc_hz: float = 0.0,
    next_hz: float | None = None,
    dwell_ms: float | None = None,
    t_mono_ms: int | None = None,
) -> dict[str, Any]:
    scan = cfg.get("scan") or {}
    start = float(scan.get("start_hz") or 0)
    stop = float(scan.get("stop_hz") or 0)
    step = sweep_step_hz(scan)
    if plan_len is None:
        plan_len = len(sweep_centers(scan, priority_bands=priority_bands))
    idx = max(0, min(int(sweep_i), int(plan_len)))
    progress = (idx / plan_len) if plan_len else 0.0
    locked = str(mode).upper() == "LOCK" and lock_target
    current = (
        float(lock_target) + float(afc_hz or 0)
        if locked else float(sweep_pos_hz or tuned_hz or 0)
    )
    visits = [float(v) for v in (visiting_hz or []) if v]
    if not visits and current:
        visits = [current]
    return {
        "pass_index": idx,
        "pass_count": int(plan_len),
        "current_hz": current or None,
        "start_hz": start or None,
        "stop_hz": stop or None,
        "step_hz": step or None,
        "progress_01": round(progress, 4),
        "visiting_hz": visits,
        "passes_done": int(sweeps_done),
        "t_mono_ms": int(t_mono_ms if t_mono_ms is not None else mono_ms()),
        "next_hz": None if next_hz is None else float(next_hz),
        "dwell_ms": None if dwell_ms is None else float(dwell_ms),
    }
