"""Pure views of the current FFT snapshot and sweep grid.

Used by the engine snapshot and the test-panel live API so spectrum
brackets and grid progress always read the live config, not stale
defaults baked into the last IQ capture.
"""
from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np

from .hardware_rate import USABLE_SPAN_FRACTION, requested_sweep_step_hz


def mono_ms() -> int:
    return int(time.monotonic() * 1000)


def lock_spectrum_due(lock_n: int, every: int) -> bool:
    """First lock IQ and every Nth after. every=1 → every IQ (n % 1 is always 0)."""
    n = max(1, int(every))
    return lock_n >= 1 and (lock_n - 1) % n == 0


def lock_spectrum_every(video: dict[str, Any] | None) -> int:
    """LOCK FFT cadence from live catalog. Sweep occupancy FFT ignores this."""
    v = video or {}
    if bool(v.get("spectrum_every_4", False)):
        return 4
    try:
        return max(1, int(v.get("spectrum_every", 1) or 1))
    except (TypeError, ValueError):
        return 1


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


# Locked raster with a usable score: do not slam ±2 MHz hunt/nudge.
HUNT_HOLD_SCORE = 0.35
# After a pegged-lock RF snap, wait before moving the LO again.
RF_SNAP_COOLDOWN_S = 1.5


def freeze_afc_hunt(*, pic_locked: bool, pic_score: float = 0.0,
                    min_score: float = HUNT_HOLD_SCORE) -> bool:
    """True when a locked picture is already on screen.

    Digital AFC may still tick (80 kHz deadband). Wide hunt and the
    RF ±2 MHz peg-nudge must not: they tear a good raster more than
    they help frequency.
    """
    return bool(pic_locked) and float(pic_score) >= float(min_score)


def freeze_lock_afc(*, analog_ok: bool = False, pic_locked: bool = False,
                    pic_score: float = 0.0, row_corr: float = 0.0,
                    luma_mean: float | None = None,
                    min_score: float = HUNT_HOLD_SCORE,
                    min_corr: float = 0.18,
                    min_luma: float = 12.0) -> bool:
    """Hold digital AFC only when a visible analog picture is on screen.

    Publishing 282 black lines is not a picture: freq_error then walks
    the LO (3430→3431) while sticky t0 paints sync/porch.
    """
    if luma_mean is not None and float(luma_mean) < float(min_luma):
        return False
    if analog_ok:
        return True
    if bool(pic_locked) and float(row_corr) >= float(min_corr):
        return True
    return freeze_afc_hunt(pic_locked=pic_locked, pic_score=pic_score,
                           min_score=min_score)


def rf_snap_due(*, pic_locked: bool, pic_score: float, afc_hz: float,
                digital_max_hz: float, last_snap_mono: float = 0.0,
                now_mono: float | None = None,
                cooldown_s: float = RF_SNAP_COOLDOWN_S,
                min_score: float = HUNT_HOLD_SCORE) -> bool:
    """Once: real raster + digital AFC at stop. Not the ±2 MHz hunt.

    No picture / weak score → False. Cooldown blocks back-to-back frames.
    """
    if not freeze_afc_hunt(pic_locked=pic_locked, pic_score=pic_score,
                           min_score=min_score):
        return False
    if not afc_is_pegged(afc_hz, digital_max_hz):
        return False
    now = time.monotonic() if now_mono is None else float(now_mono)
    if last_snap_mono and (now - float(last_snap_mono)) < float(cooldown_s):
        return False
    return True


def afc_should_nudge(afc_hz: float, freq_err_hz: float, digital_max_hz: float,
                     min_err_hz: float = 250e3, *,
                     pic_locked: bool = False) -> bool:
    """Wide hunt / lock step only if pegged and residual error is still large.

    `pic_locked` freezes the ±2 MHz hunt/nudge while a picture exists.
    """
    if pic_locked:
        return False
    if not afc_is_pegged(afc_hz, digital_max_hz):
        return False
    return abs(float(freq_err_hz)) >= float(min_err_hz)


def lock_channel_bw(fs: float, want_bw: float, *,
                    off_hz: float = 0.0,
                    headroom_hz: float = 1.5e6,
                    min_bw: float = 8e6) -> float:
    """LOCK channel width. Keep a narrow analog IF (≈9 MHz).

    20 Msps + 9 MHz is dec=2; channelize() has a 3-sample path that
    keeps PAL edges. Do not raise want_bw up to fs — that dropped the
    PAL-preserving dec=2 path and demodulated the whole USB span.
    """
    fs = float(fs)
    want = float(want_bw)
    if fs < 1.0:
        return want
    floor = min(float(min_bw), want) if want > 0 else float(min_bw)
    ch = min(want, fs * 0.9)
    ch = max(ch, min(floor, fs * 0.9))
    max_bw = 2.0 * max(6e6, fs * 0.45 - abs(float(off_hz)) - float(headroom_hz))
    ch = min(ch, max_bw)
    ch = max(ch, min(floor, fs * 0.9))
    return ch


def lock_decimation(fs: float, ch_bw: float) -> int:
    """LOCK decimation, matching demod.channelize (dec=2 is allowed)."""
    return max(1, int(float(fs) / max(float(ch_bw), 1.0)))


def prelock_mix_hz(
    blob_hz: float,
    *,
    tracking: bool,
    fs: float,
    ch_bw: float,
    off_hz: float = 0.0,
    max_hz: float = 2.5e6,
    deadband_hz: float = 80e3,
) -> float:
    """Digital mixer nudge onto the FM energy blob before CVBS lock.

    Video AFC / hunt are gated on a locked raster, so a 1–2 MHz click
    error stayed mixed off-center and decode returned None (black pane).
    """
    if tracking:
        return 0.0
    nyq = max(0.0, float(fs) * 0.45 - abs(float(off_hz)) - float(ch_bw) / 2.0)
    lim = min(float(max_hz), nyq)
    if lim < 50e3:
        return 0.0
    x = float(blob_hz)
    if abs(x) < float(deadband_hz):
        return 0.0
    if x > lim:
        return lim
    if x < -lim:
        return -lim
    return x


def hunt_span_hz(offsets_mhz: list | None, *, pegged: bool, wide_hz: float = 2e6) -> float:
    offs = [abs(float(m)) * 1e6 for m in (offsets_mhz or [0.25])]
    span = max(offs) if offs else 0.25e6
    if pegged:
        return max(span, float(wide_hz))
    return span


PENDING_REASONS: dict[str, str] = {
    "scan.sample_rate": "request recorded; session source rate remains shared",
    "video.sample_rate": "request recorded; session source rate remains shared",
    "video.lo_offset_hz": "next lock retune",
    "sdr.settle_us": "next retune",
    "scan.fft_size": "next sweep FFT",
    "scan.averages": "next sweep FFT",
    "scan.edge_guard": "next sweep FFT",
    "scan.noise_percentile": "next sweep FFT",
    "scan.threshold_mode": "next sweep FFT",
    "scan.threshold_offset_db": "next sweep FFT",
    "scan.channel_bw_hz": "next sweep step/FFT",
    "scan.step_hz": "next sweep plan",
    "scan.start_hz": "next sweep plan",
    "scan.stop_hz": "next sweep plan",
    "scan.cluster_step_mhz": "next sweep plan",
    "scan.cluster_see_hz": "next sweep plan",
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
        "scan.threshold_mode", "scan.threshold_offset_db",
        "scan.threshold_min_db", "scan.threshold_max_db", "scan.threshold_k",
        "scan.noise_percentile",
        "scan.dc_notch_hz", "scan.edge_guard", "video.spectrum_every",
        "video.spectrum_every_4",
        "video.spectrum_pin_center", "video.spectrum_ema",
        "video.spectrum_smooth3",
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


LOCK_HARDWARE_KEYS = frozenset({"video.lo_offset_hz"})


def needs_lock_refresh(keys: Iterable[str]) -> bool:
    """Restart LOCK only for settings that change its RF tuning."""
    return any(k in LOCK_HARDWARE_KEYS for k in keys)


# Coarse Nyquist tile (~26 MHz at 35 Msps). Cluster extras are
# injected only around a coarse hit. Default spacing is 8 MHz.
SWEEP_CLUSTER_STEP_DEFAULT_MHZ = 4.0
SWEEP_HIT_TOL_HZ = 18.0e6
SWEEP_HIT_SEE_HZ = 28.0e6
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
    usable = fs * USABLE_SPAN_FRACTION if fs > 0 else None
    coarse = requested_sweep_step_hz(scan, usable_span_hz=usable)
    # A configured step is a request, not permission to leave holes.  Clamp
    # it to the transition-safe source span instead of raising USB rate.
    if usable is not None:
        coarse = min(coarse, usable)
    if not cluster:
        return coarse
    wanted = cluster_step_hz(scan)
    if wanted is None:
        return coarse
    return wanted


def _unique_hz(points: Iterable[float], tol_hz: float = 1.0e6) -> list[float]:
    ordered = sorted(float(p) for p in points)
    out: list[float] = []
    for hz in ordered:
        if not out or abs(hz - out[-1]) >= tol_hz:
            out.append(hz)
    return out


def sweep_cluster_bands(priority_bands: Iterable[Any] | None) -> list[Any]:
    """Dense extras: 433 / 900 / 1.2 / 2.4 / 3.3 / 5.8. Not a 400–2 GHz carpet."""
    from fpvscan.bands import PRIORITY_BANDS
    always = {b.name for b in PRIORITY_BANDS if b.name in {
        "433", "900", "1G2", "2G4", "3G3", "5G8",
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
    """433 / 900 / 1.2 / 2.4 / 3.3 / 5.8 band that holds freq, or None."""
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
    """Frequency-agnostic dense extras around any nearby occupied region."""
    if cluster_step_hz(scan) is None:
        return []
    see = float(scan.get("cluster_see_hz") or SWEEP_HIT_SEE_HZ)
    if abs(float(peak_hz) - float(dwell_hz)) > see:
        return []
    # Operational lock frequencies change every session.  Named catalogue
    # bands remain useful UI metadata, but must never gate RF follow-up.
    extras = densify_around(peak_hz, scan, band=None)
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
    usable = fs * USABLE_SPAN_FRACTION
    first = start + usable / 2.0
    last = stop - usable / 2.0
    if last < first:
        return [(start + stop) / 2.0]
    pts = [float(p) for p in np.arange(first, last + 1.0, step)]
    # Cover the upper endpoint even when arange's final tile falls short.
    if not pts or pts[-1] + usable / 2.0 < stop - 1.0:
        pts.append(last)
    out = _unique_hz(pts)
    if out and out[-1] + usable / 2.0 < stop - 1.0:
        if last - out[-1] < 1.0e6:
            out[-1] = last
        else:
            out.append(last)
    return out


def coarse_sweep_len(scan: dict[str, Any]) -> int:
    return len(sweep_centers(scan))


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
        hit = float(lock_target)
        pin = True if "spectrum_pin_center" not in video else bool(
            video.get("spectrum_pin_center"))
        # Pin: axis + yellow marker on published hit. Digital AFC still
        # mixes IQ; bins may sit off-center. Off: cursor tracks _afc.
        cursor = hit if pin else hit + float(afc_hz or 0)
        center = hit + off
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
