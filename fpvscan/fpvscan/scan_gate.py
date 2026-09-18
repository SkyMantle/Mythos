"""Sweep inspect / auto-peek gates. Keep engine._do_sweep thin."""
from __future__ import annotations

from typing import Any

from fpvscan.bands import band_of

HIT_TOL_HZ = 18.0e6
FFT_MIN_BW_HZ = 0.6e6
INSPECT_MIN_BW_HZ = 1.5e6
INSPECT_BW_HZ = 18.0e6
INSPECT_BW_5G8_HZ = 22.0e6
INSPECT_MS = 90.0
INSPECT_MS_5G8 = 110.0
INSPECT_MS_MIN = 5.0
INSPECT_MS_MAX = 200.0
# Live YAML sometimes stores LOCK capture_ms (20) in inspect_ms.
# Classification of a PAL comb needs a field-scale window, not that leak.
INSPECT_MS_CAPTURE_LEAK = 40.0
# Wall-clock cap for inspect offsets/decode so a queued LOCK is not stuck
# behind several cvbs.decode passes (GIL + no command drain).
INSPECT_WALL_S = 0.35
AUTO_PEEK_MIN_PIC = 0.12
SOFT_MAX_BW_HZ = 25.0e6
INSPECT_MIN_SNR_DB = 1.0
ENERGY_MIN_SNR_DB = 1.0
INSPECT_NARROW_FRAC = 0.4
LINE_PROMINENCE_DB = 2.0
STABLE_ANALOG_PROMINENCE_DB = 8.0
INSPECT_CONF_BYPASS = 0.10
INSPECT_MIN_LINES = 32
INSPECT_SNOW_CORR = 0.02
INSPECT_SNOW_LINES = 80


def fft_min_bw_hz(scan: dict[str, Any]) -> float:
    return float(scan.get("fft_min_bw_hz") or FFT_MIN_BW_HZ)


def uses_high_band_budget(freq_hz: float) -> bool:
    """Generic RF split for legacy wide-budget config keys."""
    return float(freq_hz) >= 2.0e9


def inspect_bw_hz(scan: dict[str, Any], freq_hz: float) -> float:
    if uses_high_band_budget(freq_hz):
        return float(scan.get("inspect_bw_5g8_hz") or INSPECT_BW_5G8_HZ)
    return float(scan.get("inspect_bw_hz") or INSPECT_BW_HZ)


def inspect_ms(scan: dict[str, Any], freq_hz: float) -> float:
    if uses_high_band_budget(freq_hz):
        raw = float(scan.get("inspect_ms_5g8") or INSPECT_MS_5G8)
        floor = INSPECT_MS_5G8
    else:
        raw = float(scan.get("inspect_ms") or INSPECT_MS)
        floor = INSPECT_MS
    if raw < INSPECT_MS_CAPTURE_LEAK:
        raw = floor
    return max(INSPECT_MS_MIN, min(INSPECT_MS_MAX, raw))


def inspect_wall_s(scan: dict[str, Any] | None = None) -> float:
    """Max seconds for inspect offset/decode after the IQ grab."""
    raw = float((scan or {}).get("inspect_wall_s") or INSPECT_WALL_S)
    return max(0.05, min(2.0, raw))


def should_full_inspect(
    *,
    dwell_hz: float,
    center_hz: float,
    bandwidth_hz: float,
    snr_db: float,
    from_extra: bool,
    scan: dict[str, Any],
    line_hint: bool = False,
) -> bool:
    """Expensive inspect only near a dwell, and only with a PAL/NTSC comb.

    Far coarse LTE blobs do not stall the pass.  Narrow analog next to the
    dwell keeps the 0.6× min_bw path.  Default ``inspect_need_comb`` /
    ``inspect_extras_need_comb`` skip the 90 ms decode when the dwell IQ
    already classified analog — or had no comb at all.
    """
    hit_tol = float(scan.get("hit_tol_hz") or HIT_TOL_HZ)
    min_bw = float(scan.get("min_bw_hz") or INSPECT_MIN_BW_HZ)
    max_bw = float(scan.get("max_bw_hz") or 30e6)
    min_snr = float(scan.get("inspect_min_snr_db", INSPECT_MIN_SNR_DB))
    near = abs(float(center_hz) - float(dwell_hz)) <= hit_tol
    bw = float(bandwidth_hz)
    narrow = float(scan.get("inspect_narrow_frac", INSPECT_NARROW_FRAC))
    if not (min_bw <= bw <= max_bw):
        if not (near and bw >= narrow * min_bw and float(snr_db) >= min_snr):
            return False
    if from_extra:
        if scan.get("inspect_extras_need_comb", True):
            return bool(line_hint)
        return True
    if not near:
        return False
    if scan.get("inspect_need_comb", True):
        return bool(line_hint)
    return True


def inspect_soft_ok(
    *,
    standard: str,
    bandwidth_hz: float,
    prominence_db: float,
    confidence: float,
    harmonics: int,
    row_corr: float,
    pic_lines: int,
    scan: dict[str, Any],
    center_hz: float = 0.0,
) -> bool:
    """Spectral PAL/NTSC bypass when 40 ms did not assemble a frame.

    PAL/NTSC with line harmonics skip the 15 MHz cap and 0.90×inspect_bw
    5018-snow rule. Snow rasters (many lines, no row corr) still fail.
    """
    if standard not in ("PAL", "NTSC"):
        return False
    bw = float(bandwidth_hz)
    has_comb = int(harmonics) >= 1
    min_bw = float(scan.get("min_bw_hz") or INSPECT_MIN_BW_HZ)
    if has_comb:
        max_bw = float(scan.get("max_bw_hz") or 30e6)
        if not (min_bw <= bw <= max_bw):
            return False
    elif not (min_bw <= bw <= SOFT_MAX_BW_HZ):
        return False
    corr = float(row_corr)
    snow_corr = float(scan.get("inspect_snow_corr", INSPECT_SNOW_CORR))
    snow_lines = int(scan.get("inspect_snow_lines", INSPECT_SNOW_LINES))
    if int(pic_lines) >= snow_lines and corr < snow_corr:
        return False
    if not has_comb:
        cap = inspect_bw_hz(scan, center_hz)
        if bw >= 0.90 * cap and corr < 0.04:
            return False
    prom = float(scan.get("line_prominence_db", LINE_PROMINENCE_DB))
    if float(prominence_db) < prom:
        return False
    bypass = float(scan.get("inspect_conf_bypass", INSPECT_CONF_BYPASS))
    if float(confidence) >= bypass:
        return True
    return has_comb


def energy_hit_ok(
    *,
    dwell_hz: float,
    center_hz: float,
    bandwidth_hz: float,
    snr_db: float,
    scan: dict[str, Any],
    offset_db: float | None = None,
) -> bool:
    """Publish FFT occupancy as a hit without PAL/NTSC — like a spectrum scanner.

    Limited to FPV bands (or near the dwell) so 400–6000 MHz does not flood
    the list with every LTE/Wi-Fi blob.
    """
    if not scan.get("accept_energy", False):
        return False
    min_snr = float(scan.get("energy_min_snr_db", ENERGY_MIN_SNR_DB))
    if offset_db is not None:
        min_snr = min(min_snr, float(offset_db))
    min_bw = fft_min_bw_hz(scan)
    max_bw = float(scan.get("max_bw_hz") or 30e6)
    if float(snr_db) < min_snr:
        return False
    bw = float(bandwidth_hz)
    if not (min_bw <= bw <= max_bw):
        return False
    hit_tol = float(scan.get("hit_tol_hz") or HIT_TOL_HZ)
    near = abs(float(center_hz) - float(dwell_hz)) <= hit_tol
    return near or band_of(float(center_hz)) != "—"


def auto_peek_allowed(det: Any, scan: dict[str, Any]) -> bool:
    """Allow LOCK for decoded video or repeated bounded analogue evidence."""
    if not scan.get("auto_peek", True):
        return False
    conf = float(getattr(det, "confidence", 0.0) or 0.0)
    analog = bool(getattr(det, "analog_evidence", False))
    votes = int(getattr(det, "inspect_votes", 0) or 0)
    prominence = float(getattr(det, "prominence_db", 0.0) or 0.0)
    stable_analog = bool(
        analog
        and votes >= 2
        and prominence >= float(scan.get(
            "inspect_stable_prominence_db", STABLE_ANALOG_PROMINENCE_DB,
        ))
    )
    if conf < float(scan.get("auto_peek_min_conf", 0.6)) and not stable_analog:
        return False
    pic = float(getattr(det, "pic_score", 0.0) or 0.0)
    locked = bool(getattr(det, "pic_locked", False))
    min_pic = float(scan.get("auto_peek_min_pic") or AUTO_PEEK_MIN_PIC)
    if pic < min_pic and not locked and not stable_analog:
        return False
    return True
