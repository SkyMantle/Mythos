"""Оцінка спектра та пошук зайнятих смуг."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

_WIN_CACHE: dict[int, np.ndarray] = {}


def _win(n: int) -> np.ndarray:
    w = _WIN_CACHE.get(n)
    if w is None:
        w = np.hanning(n).astype(np.float32)
        _WIN_CACHE[n] = w
    return w


def psd_db(iq: np.ndarray, nfft: int = 4096, averages: int = 8) -> np.ndarray:
    """Усереднена періодограма, дБ відносно повної шкали, впорядкована
    від -fs/2 до +fs/2."""
    n = min(averages, max(1, len(iq) // nfft))
    w = _win(nfft)
    acc = np.zeros(nfft, np.float32)
    for k in range(n):
        seg = iq[k * nfft:(k + 1) * nfft] * w
        acc += np.abs(np.fft.fft(seg, nfft)) ** 2
    acc /= (n * nfft * np.sum(w ** 2) / nfft)
    return 10 * np.log10(np.fft.fftshift(acc) + 1e-20)


def noise_floor_db(psd: np.ndarray, percentile: float = 25.0) -> float:
    """Робастна шумова підлога.

    Медіана (50) тоне, коли ЧМ-полиця займає більше половини вікна —
    тоді «підлога» = сама полиця і 3–10 дБ аналог зникає. 25-й
    перцентиль лишається в шумі, поки зайнятість < ~75%.
    """
    x = np.asarray(psd, dtype=np.float64)
    if x.size == 0:
        return -120.0
    p = float(percentile)
    if p >= 49.5:
        return float(np.median(x))
    return float(np.percentile(x, np.clip(p, 1.0, 49.0)))


def noise_spread_db(psd: np.ndarray) -> float:
    """MAD бінів на рівні підлоги або нижче — σ шуму, не полиці."""
    x = np.asarray(psd, dtype=np.float64)
    if x.size < 8:
        return 0.0
    med = float(np.median(x))
    low = x[x <= med]
    if low.size < 8:
        low = x
    mad = float(np.median(np.abs(low - np.median(low))))
    return 1.4826 * mad


def _static_threshold_db(scan: dict | None) -> float:
    raw = (scan or {}).get("threshold_db", 1.0)
    if raw in ("auto", "Auto", "AUTO", "", None):
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def occupancy_offset_db(view: np.ndarray, scan: dict | None = None) -> float:
    """CFAR-зсув над робастною підлогою для цього dwell.

    auto: max(offset, threshold_db як мінімум, k·MAD), у [min, max].
    fixed: threshold_db як раніше. Не піднімає планку «бо шумно».
    """
    sc = scan or {}
    mode = str(sc.get("threshold_mode") or "auto").strip().lower()
    static = _static_threshold_db(sc)
    lo = float(sc.get("threshold_min_db", 1.0))
    hi = float(sc.get("threshold_max_db", 2.5))
    if mode in ("fixed", "off", "manual"):
        return max(0.0, static if static > 0 else lo)
    offset = float(sc.get("threshold_offset_db", 1.2))
    k = float(sc.get("threshold_k", 0.75))
    spread = noise_spread_db(view)
    raw = max(offset, static, k * spread)
    if hi > 0:
        raw = min(raw, hi)
    return max(lo, raw)


def usable_view(psd: np.ndarray, fs: float, edge_guard: float = 0.95,
                dc_notch_hz: float = 200e3,
                noise_percentile: float = 25.0) -> np.ndarray:
    """Спектр без країв смуги і без сплеску на нулі.

    Мірити рівень сигналу по повному спектру не можна: постійна
    складова АЦП і витік гетеродина сидять у центрі, не залежать від
    підсилення і забивають собою максимум.
    """
    nfft = len(psd)
    bin_hz = fs / nfft
    guard = int(nfft * (1 - edge_guard) / 2)
    v = psd[guard:nfft - guard].copy()
    dc = nfft // 2 - guard
    h = max(1, int(dc_notch_hz / bin_hz / 2))
    nf = noise_floor_db(v, percentile=noise_percentile)
    v[max(0, dc - h):dc + h + 1] = nf
    return v


@dataclass
class Occupancy:
    center_hz: float
    bandwidth_hz: float
    peak_db: float
    snr_db: float
    peak_hz: float = 0.0
    occupancy_fraction: float = 0.0
    edge_margin_hz: float = 0.0
    center_method: str = "energy_centroid"


def find_occupied(psd: np.ndarray, center_hz: float, fs: float,
                  threshold_db: float = 1.5,
                  edge_guard: float = 0.95,
                  min_bw_hz: float = 0.6e6,
                  dc_notch_hz: float = 200e3,
                  noise_percentile: float = 25.0) -> list[Occupancy]:
    """Знаходить неперервні ділянки, що піднімаються над підлогою.

    edge_guard відкидає краї смуги, де завалює ФНЧ приймача і де
    сидить дзеркало/LO-витік. Підлога — percentile, не медіана.
    """
    nfft = len(psd)
    bin_hz = fs / nfft
    guard = int(nfft * (1 - edge_guard) / 2)
    view = usable_view(
        psd, fs, edge_guard=edge_guard, dc_notch_hz=dc_notch_hz,
        noise_percentile=noise_percentile)
    nf = noise_floor_db(view, percentile=noise_percentile)

    mask = view > (nf + threshold_db)

    # ЧМ-відео дає «рвану» вершину з короткими провалами.  This must be a
    # real binary closing (dilate then erode): the former implementation
    # accidentally dilated twice, so sparse noise peaks grew into one region
    # spanning 95% of every capture.
    gap = max(1, int(0.25e6 / bin_hz))
    if gap > 1:
        k = np.ones(gap + 1, dtype=np.int16)
        dil = np.convolve(mask.astype(np.int16), k, mode="same") > 0
        mask = np.convolve(dil.astype(np.int16), k, mode="same") >= len(k)

    out: list[Occupancy] = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        bw = (j - i) * bin_hz
        if bw >= min_bw_hz:
            # A short region clipped by the analogue front-end edge is much
            # more likely to be filter roll-off than a usable proposal.  A
            # genuinely broad FM shelf is retained and will be observed again
            # in the overlapping dwell.
            touches_edge = i == 0 or j == len(mask)
            if touches_edge and bw < max(min_bw_hz, 0.08 * fs):
                i = j
                continue
            seg = view[i:j]
            local = int(np.argmax(seg))
            frac = 0.0
            if 0 < local < len(seg) - 1:
                y0, y1, y2 = float(seg[local - 1]), float(seg[local]), float(seg[local + 1])
                denom = y0 - 2.0 * y1 + y2
                if abs(denom) > 1e-6:
                    frac = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
            peak_bin = guard + i + local + frac
            peak_hz = center_hz + (peak_bin - nfft / 2) * bin_hz

            # Analogue FM video is commonly a broad, uneven shelf.  Its
            # strongest FFT bin can sit several MHz from the RF centre, so use
            # the occupied-energy centroid for tuning and keep the peak only
            # as a diagnostic.  Clip weights to keep one narrow spur from
            # dragging a broad proposal back to its peak.
            excess = np.maximum(seg.astype(np.float64) - nf, 0.0)
            if excess.size:
                cap = float(np.percentile(excess, 80.0))
                weights = np.minimum(excess, cap) if cap > 0 else np.ones_like(excess)
                total = float(np.sum(weights))
            else:
                weights = excess
                total = 0.0
            center_local = (
                float(np.dot(np.arange(i, j, dtype=np.float64) + 0.5, weights) / total)
                if total > 1e-12 else (i + j) / 2.0
            )
            occupied_center_bin = guard + center_local
            occupied_center_hz = (
                center_hz + (occupied_center_bin - nfft / 2) * bin_hz
            )
            edge_bins = min(i, len(mask) - j)
            out.append(Occupancy(
                center_hz=occupied_center_hz,
                bandwidth_hz=bw,
                peak_db=float(seg.max()),
                snr_db=float(seg.max() - nf),
                peak_hz=peak_hz,
                occupancy_fraction=float((j - i) / max(1, len(mask))),
                edge_margin_hz=float(edge_bins * bin_hz),
            ))
        i = j
    return out


def downsample_for_display(psd: np.ndarray, target: int = 512) -> list[float]:
    """Стискає спектр для передачі у браузер, зберігаючи піки (max-hold)."""
    if len(psd) <= target:
        return [round(float(v), 1) for v in psd]
    step = len(psd) // target
    trimmed = psd[:step * target].reshape(target, step)
    return [round(float(v), 1) for v in trimmed.max(axis=1)]
