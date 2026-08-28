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


def noise_floor_db(psd: np.ndarray) -> float:
    """Робастна оцінка шумової підлоги — медіана стійка до сигналів,
    що займають до половини смуги."""
    return float(np.median(psd))


def usable_view(psd: np.ndarray, fs: float, edge_guard: float = 0.9,
                dc_notch_hz: float = 200e3) -> np.ndarray:
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
    v[max(0, dc - h):dc + h + 1] = np.median(v)
    return v


@dataclass
class Occupancy:
    center_hz: float
    bandwidth_hz: float
    peak_db: float
    snr_db: float


def find_occupied(psd: np.ndarray, center_hz: float, fs: float,
                  threshold_db: float = 8.0,
                  edge_guard: float = 0.9,
                  min_bw_hz: float = 2e6,
                  dc_notch_hz: float = 200e3) -> list[Occupancy]:
    """Знаходить неперервні ділянки, що піднімаються над підлогою.

    edge_guard відкидає краї смуги, де завалює ФНЧ приймача і де
    сидить дзеркало/LO-витік.
    """
    nfft = len(psd)
    bin_hz = fs / nfft
    guard = int(nfft * (1 - edge_guard) / 2)
    view = psd[guard:nfft - guard]
    nf = noise_floor_db(view)

    # Прибираємо сплеск на нулі. На реальному приймачі це не один бін:
    # витік гетеродина плюс постійне зміщення АЦП дають помітний горб,
    # який інакше знаходиться на кожному кроці свіпу як «передача».
    dc = nfft // 2 - guard
    half_notch = max(1, int(dc_notch_hz / bin_hz / 2))
    view = view.copy()
    view[max(0, dc - half_notch):dc + half_notch + 1] = nf

    mask = view > (nf + threshold_db)

    # ЧМ-відео дає «рвану» вершину з провалами — замикаємо дірки
    # завширшки до gap бінів, інакше один канал розсипається на десяток.
    gap = max(1, int(1.0e6 / bin_hz))
    if gap > 1:
        k = np.ones(2 * gap + 1, dtype=bool)
        dil = np.convolve(mask, k, mode="same") > 0
        mask = np.convolve(~dil, k, mode="same") == 0

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
            seg = view[i:j]
            lo = (guard + i - nfft / 2) * bin_hz
            hi = (guard + j - nfft / 2) * bin_hz
            out.append(Occupancy(
                center_hz=center_hz + (lo + hi) / 2,
                bandwidth_hz=bw,
                peak_db=float(seg.max()),
                snr_db=float(seg.max() - nf),
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
