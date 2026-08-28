"""Плани діапазонів, де реально живе аналогове FPV-відео.

Скан 400 МГц – 6 ГГц робиться суцільно, але ці зони обходяться
частіше (пріоритетний список), бо там ~99% трафіку.
"""
from dataclasses import dataclass

MHz = 1e6
GHz = 1e9


@dataclass(frozen=True)
class Band:
    name: str
    start_hz: float
    stop_hz: float
    typical_bw_hz: float  # зайнята смуга одного каналу


# Основні FPV-діапазони. typical_bw — ширина ЧМ-відеоканалу за Карсоном.
PRIORITY_BANDS = [
    Band("433",     420 * MHz,   470 * MHz,   10 * MHz),
    Band("900",     840 * MHz,   960 * MHz,   14 * MHz),
    Band("1G2",    1040 * MHz,  1400 * MHz,   18 * MHz),
    Band("2G4",    2300 * MHz,  2550 * MHz,   20 * MHz),
    Band("3G3",    3100 * MHz,  4000 * MHz,   20 * MHz),
    Band("5G8",    5100 * MHz,  6000 * MHz,   27 * MHz),
]

# Класична сітка 5.8 ГГц (A/B/E/F/R), для прив'язки знахідки до каналу.
RACEBAND = {f"R{i+1}": f for i, f in enumerate(
    [5658, 5695, 5732, 5769, 5806, 5843, 5880, 5917])}
BAND_F = {f"F{i+1}": f for i, f in enumerate(
    [5740, 5760, 5780, 5800, 5820, 5840, 5860, 5880])}
BAND_A = {f"A{i+1}": f for i, f in enumerate(
    [5865, 5845, 5825, 5805, 5785, 5765, 5745, 5725])}
BAND_B = {f"B{i+1}": f for i, f in enumerate(
    [5733, 5752, 5771, 5790, 5809, 5828, 5847, 5866])}
BAND_E = {f"E{i+1}": f for i, f in enumerate(
    [5705, 5685, 5665, 5645, 5885, 5905, 5925, 5945])}

ALL_5G8 = {**RACEBAND, **BAND_F, **BAND_A, **BAND_B, **BAND_E}


def nearest_channel(freq_hz: float, tol_mhz: float = 6.0) -> str | None:
    """Повертає ім'я каналу сітки 5.8 ГГц, якщо частота близька до нього."""
    f_mhz = freq_hz / MHz
    best, best_d = None, 1e9
    for name, ch in ALL_5G8.items():
        d = abs(ch - f_mhz)
        if d < best_d:
            best, best_d = name, d
    return best if best_d <= tol_mhz else None


def band_of(freq_hz: float) -> str:
    for b in PRIORITY_BANDS:
        if b.start_hz <= freq_hz <= b.stop_hz:
            return b.name
    return "—"
