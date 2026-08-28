"""Симулятор ефіру: дозволяє розробляти й тестувати весь конвеєр
без bladeRF, на звичайному ноутбуці під Windows.

Генерує шумову підлогу плюс задані передавачі. Передавач типу
"fpv" — це справжній композитний відеосигнал (синхроімпульси,
кадрова синхра, тестова таблиця), промодульований частотною
модуляцією. Тобто демодулятор і декодер CVBS отримують на вхід
рівно те, що вони побачать в реальному ефірі, і на екрані має
з'явитись картинка.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .base import SdrSource

C_LINE_NTSC = 15734.264  # Гц, рядкова частота
C_LINE_PAL = 15625.0


@dataclass
class Emitter:
    freq_hz: float
    power_db: float = -20.0      # відносно повної шкали
    kind: str = "fpv"            # "fpv" | "cw" | "noise"
    deviation_hz: float = 10.0e6  # девіація ЧМ відео (розмах ~10 МГц)
    line_rate: float = C_LINE_NTSC
    label: str = "sim"


def _cvbs(t: np.ndarray, line_rate: float) -> np.ndarray:
    """Композитний відеосигнал у вигляді функції часу.

    Рівні (нормовані): вершина синхри 0.0, гасіння 0.30, білий 1.0.
    Векторизовано, тому дешево навіть на 40 Мвідл/с.
    """
    lt = 1.0 / line_rate                 # період рядка
    line = np.floor(t / lt).astype(np.int64)
    x = t - line * lt                    # позиція всередині рядка

    sync_w = 4.7e-6
    bp_end = 9.4e-6                      # кінець задньої площадки
    active = lt - 1.5e-6                 # початок передньої площадки

    v = np.full(t.shape, 0.30, dtype=np.float32)

    # --- горизонтальна синхра ---
    v[x < sync_w] = 0.0

    # --- активна частина: тестова таблиця ---
    in_act = (x >= bp_end) & (x < active)
    u = (x - bp_end) / (active - bp_end)          # 0..1 по горизонталі
    field_line = line % 262
    row = np.clip((field_line - 20) / 220.0, 0, 1)  # 0..1 по вертикалі

    # 8 вертикальних градацій сірого + рухома рамка, щоб бачити «живе»
    bars = np.floor(u * 8) / 7.0
    phase = (line // 262) * 0.02
    box = ((np.abs(u - (0.5 + 0.3 * np.sin(phase * 6.28))) < 0.06)
           & (np.abs(row - 0.5) < 0.12))
    pic = 0.30 + 0.70 * bars
    pic = np.where(box, 1.0, pic)
    pic = np.where(row < 0.02, 0.30, pic)          # рамка зверху
    v[in_act] = pic[in_act].astype(np.float32)

    # --- кадрова синхра: 3 рядки широких імпульсів на початку поля ---
    vs = field_line < 3
    if np.any(vs):
        x2 = np.mod(x, lt / 2)          # широкі імпульси йдуть з півперіодом
        broad = np.where(x2 < (lt / 2 - 4.7e-6), 0.0, 0.30).astype(np.float32)
        v[vs] = broad[vs]
    # зрівнювальні + гасіння
    vb = (field_line >= 3) & (field_line < 20)
    if np.any(vb):
        v[vb & ~(x < sync_w)] = 0.30
    return v


class SimSource(SdrSource):
    name = "sim"

    def __init__(self, emitters: list[Emitter] | None = None,
                 noise_db: float = -70.0, seed: int = 0):
        self.emitters = emitters if emitters is not None else default_scene()
        self.noise_db = noise_db
        self._fc = 5800e6
        self._fs = 40e6
        self._gain = 40.0
        self._t = 0.0                     # глобальний час, безперервний
        self._phase: dict[int, float] = {}
        self._rng = np.random.default_rng(seed)
        self._open = False

    # --- керування ---
    def open(self): self._open = True

    def close(self): self._open = False

    def set_center_freq(self, hz): self._fc = float(hz)

    def set_sample_rate(self, hz): self._fs = float(hz)

    def set_gain(self, db): self._gain = float(db)

    @property
    def center_freq(self): return self._fc

    @property
    def sample_rate(self): return self._fs

    def retune_and_read(self, hz, n):
        # у симуляторі перебудова миттєва (як quick-tune на bladeRF)
        self.set_center_freq(hz)
        return self.read(n)

    # --- дані ---
    def read(self, n: int) -> np.ndarray:
        fs = self._fs
        t = self._t + np.arange(n, dtype=np.float64) / fs

        amp_n = 10 ** (self.noise_db / 20)
        out = (self._rng.normal(0, amp_n, n)
               + 1j * self._rng.normal(0, amp_n, n)).astype(np.complex64)

        half = fs / 2
        for i, em in enumerate(self.emitters):
            off = em.freq_hz - self._fc
            dev = max(em.deviation_hz, 1.0)
            # Вхідний ФНЧ не відрізає сигнал разом, а пропускає ту
            # частину смуги, що потрапила всередину. Саме тому канал,
            # який ліг на край смуги, ще видно — але вже покаліченим.
            pb = half * 0.9
            inside = (min(off + dev, pb) - max(off - dev, -pb)) / (2 * dev)
            if inside <= 0.05 or abs(off) > pb:
                continue
            amp = 10 ** (em.power_db / 20) * inside
            ph0 = self._phase.get(i, 0.0)

            if em.kind == "cw":
                ph = ph0 + 2 * np.pi * off * (t - self._t)
            else:
                v = _cvbs(t, em.line_rate)
                # ЧМ: вершина синхри — нижня частота (як у FPV-передавачах)
                inst = off + (v - 0.30) * em.deviation_hz
                ph = ph0 + 2 * np.pi * np.cumsum(inst) / fs
            self._phase[i] = float(ph[-1] % (2 * np.pi))
            out += (amp * np.exp(1j * ph)).astype(np.complex64)

        self._t += n / fs
        return out * (10 ** ((self._gain - 40) / 40))


def default_scene() -> list[Emitter]:
    """Типова обстановка: два аналогові борти + пара завад."""
    return [
        Emitter(5800e6, -18.0, "fpv", 10.0e6, C_LINE_NTSC, "борт-1 F4"),
        Emitter(1280e6, -26.0, "fpv", 8.0e6, C_LINE_PAL, "борт-2 1G2"),
        Emitter(2437e6, -30.0, "cw", label="wifi-подібна завада"),
        Emitter(433.9e6, -35.0, "cw", label="телеметрія"),
    ]
