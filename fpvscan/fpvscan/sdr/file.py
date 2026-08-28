"""Відтворення записаного ефіру з файлу.

Це основний інструмент відлагодження після того, як з'явилось залізо.
Записав 20 секунд реального стенду на Pi -> перетягнув .cf32 на
Windows -> ганяєш алгоритми скільки треба, з тим самим сигналом і
відтворюваним результатом. Без цього кожна зміна порогу вимагає
йти до стенду.

Формат: сирі complex64 (I,Q float32 по черзі), поруч .json з
параметрами захоплення.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from .base import SdrSource


class FileSource(SdrSource):
    name = "file"
    fixed_freq = True

    def __init__(self, path: str | Path, loop: bool = True,
                 realtime: bool = False):
        self.path = Path(path)
        self.loop = loop
        self.realtime = realtime      # чи витримувати реальний темп
        self._data: np.ndarray | None = None
        self._pos = 0
        self._meta: dict = {}
        self._fc = 0.0
        self._fs = 0.0

    def open(self):
        meta_p = self.path.with_suffix(".json")
        if meta_p.exists():
            self._meta = json.loads(meta_p.read_text(encoding="utf-8"))
        self._fc = float(self._meta.get("center_hz", 0.0))
        self._fs = float(self._meta.get("sample_rate", 40e6))
        self._data = np.fromfile(self.path, dtype=np.complex64)
        if self._data.size == 0:
            raise IOError(f"Порожній запис: {self.path}")
        dur = self._data.size / self._fs
        print(f"[file] {self.path.name}: {self._data.size} відл., "
              f"{dur:.2f} с, {self._fc/1e6:.1f} МГц @ {self._fs/1e6:.1f} Мвідл/с")

    def close(self):
        self._data = None

    def read(self, n: int) -> np.ndarray:
        d = self._data
        if d is None:
            raise IOError("Джерело не відкрите")
        if self._pos + n > d.size:
            if not self.loop:
                raise EOFError("Запис закінчився")
            self._pos = 0
        out = d[self._pos:self._pos + n]
        self._pos += n
        if out.size < n:                      # запис коротший за запит
            reps = int(np.ceil(n / out.size))
            out = np.tile(out, reps)[:n]
        if self.realtime:
            import time
            time.sleep(n / self._fs)
        return out.copy()

    # Частота і смуга зафіксовані записом. Перебудову ігноруємо, але
    # чесно повідомляємо, щоб не гадати, чому свіп нічого не бачить.
    def set_center_freq(self, hz: float):
        if abs(hz - self._fc) > self._fs / 2:
            pass                              # поза записом — віддамо той самий шматок

    def set_sample_rate(self, hz: float):
        pass

    def set_gain(self, db: float):
        pass

    @property
    def center_freq(self): return self._fc

    @property
    def sample_rate(self): return self._fs

    def retune_and_read(self, hz: float, n: int) -> np.ndarray:
        return self.read(n)


def write_capture(path: str | Path, iq: np.ndarray, center_hz: float,
                  sample_rate: float, gain_db: float = 0.0, note: str = ""):
    p = Path(path)
    iq.astype(np.complex64).tofile(p)
    p.with_suffix(".json").write_text(json.dumps({
        "center_hz": center_hz,
        "sample_rate": sample_rate,
        "gain_db": gain_db,
        "samples": int(iq.size),
        "duration_s": float(iq.size / sample_rate),
        "format": "complex64",
        "note": note,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
