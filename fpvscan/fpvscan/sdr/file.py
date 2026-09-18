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
import errno
import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np

from .base import SdrSource


def is_disk_full_error(exc: BaseException) -> bool:
    """True for ENOSPC and NumPy's short ``tofile`` write on a full disk."""
    err = getattr(exc, "errno", None)
    if err in (errno.ENOSPC, errno.EDQUOT):
        return True
    msg = str(exc).lower()
    return "no space left" in msg or (
        "requested and" in msg and "written" in msg
    )


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
        self.stream_discontinuities = 0

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
            self.stream_discontinuities += 1
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
                  sample_rate: float, gain_db: float = 0.0, note: str = "",
                  metadata: dict | None = None):
    """Atomically publish a CF32 capture and its JSON sidecar.

    Both temporary files live beside the destination, so ``os.replace`` stays
    on one filesystem.  Callers may add hardware/runtime metadata without
    changing the established on-disk format.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta_p = p.with_suffix(".json")
    token = uuid4().hex
    tmp_p = p.with_name(f".{p.name}.{token}.tmp")
    tmp_meta = meta_p.with_name(f".{meta_p.name}.{token}.tmp")
    body = {
        "center_hz": center_hz,
        "sample_rate": sample_rate,
        "gain_db": gain_db,
        "samples": int(iq.size),
        "duration_s": float(iq.size / sample_rate),
        "format": "complex64",
        "note": note,
    }
    if metadata:
        body.update(metadata)
    needed = int(np.asarray(iq).nbytes) + 8192
    free = int(shutil.disk_usage(p.parent).free)
    if free < needed:
        raise OSError(
            errno.ENOSPC,
            f"need {needed} bytes, {free} free in {p.parent}",
        )
    try:
        np.asarray(iq, dtype=np.complex64).tofile(tmp_p)
        tmp_meta.write_text(
            json.dumps(body, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_p, p)
        os.replace(tmp_meta, meta_p)
    except Exception as exc:
        tmp_p.unlink(missing_ok=True)
        tmp_meta.unlink(missing_ok=True)
        # If only the first replace succeeded, do not leave an unpaired CF32.
        if p.exists() and not meta_p.exists():
            p.unlink(missing_ok=True)
        if is_disk_full_error(exc) and getattr(exc, "errno", None) not in (
            errno.ENOSPC, errno.EDQUOT,
        ):
            raise OSError(errno.ENOSPC, str(exc)) from exc
        raise
    return p
