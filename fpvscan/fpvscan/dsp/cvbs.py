"""Декодер композитного відео (CVBS) у растровий кадр.

Вхід — демодульований відеосигнал. Вихід — напівкадр у градаціях
сірого. Кольорову піднесучу навмисно не декодуємо: у реальному
перехопленні відношення сигнал/шум зазвичай таке, що яскравість
читається, а колірна синхронізація вже розсипається. Яскравіша
картинка дає все, що потрібно для розпізнавання обстановки.

Порядок роботи:
  1. визначення полярності (у якому боці вершина синхроімпульсу)
  2. поріг між вершиною синхри та рівнем гасіння
  3. пошук фронтів рядкової синхри, оцінка періоду рядка
  4. пошук кадрової синхри (широкі імпульси, > пів рядка низького рівня)
  5. вирізання активної частини кожного рядка та ресемплінг у пікселі
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

SYNC_US = 4.7e-6
# BACK_PORCH_US = 9.4e-6
# FRONT_PORCH_US = 1.5e-6
STD_GEOM = {
    #          початок активної,   кінець,   рядків кадрового гасіння
    "PAL":  (10.5 / 64.0,       62.5 / 64.0,   25),
    "NTSC": (9.4 / 63.556,      62.0 / 63.556, 20),
    "?":    (10.0 / 64.0,       62.3 / 64.0,   22),
}


@dataclass
class Frame:
    luma: np.ndarray       # uint8, (висота, ширина)
    line_rate: float
    lines: int
    standard: str
    locked: bool


def _sync_edges(v: np.ndarray, thr: float):
    below = v < thr
    edges = np.flatnonzero(below[1:] & ~below[:-1]) + 1
    return below, edges


def _attempt(v: np.ndarray, fs: float, width: int, max_lines: int):
    """Одна спроба декодування за заданої полярності.

    Повертає (оцінка_якості, Frame|None). Оцінка — частка міжсинхронних
    інтервалів, що лягли в ±20% від медіани. У шумі або при
    перевернутому сигналі вона розсипається, тому за нею й обираємо
    полярність.
    """
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi - lo < 1e-9:
        return 0.0, None
    v = (v - lo) / (hi - lo)          # вершина синхри ≈ 0, білий ≈ 1

    thr = 0.18                        # між вершиною синхри і рівнем гасіння
    below, edges = _sync_edges(v, thr)
    if len(edges) < 24:
        return 0.0, None

    d = np.diff(edges).astype(np.float64)
    # орієнтир — найдовший поширений інтервал (півкадрові імпульси коротші)
    med = float(np.median(d[d > np.percentile(d, 40)]))
    keep = (d > med * 0.8) & (d < med * 1.2)
    score = float(keep.mean())
    if keep.sum() < 12:
        return 0.0, None
    period = float(np.median(d[keep]))
    e = edges.astype(np.float64)
    k = np.round((e - e[0]) / period)
    good_k = np.abs((e - e[0]) - k * period) < period * 0.2
    if good_k.sum() >= 8:
        kk, ee = k[good_k], e[good_k]
        A = np.vstack([kk, np.ones_like(kk)]).T
        period, _ = np.linalg.lstsq(A, ee, rcond=None)[0]
        period = float(period)
    line_rate = fs / period
    if not (14000 < line_rate < 17500):
        return 0.0, None
    standard = ("PAL" if abs(line_rate - 15625) < 120 else
                "NTSC" if abs(line_rate - 15734) < 120 else "?")
    a0_frac, a1_frac, vblank = STD_GEOM[standard]

    # --- кадрова синхра: вікно в один рядок, де низького рівня > 55% ---
    win = int(period)
    csum = np.cumsum(np.concatenate(([0], below.astype(np.int32))))
    frac = (csum[win:] - csum[:-win]) / win
    vs = np.flatnonzero(frac > 0.55)
    locked = len(vs) > 0
    if locked:
        brk = np.flatnonzero(np.diff(vs) > win)
        end_vs = vs[brk[0]] if len(brk) else vs[-1]
        start = int(end_vs + win * vblank)     # пропускаємо кадрове гасіння
    else:
        start = int(edges[0])

    nxt = edges[edges >= start]
    if len(nxt) == 0:
        nxt = edges
    t0 = float(nxt[0])

    a0 = a0_frac * period
    a1 = a1_frac * period
    n_lines = int(min(max_lines, (len(v) - t0 - period) / period))
    if a1 <= a0 or n_lines < 32:
        return 0.0, None

    offs = np.linspace(a0, a1, width, dtype=np.float32)
    idx = (t0 + np.arange(n_lines, dtype=np.float32)[:, None] * period
           + offs[None, :])
    idx = np.clip(idx, 0, len(v) - 2)
    i0 = idx.astype(np.int32)
    fr = idx - i0
    samp = v[i0] * (1 - fr) + v[i0 + 1] * fr      # лінійна інтерполяція

    luma = np.clip((samp - 0.30) / 0.70, 0, 1)    # гасіння -> чорний
    return score, Frame(
        luma=(luma * 255).astype(np.uint8),
        line_rate=line_rate, lines=n_lines,
        standard=standard, locked=locked)


def decode(base: np.ndarray, fs: float, width: int = 640,
           max_lines: int = 288) -> Frame | None:
    """Декодує напівкадр. Полярність визначається перебором:
    у реальному ефірі вона залежить від передавача, а гістограмна
    евристика на слабкому сигналі помиляється."""
    if len(base) < int(fs * 0.02):        # менше 20 мс — нема сенсу
        return None
    v = base.astype(np.float32)
    best_s, best_f = 0.0, None
    for sign in (1.0, -1.0):
        s, f = _attempt(v * sign, fs, width, max_lines)
        if f is not None and s > best_s:
            best_s, best_f = s, f
    return best_f if best_s > 0.5 else None


def encode(frame: Frame, fmt: str = "webp", quality: int = 80,
           height: int | None = 480) -> bytes:
    """Кадр у стиснений формат.

    WebP на сірій картинці дає приблизно вчетверо менший файл, ніж
    JPEG тієї ж візуальної якості, і підтримується всіма браузерами.
    Саме він іде і в веб-консоль, і в знімки.
    """
    from io import BytesIO
    from PIL import Image
    img = Image.fromarray(frame.luma, mode="L")
    if height:
        img = img.resize((frame.luma.shape[1], height), Image.BILINEAR)
    buf = BytesIO()
    f = fmt.lower()
    if f == "png":
        img.save(buf, "PNG", optimize=True)
    elif f in ("jpeg", "jpg"):
        img.save(buf, "JPEG", quality=quality)
    else:
        img.save(buf, "WEBP", quality=quality, method=4)
    return buf.getvalue()


def to_jpeg(frame: Frame, quality: int = 70, height: int = 480) -> bytes:
    return encode(frame, "jpeg", quality, height)
