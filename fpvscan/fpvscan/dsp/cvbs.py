"""Декодер композитного відео (CVBS) у растровий кадр.

Вхід — демодульований відеосигнал. Вихід — напівкадр у градаціях
сірого. Кольорову піднесучу навмисно не декодуємо: у реальному
перехопленні відношення сигнал/шум зазвичай таке, що яскравість
читається, а колірна синхронізація вже розсипається. Яскравіша
картинка дає все, що потрібно для розпізнавання обстановки.

Два режими пошуку кадрової синхри:

  * "сліпий" (`_attempt`) — повний перебір: полярність, поріг,
    пошук усіх фронтів рядкової синхри, МНК-уточнення періоду,
    пошук кадрової синхри за часткою низького рівня у вікні.
    Використовується на першому виклику і як запасний варіант.

  * "трекінг" (`_attempt_tracked`) — коли з попереднього успішного
    кадру відомі період і полярність (`DecodeState`), наступний
    фронт кадрової синхри прогнозується екстраполяцією по цілій
    кількості польових періодів і шукається лише у вузькому вікні
    навколо прогнозу. Це і є успадкування фази між блоками
    захоплення: кожен новий знімок з кільцевого буфера не починає
    пошук синхри «з нуля», а продовжує з того місця, де зупинився
    попередній, — звідси стабільніша, менш «стрибаюча» картинка.
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
# орієнтовна кількість рядків в одному півкадрі (полі) — потрібна лише
# для екстраполяції позиції наступної кадрової синхри в трекінгу,
# точність тут не критична (похибка в межах вікна пошуку tol_frac*period)
FIELD_LINES = {"PAL": 312.5, "NTSC": 262.5, "?": 287.5}


@dataclass
class Frame:
    luma: np.ndarray       # uint8, (висота, ширина)
    line_rate: float
    lines: int
    standard: str
    locked: bool


@dataclass
class DecodeState:
    """Пам'ять декодера між послідовними викликами decode() для одного
    каналу. Тримає Engine (по одному екземпляру на LOCK), передається
    в decode() і оновлюється на місці.

    `abs_t0` — абсолютна позиція (у відліках потоку, у тій самій шкалі,
    що й параметр `abs_start` у decode()) останнього достовірно
    знайденого фронту кадрової синхри. Саме вона й дозволяє прогнозувати
    наступний фронт незалежно від того, наскільки новий знімок
    зсунутий чи розірваний відносно попереднього.
    """
    sign: float = 1.0
    period: float | None = None      # період рядка, у відліках поточної fs
    standard: str = "?"
    abs_t0: float | None = None
    lost: int = 0                    # підряд невдалих спроб трекінгу


def _sync_edges(v: np.ndarray, thr: float):
    below = v < thr
    edges = np.flatnonzero(below[1:] & ~below[:-1]) + 1
    return below, edges


def _attempt(v: np.ndarray, fs: float, width: int, max_lines: int):
    """Одна спроба сліпого декодування за заданої полярності.

    Повертає (оцінка_якості, Frame|None, t0|None). Оцінка — частка
    міжсинхронних інтервалів, що лягли в ±20% від медіани. У шумі або
    при перевернутому сигналі вона розсипається, тому за нею й обираємо
    полярність. `t0` — локальний індекс використаного фронту кадрової
    синхри, потрібен викликачу для того, щоб засіяти DecodeState.
    """
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi - lo < 1e-9:
        return 0.0, None, None
    v = (v - lo) / (hi - lo)          # вершина синхри ≈ 0, білий ≈ 1

    thr = 0.18                        # між вершиною синхри і рівнем гасіння
    below, edges = _sync_edges(v, thr)
    if len(edges) < 24:
        return 0.0, None, None

    d = np.diff(edges).astype(np.float64)
    # орієнтир — найдовший поширений інтервал (півкадрові імпульси коротші)
    med = float(np.median(d[d > np.percentile(d, 40)]))
    keep = (d > med * 0.8) & (d < med * 1.2)
    score = float(keep.mean())
    if keep.sum() < 12:
        return 0.0, None, None
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
        return 0.0, None, None
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
        return 0.0, None, None

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
        standard=standard, locked=locked), t0


def _predict_local_t0(state: DecodeState, abs_start: float) -> float | None:
    """Прогнозує локальну (відносно початку нового знімку) позицію
    найближчого фронту кадрової синхри за відомими період+abs_t0.

    Екстраполяція йде по цілій кількості польових періодів
    (`round(...)`), тому прогноз завжди влучає в межі поточного поля
    незалежно від того, наскільки великий розрив між знімками —
    буфер міг не встигнути записати частину відліків, виклик
    decode() міг прийти з затримкою тощо. «Пливти» разом з розміром
    розриву тут нема чому.
    """
    if state.period is None or state.abs_t0 is None:
        return None
    field_period = state.period * FIELD_LINES.get(state.standard, FIELD_LINES["?"])
    if field_period <= 0:
        return None
    n = round((abs_start - state.abs_t0) / field_period)
    abs_pred = state.abs_t0 + n * field_period
    return float(abs_pred - abs_start)


def _attempt_tracked(v: np.ndarray, fs: float, width: int, max_lines: int,
                      state: DecodeState, abs_start: float,
                      tol_frac: float = 0.55):
    """Швидка спроба декодування зі знанням періоду й полярності.

    Шукає фронт кадрової синхри лише у вузькому вікні
    (±tol_frac·period) навколо прогнозованої позиції, а сам період і
    полярність не переоцінює — бере як є з `state`. Це не лише
    швидше за повний сліпий пошук, а й головна причина стабільнішої
    картинки: період більше не «сіпається» від незалежних оцінок
    кожного блоку.

    Повертає (Frame, abs_t0) або None, якщо синхру у вікні не
    знайдено — тоді викликач має відкотитись на _attempt().
    """
    local_t0_pred = _predict_local_t0(state, abs_start)
    if local_t0_pred is None:
        return None
    period = state.period
    tol = tol_frac * period
    lo = int(max(0, local_t0_pred - tol))
    hi = int(min(len(v), local_t0_pred + tol))
    if hi - lo < 8:
        return None

    vv = v * state.sign
    p_lo, p_hi = np.percentile(vv, [0.5, 99.5])
    if p_hi - p_lo < 1e-9:
        return None
    vv = (vv - p_lo) / (p_hi - p_lo)

    thr = 0.18
    window = vv[lo:hi]
    below_w = window < thr
    edges_w = np.flatnonzero(below_w[1:] & ~below_w[:-1]) + 1 + lo
    if len(edges_w) == 0:
        return None
    t0 = float(edges_w[np.argmin(np.abs(edges_w - local_t0_pred))])

    standard = state.standard
    a0_frac, a1_frac, _vblank = STD_GEOM[standard]
    a0 = a0_frac * period
    a1 = a1_frac * period
    n_lines = int(min(max_lines, (len(vv) - t0 - period) / period))
    if a1 <= a0 or n_lines < 32:
        return None

    offs = np.linspace(a0, a1, width, dtype=np.float32)
    idx = (t0 + np.arange(n_lines, dtype=np.float32)[:, None] * period
           + offs[None, :])
    idx = np.clip(idx, 0, len(vv) - 2)
    i0 = idx.astype(np.int32)
    fr = idx - i0
    samp = vv[i0] * (1 - fr) + vv[i0 + 1] * fr

    luma = np.clip((samp - 0.30) / 0.70, 0, 1)
    frame = Frame(luma=(luma * 255).astype(np.uint8),
                  line_rate=fs / period, lines=n_lines,
                  standard=standard, locked=True)
    return frame, abs_start + t0


def decode(base: np.ndarray, fs: float, width: int = 640,
           max_lines: int = 288,
           state: DecodeState | None = None,
           abs_start: float = 0.0) -> Frame | None:
    """Декодує напівкадр.

    Без `state` (або на першому виклику) — точнісінько як раніше:
    повний перебір полярності й сліпий пошук синхри. Якщо переданий
    `state` уже містить період з попереднього успішного кадру, спершу
    пробує трекінг у вузькому вікні; лише після трьох поспіль
    невдалих спроб трекінгу («lost») відкочується на сліпий пошук і
    засіює `state` заново.
    """
    if len(base) < int(fs * 0.02):        # менше 20 мс — нема сенсу
        return None
    v = base.astype(np.float32)

    if state is not None and state.period is not None and state.lost < 3:
        tracked = _attempt_tracked(v, fs, width, max_lines, state, abs_start)
        if tracked is not None:
            frame, abs_t0 = tracked
            state.abs_t0 = abs_t0
            state.lost = 0
            return frame
        state.lost += 1

    best_s, best_f, best_t0, best_sign = 0.0, None, None, 1.0
    for sign in (1.0, -1.0):
        s, f, t0 = _attempt(v * sign, fs, width, max_lines)
        if f is not None and s > best_s:
            best_s, best_f, best_t0, best_sign = s, f, t0, sign
    if best_s <= 0.5:
        return None

    if state is not None:
        state.sign = best_sign
        state.period = fs / best_f.line_rate
        state.standard = best_f.standard
        state.abs_t0 = abs_start + best_t0
        state.lost = 0
    return best_f


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