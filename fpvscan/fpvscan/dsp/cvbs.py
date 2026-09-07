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
from io import BytesIO

import numpy as np
from PIL import Image

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
    field_parity: int | None = None  # 0/1 — яке поле; щоб не плести сусідні


@dataclass
class PictureScore:
    """Спільна оцінка «чи це справжня картинка».

    Один рахунок і для димової перевірки INSPECT, і для пошуку
    частоти в LOCK — щоб не розмножувати критерії.
    """
    value: float
    locked: bool
    lines: int
    row_corr: float

    def is_analog(self, min_corr: float = 0.25, require_lock: bool = False,
                  min_lines: int = 80) -> bool:
        """Справжнє CVBS: рядки + схожість рядків (+ кадрова, якщо треба).

        Сама кадрова синхра в шумі спрацьовує часто — «залочене» поле
        зі снігу не має потрапляти в список. Кореляція рядків це і
        відсікає: шум ~0, живе відео помітно додатне.
        """
        if self.lines < min_lines:
            return False
        if require_lock and not self.locked:
            return False
        return self.row_corr >= min_corr


def row_correlation(luma: np.ndarray, pairs: int = 8) -> float:
    """Середня кореляція сусідніх рядків активної частини кадру.

    Шум дає ~0; аналогове відео — помітно додатну величину, бо
    сусідні рядки майже однакові.
    """
    if luma is None or luma.ndim != 2:
        return 0.0
    h, w = luma.shape
    if h < 16 or w < 16:
        return 0.0
    lo = int(h * 0.12)
    hi = int(h * 0.88)
    if hi - lo < 4:
        return 0.0
    n = min(pairs, hi - lo - 1)
    ys = np.linspace(lo, hi - 2, n, dtype=np.int32)
    x = luma.astype(np.float32)
    acc = 0.0
    n_ok = 0
    for y in ys:
        a = x[y] - x[y].mean()
        b = x[y + 1] - x[y + 1].mean()
        na = float(np.dot(a, a))
        nb = float(np.dot(b, b))
        if na < 1.0 or nb < 1.0:
            continue
        acc += float(np.dot(a, b) / np.sqrt(na * nb))
        n_ok += 1
    return acc / n_ok if n_ok else 0.0


def score_picture(frame: Frame | None) -> PictureScore:
    """Скаляр 0..1: кадрова синхра + рядки + схожість рядків."""
    if frame is None or frame.luma is None or frame.luma.size == 0:
        return PictureScore(0.0, False, 0, 0.0)
    corr = row_correlation(frame.luma)
    corr_n = float(np.clip(corr, 0.0, 1.0))
    lines_n = min(frame.lines / 250.0, 1.0)
    locked_n = 1.0 if frame.locked else 0.0
    # Кореляція рядків важливіша за кадрову: інакше заложений сніг
    # (corr≈0, locked=1, багато рядків) перебивав живе відео в hunt/inspect.
    value = 0.55 * corr_n + 0.25 * locked_n + 0.20 * lines_n
    if corr_n < 0.06:
        value *= 0.45
    return PictureScore(value, bool(frame.locked), int(frame.lines), float(corr))


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

    `level_lo`/`level_hi` — згладжені чорна/біла точки (проти мерехтіння
    яскравості). `target_lines` — стабільна висота растру, щоб n_lines
    не скакав кадр-кадр. `t0_err` — згладжена похибка старту поля.
    """
    sign: float = 1.0
    period: float | None = None      # період рядка, у відліках поточної fs
    standard: str = "?"
    abs_t0: float | None = None
    lost: int = 0                    # підряд невдалих спроб трекінгу
    level_lo: float | None = None
    level_hi: float | None = None
    target_lines: int | None = None
    t0_err: float | None = None
    h_roll: int | None = None        # зсув розгортки, якщо H-синхра в кадрі


def _sync_edges(v: np.ndarray, thr: float):
    below = v < thr
    edges = np.flatnonzero(below[1:] & ~below[:-1]) + 1
    return below, edges


def _genlock_starts(v: np.ndarray, t0: float, period: float, n_lines: int,
                    thr: float, max_corr_frac: float = 0.12) -> np.ndarray:
    """Порядковий генлок (часовий коректор бази, TBC).

    Замість того, щоб брати старт кожного рядка як `t0 + i*period` за
    єдиним глобальним періодом, знаходимо фактичний передній фронт
    рядкової синхри поряд із прогнозом і рівняємо рядок по ньому. Це
    прибирає дві найпомітніші вади: **нахил вертикалей** (навіть частка
    відлічку похибки періоду накопичується у зсув за 288 рядків) і
    **розрив кадру** по діагоналі. Корекція обмежена вузьким вікном
    (±max_corr_frac·period), тож завади й зрівнювальні імпульси кадрового
    гасіння не здатні «перекинути» рядок на сусідній період; де фронт не
    знайдено — лишаємо прогноз.
    """
    w = max(2, int(max_corr_frac * period))
    starts = t0 + np.arange(n_lines, dtype=np.float64) * period
    base = np.floor(starts).astype(np.int64)
    fracpos = starts - base
    rel = np.arange(-w, w + 2, dtype=np.int64)
    wi = base[:, None] + rel[None, :]
    np.clip(wi, 0, len(v) - 1, out=wi)
    seg = v[wi]                                   # (n_lines, len(rel))
    below = seg < thr
    fall = below[:, 1:] & ~below[:, :-1]          # передній фронт синхри
    # субвідлікова позиція перетину порогу для кожного потенційного фронту
    a = seg[:, :-1]
    b = seg[:, 1:]
    denom = a - b
    safe = np.abs(denom) > 1e-6
    cross_frac = np.where(safe, (a - thr) / np.where(safe, denom, 1.0), 0.0)
    edge_pos = rel[:-1].astype(np.float64)[None, :] + np.clip(cross_frac, 0.0, 1.0)
    big = 1e9
    dist = np.where(fall, np.abs(edge_pos - fracpos[:, None]), big)
    j = np.argmin(dist, axis=1)
    rows = np.arange(n_lines)
    found = dist[rows, j] < big
    corr = base + edge_pos[rows, j]
    corr = np.clip(corr, starts - w, starts + w)
    # Не беремо повний стрибок фронту: 65% виміру + 35% прогнозу гасить
    # поодинокі хибні фронти (зрівнювальні імпульси, шум), які інакше
    # рвуть вертикалі. Де фронту немає — лишаємо прогноз.
    blended = 0.65 * corr + 0.35 * starts
    return np.where(found, blended, starts)


TBC_SEARCH_FRAC = 0.52
TBC_TRACK_FRAC = 0.15
TBC_DELTA_CLAMP = 1.5
TBC_SMOOTH_EMA = 0.35
TBC_WEAK_FOUND = 0.70
TBC_FOOTER_HOLD = 8
H_PHASE_DEADBAND = 0.01
H_PORCH_FRAC = 0.08
H_PORCH_OK = 0.15
CROP_LEFT_FRAC = 0.07
CROP_BOTTOM_LINES = 6


def _tbc_line_edges(v: np.ndarray, nom_starts: np.ndarray,
                    period: float, thr: float,
                    search_frac: float = TBC_SEARCH_FRAC,
                    return_found: bool = False):
    """Per-line 1D H-sync leading edge around the period-grid estimate.

    `search_frac` is ±52% (blind / lost / weak) or ±15% (tracked lock).
    Score = falling drop × stay-low over the sync pulse.
    """
    n_lines = int(len(nom_starts))
    if n_lines < 1 or period < 8.0 or len(v) < 8:
        out = np.asarray(nom_starts, dtype=np.float64)
        if return_found:
            return out, np.zeros(out.shape, dtype=bool)
        return out
    w = max(4, int(search_frac * period))
    sync_w = max(2, int((SYNC_US / 64e-6) * period))
    base = np.floor(nom_starts).astype(np.int64)
    rel = np.arange(-w, w + 2, dtype=np.int64)
    wi = base[:, None] + rel[None, :]
    np.clip(wi, 0, len(v) - 1, out=wi)
    seg = v[wi]
    below = seg < thr
    fall = below[:, 1:] & ~below[:, :-1]
    drop = seg[:, :-1] - seg[:, 1:]
    padded = np.pad(seg, ((0, 0), (0, sync_w)), mode="edge")
    csum = np.cumsum(padded, axis=1)
    k0 = np.arange(fall.shape[1])
    stay = (csum[:, k0 + sync_w] - csum[:, k0]) / float(sync_w)
    stay_low = np.clip((thr + 0.08) - stay, 0.0, 1.5)
    score = np.where(fall, drop * (0.35 + stay_low), -1.0e9)
    rows = np.arange(n_lines)
    j = np.argmax(score, axis=1)
    found = score[rows, j] > 0.0
    a = seg[rows, j]
    b = seg[rows, j + 1]
    denom = a - b
    safe = np.abs(denom) > 1e-6
    frac = np.where(safe, (a - thr) / np.where(safe, denom, 1.0), 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    edges = base.astype(np.float64) + rel[j].astype(np.float64) + frac
    out = np.where(found, edges, nom_starts.astype(np.float64))
    if return_found:
        return out, found
    return out


def _tbc_smooth_edges(edges: np.ndarray, period: float,
                      max_delta: float = TBC_DELTA_CLAMP,
                      ema: float = TBC_SMOOTH_EMA) -> np.ndarray:
    """median-3 + 1-tap EMA + Δ clamp on per-line residuals.

    Tracked lock already searches ±15%. Does not roll the raster.
    """
    e = np.asarray(edges, dtype=np.float64)
    n = int(e.size)
    if n < 3 or period < 8.0:
        return e
    idx = np.arange(n, dtype=np.float64)
    base = float(np.median(e - idx * period))
    grid = base + idx * period
    r = e - grid
    prev = np.empty(n, dtype=np.float64)
    nxt = np.empty(n, dtype=np.float64)
    prev[0] = r[0]
    prev[1:] = r[:-1]
    nxt[-1] = r[-1]
    nxt[:-1] = r[1:]
    r = np.median(np.stack((prev, r, nxt), axis=0), axis=0)
    out = np.empty(n, dtype=np.float64)
    out[0] = r[0]
    a = float(np.clip(ema, 0.05, 0.95))
    lim = float(max_delta)
    for i in range(1, n):
        pred = (1.0 - a) * out[i - 1] + a * r[i]
        d = pred - out[i - 1]
        if d > lim:
            pred = out[i - 1] + lim
        elif d < -lim:
            pred = out[i - 1] - lim
        out[i] = pred
    return grid + out


def _tbc_hold_footer(edges: np.ndarray, period: float,
                     hold: int = TBC_FOOTER_HOLD) -> np.ndarray:
    """Last N lines copy the line above the footer (VBI must not hunt)."""
    e = np.asarray(edges, dtype=np.float64).copy()
    n = int(e.size)
    k = min(int(hold), max(0, n // 5))
    if k < 2 or n <= k + 2 or period < 8.0:
        return e
    src = n - k - 1
    off = np.arange(1, k + 1, dtype=np.float64)
    e[-k:] = e[src] + off * period
    return e


def tbc_search_frac(*, locked: bool, found_frac: float | None = None) -> float:
    """Wide ±52% on blind/lost/weak edges; narrow ±15% once lock holds."""
    if locked and (found_frac is None or found_frac >= TBC_WEAK_FOUND):
        return TBC_TRACK_FRAC
    return TBC_SEARCH_FRAC


def h_blank_col(luma: np.ndarray) -> int:
    """Column of the darkest vertical strip (H-blank / sync), or 0 if none.

    Uses a wide column average so a noisy pit still wins over mid-frame
    content. A pit already on the left porch is treated as parked (0).
    """
    if luma is None or luma.ndim != 2:
        return 0
    h, w = luma.shape
    if w < 48 or h < 24:
        return 0
    col = luma.astype(np.float32).mean(axis=0)
    ksz = max(9, (w // 32) | 1)
    ker = np.ones(ksz, dtype=np.float32) / ksz
    sm = np.convolve(col, ker, mode="same")
    j = int(np.argmin(sm))
    med = float(np.median(sm))
    pit = float(sm[j])
    if med < 1.0:
        return 0
    if pit > med * 0.92:
        return 0
    if h_blank_parked(j, w):
        return 0
    return j


def _h_sync_edge_raw(luma: np.ndarray) -> int | None:
    """Column of the 2D H-sync leading edge, or None if none."""
    if luma is None or luma.ndim != 2:
        return None
    h, w = luma.shape
    if w < 48 or h < 24:
        return None
    x = luma.astype(np.float32)
    fall = x[:, :-1] - x[:, 1:]
    span = float(np.median(x.max(axis=1) - x.min(axis=1)))
    thr = max(12.0, 0.15 * span)
    votes = (fall >= thr).sum(axis=0)
    j = int(np.argmax(votes))
    if int(votes[j]) < max(8, int(0.70 * h)):
        return None
    return j


def h_sync_edge_col(luma: np.ndarray) -> int:
    """H-sync leading edge: falling step that is on nearly every line.

    Median across rows of horizontal fall energy. A dark wardrobe is a
    pit on some rows and loses to a sync edge present on all lines.
    """
    raw = _h_sync_edge_raw(luma)
    return 0 if raw is None else raw


def h_phase_col(luma: np.ndarray) -> int:
    """Phase estimate only: sync edge first, darkest column as fallback.

    A parked edge is 0 (already on the left porch) — do not fall through
    to a mid-frame wardrobe pit.
    """
    raw = _h_sync_edge_raw(luma)
    if raw is None:
        return h_blank_col(luma)
    w = int(luma.shape[1])
    return 0 if h_blank_parked(raw, w) else raw


def h_blank_parked(col: int, width: int) -> bool:
    return 0 <= int(col) <= int(width * H_PORCH_OK)


def h_pit_in_mid_half(col: int, width: int) -> bool:
    return int(width * 0.25) <= int(col) < int(width * 0.75)


def h_pit_unsafe(col: int, width: int) -> bool:
    """Blank in the middle 50% or wrapped in from the right edge."""
    c = int(col) % max(1, int(width))
    return h_pit_in_mid_half(c, width) or c >= int(width * 0.75)


def h_auto_roll(pit: int, width: int) -> int:
    """Pixels to roll left so the pit sits on the left porch. 0 if unsafe."""
    if pit <= 0 or h_blank_parked(pit, width):
        return 0
    target = int(width * H_PORCH_FRAC)
    roll = (int(pit) - target) % width
    dest = (int(pit) - roll) % width
    if h_pit_unsafe(dest, width):
        return 0
    return roll


def h_phase_manual_px(frac: float, width: int) -> int:
    """Extra roll after auto park. |frac| < 0.01 is slider noise → 0.

    Sign is inverted vs the 13:00 clip (`frac=-0.06` used to dump blank
    into active video). Positive frac shifts the picture left (more of
    the right side comes in). Negative shifts the picture right (more
    left porch) but the caller rejects a dest in the middle 50%.
    """
    if abs(float(frac)) < H_PHASE_DEADBAND:
        return 0
    return int(round(-float(frac) * width))


def _h_unwrap(luma: np.ndarray, state: DecodeState | None,
              h_phase_frac: float = 0.0) -> np.ndarray:
    """Park H-blank on the left porch, then a deadbanded manual nudge.

    Phase only — crop is a later slice, never a roll into the middle.
    A large h_phase_frac that would land mid-frame is ignored (auto
    park stays); we do not pretend the slider applied.
    """
    if luma is None or luma.ndim != 2:
        return luma
    w = luma.shape[1]
    pit = h_phase_col(luma)
    auto = h_auto_roll(pit, w)
    if state is not None:
        if state.h_roll is None:
            state.h_roll = auto
        else:
            prev = int(state.h_roll)
            d = (auto - prev + w // 2) % w - w // 2
            state.h_roll = int(prev + 0.75 * d) % w
        auto = int(state.h_roll)
        dest = (pit - auto) % w
        if auto and h_pit_unsafe(dest, w):
            auto = 0
            state.h_roll = 0
    manual = h_phase_manual_px(h_phase_frac, w)
    shift = (auto + manual) % w
    dest = (pit - shift) % w
    if shift and h_pit_unsafe(dest, w):
        shift = auto
        dest = (pit - shift) % w
        if shift and h_pit_unsafe(dest, w):
            return luma
    if not shift:
        return luma
    return np.roll(luma, -shift, axis=1)


def _h_crop(luma: np.ndarray,
            left_frac: float = CROP_LEFT_FRAC,
            bottom_lines: int = CROP_BOTTOM_LINES) -> np.ndarray:
    """Slice porch/sync (left) and field footer (bottom) after park.

    Crop only — never roll bars into the middle of the picture.
    """
    if luma is None or luma.ndim != 2:
        return luma
    h, w = luma.shape
    lf = min(max(float(left_frac), 0.0), 0.25)
    bl = min(max(int(bottom_lines), 0), 24)
    left = int(round(w * lf))
    left = min(left, max(0, w - 48))
    bottom = min(bl, max(0, h - 24))
    out = luma[:, left:] if left else luma
    if bottom:
        out = out[:-bottom]
    return out


def _fit_height(luma: np.ndarray, target: int) -> np.ndarray:
    """Підганяє растр до стабільної висоти: обрізає зверху або дописує
    останнім рядком (менше мерехтить, ніж чорна смуга)."""
    h, w = luma.shape
    if h == target:
        return luma
    if h > target:
        return luma[:target]
    out = np.empty((target, w), dtype=luma.dtype)
    out[:h] = luma
    # Do not repeat the VBI/footer row — that twitches when stretched to 288.
    src = luma[max(0, h - TBC_FOOTER_HOLD - 1)]
    out[h:] = src
    return out


def _render(v: np.ndarray, starts: np.ndarray, period: float,
            a0_frac: float, a1_frac: float, width: int,
            auto_levels: bool = True, sharpen: float = 0.0,
            state: DecodeState | None = None,
            h_phase_frac: float = 0.0,
            thr: float = 0.18,
            tbc_locked: bool = False) -> np.ndarray:
    """TBC: slice PAL/NTSC active window from each line's H-sync edge.

    `starts` are the period-grid / genlock estimate. Per-line TBC finds
    the 1D leading edge, then copies a0…a1 from that edge so H-blank
    never enters luma. Tracked lock uses a narrow search; blind / lost
    / weak edges keep ±52%. Edges are median-3 + EMA, Δ clamped.
    `h_phase_frac` is a residual offset after TBC (0 / |x|<0.01 = none).
    2D roll is not used to hide blank.
    """
    frac = tbc_search_frac(locked=tbc_locked)
    edges, found = _tbc_line_edges(
        v, starts, period, thr, search_frac=frac, return_found=True)
    if tbc_locked and float(found.mean()) < TBC_WEAK_FOUND:
        edges, found = _tbc_line_edges(
            v, starts, period, thr, search_frac=TBC_SEARCH_FRAC,
            return_found=True)
    edges = _tbc_smooth_edges(edges, period)
    edges = _tbc_hold_footer(edges, period)
    phase = 0.0 if abs(float(h_phase_frac)) < H_PHASE_DEADBAND else float(h_phase_frac)
    a0_f = float(np.clip(a0_frac + phase, 8.0 / 64.0, 22.0 / 64.0))
    span = float(a1_frac - a0_frac)
    a1_f = float(min(a0_f + span, 63.0 / 64.0))
    if a1_f <= a0_f + 4.0 / 64.0:
        a0_f, a1_f = float(a0_frac), float(a1_frac)
    a0 = a0_f * period
    a1 = a1_f * period
    offs = np.linspace(a0, a1, width, dtype=np.float32)
    idx = edges[:, None].astype(np.float32) + offs[None, :]
    np.clip(idx, 0, len(v) - 2, out=idx)
    i0 = idx.astype(np.int32)
    fr = idx - i0
    samp = v[i0] * (1 - fr) + v[i0 + 1] * fr       # (n_lines, width)

    if auto_levels:
        # Робастні чорна/біла точки за перцентилями активного поля.
        flat = samp[::4, ::4].ravel()              # грубше прорідження — дешевше
        lo_m, hi_m = np.percentile(flat, [2.0, 99.0])
        if hi_m - lo_m < 1e-3:
            lo_m, hi_m = 0.30, 1.0
        # EMA між кадрами: різкий стрибок перцентиля більше не блимає
        # яскравістю всього кадру.
        if state is not None and state.level_lo is not None:
            a = 0.18
            lo = (1.0 - a) * state.level_lo + a * float(lo_m)
            hi = (1.0 - a) * state.level_hi + a * float(hi_m)
        else:
            lo, hi = float(lo_m), float(hi_m)
        if state is not None:
            state.level_lo = lo
            state.level_hi = hi
    else:
        lo, hi = 0.30, 1.0
    luma = np.clip((samp - lo) / (hi - lo), 0.0, 1.0)

    if sharpen > 0.0:
        # Нерізке маскування лише по горизонталі (аналогове відео втрачає
        # саме горизонтальну роздільність). Ядро 1-2-1 як дешевий ФНЧ.
        blur = luma.copy()
        blur[:, 1:-1] = 0.25 * luma[:, :-2] + 0.5 * luma[:, 1:-1] + 0.25 * luma[:, 2:]
        luma = np.clip(luma + sharpen * (luma - blur), 0.0, 1.0)

    return (luma * 255).astype(np.uint8)


def _attempt(v: np.ndarray, fs: float, width: int, max_lines: int,
             auto_levels: bool = True, sharpen: float = 0.0,
             state: DecodeState | None = None,
             h_phase_frac: float = 0.0):
    """Одна спроба сліпого декодування за заданої полярності.

    Повертає (оцінка_якості, Frame|None, t0|None). Оцінка — частка
    міжсинхронних інтервалів, що лягли в ±20% від медіани. У шумі або
    при перевернутому сигналі вона розсипається, тому за нею й обираємо
    полярність. `t0` — локальний індекс використаного фронту кадрової
    синхри, потрібен викликачу для того, щоб засіяти DecodeState.
    """
    lo, hi = np.percentile(v[::8], [0.5, 99.5])
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
    next_vs_start = None
    if locked:
        brk = np.flatnonzero(np.diff(vs) > win)
        # межі кожної групи широких імпульсів (кожна = одне кадрове гасіння)
        group_starts = np.concatenate(([vs[0]], vs[brk + 1])) if len(brk) else vs[:1]
        group_ends = np.concatenate((vs[brk], vs[-1:])) if len(brk) else vs[-1:]
        end_vs = group_ends[0]
        start = int(end_vs + win * vblank)     # пропускаємо кадрове гасіння
        # Початок наступного кадрового гасіння: далі за нього заходити не
        # можна, інакше в кадр потрапляє синхра сусіднього поля і картинка
        # «рветься» по діагоналі. Саме це й давало розрив у записах, де
        # захоплення починається посеред поля.
        later = group_starts[group_starts > start + win]
        if len(later):
            next_vs_start = int(later[0])
    else:
        start = int(edges[0])

    nxt = edges[edges >= start]
    if len(nxt) == 0:
        nxt = edges
    t0 = float(nxt[0])

    a0 = a0_frac * period
    a1 = a1_frac * period
    avail = (len(v) - t0 - period) / period
    if next_vs_start is not None:
        avail = min(avail, (next_vs_start - t0) / period - 1.0)
    n_lines = int(min(max_lines, avail))
    if a1 <= a0 or n_lines < 32:
        return 0.0, None, None

    starts = _genlock_starts(v, float(t0), period, n_lines, thr)
    luma = _render(v, starts, period, a0_frac, a1_frac, width,
                   auto_levels=auto_levels, sharpen=sharpen, state=state,
                   h_phase_frac=h_phase_frac, thr=thr, tbc_locked=False)
    if state is not None:
        if state.target_lines is None:
            state.target_lines = max_lines
        luma = _fit_height(luma, state.target_lines)
    return score, Frame(
        luma=luma,
        line_rate=line_rate, lines=n_lines,
        standard=standard, locked=locked), t0


def _predict_local_t0(state: DecodeState, abs_start: float) :
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
    return float(abs_pred - abs_start), n, field_period


def _attempt_tracked(v: np.ndarray, fs: float, width: int, max_lines: int,
                    state: DecodeState, abs_start: float,
                    tol_frac: float = 0.55,
                    auto_levels: bool = True, sharpen: float = 0.0,
                    h_phase_frac: float = 0.0, h_pll: bool = False):
    """Швидка спроба декодування зі знанням періоду й полярності.

    Шукає фронт кадрової синхри лише у вузькому вікні (±tol_frac·period)
    навколо прогнозованої позиції. Якщо `h_pll`, повільно уточнює
    state.period (вузькосмугова ФАПЧ за фазою поля). TBC ±15% завжди.

    Повертає (Frame, abs_t0) або None.
    """
    pred = _predict_local_t0(state, abs_start)
    if pred is None:
        return None
    local_t0_pred, n_fields, field_period = pred
    period = state.period
    tol = tol_frac * period
    lo = int(max(0, local_t0_pred - tol))
    hi = int(min(len(v), local_t0_pred + tol))
    if hi - lo < 8:
        return None

    vv = v * state.sign
    p_lo, p_hi = np.percentile(vv[::8], [0.5, 99.5])
    if p_hi - p_lo < 1e-9:
        return None
    vv = (vv - p_lo) / (p_hi - p_lo)

    thr = 0.18
    window = vv[lo:hi]
    below_w = window < thr
    edges_w = np.flatnonzero(below_w[1:] & ~below_w[:-1]) + 1 + lo
    if len(edges_w) == 0:
        return None
    t0_raw = float(edges_w[np.argmin(np.abs(edges_w - local_t0_pred))])
    # Згладжуємо старт поля: один шумний фронт більше не підкидає
    # увесь кадр по вертикалі. Обрізаємо викиди і мішаємо з прогнозом.
    err = t0_raw - local_t0_pred
    max_err = 0.22 * period
    err = float(np.clip(err, -max_err, max_err))
    if state.t0_err is None:
        state.t0_err = err
    else:
        state.t0_err = 0.35 * err + 0.65 * state.t0_err
    t0 = local_t0_pred + state.t0_err

    standard = state.standard
    a0_frac, a1_frac, _vblank = STD_GEOM[standard]
    a0 = a0_frac * period
    a1 = a1_frac * period
    avail = (len(vv) - t0 - period) / period
    # Той самий захист від розриву, що й у сліпому шляху: не заходити за
    # наступне кадрове гасіння. Шукаємо його лише в тій частині буфера,
    # яку збираємось рендерити (обмежений cumsum — дешево).
    win = int(period)
    lo_s = int(t0 + 5 * period)
    hi_s = int(min(len(vv), t0 + (max_lines + 6) * period))
    if hi_s - lo_s > 2 * win:
        seg_below = (vv[lo_s:hi_s] < thr).astype(np.int32)
        csum = np.cumsum(np.concatenate(([0], seg_below)))
        vfrac = (csum[win:] - csum[:-win]) / win
        vsloc = np.flatnonzero(vfrac > 0.55)
        if len(vsloc):
            avail = min(avail, (lo_s + int(vsloc[0]) - t0) / period - 1.0)
    n_lines = int(min(max_lines, avail))
    if a1 <= a0 or n_lines < 32:
        return None

    starts = _genlock_starts(vv, float(t0), period, n_lines, thr)
    luma = _render(vv, starts, period, a0_frac, a1_frac, width,
                   auto_levels=auto_levels, sharpen=sharpen, state=state,
                   h_phase_frac=h_phase_frac, thr=thr, tbc_locked=True)
    if state.target_lines is None:
        state.target_lines = max_lines
    luma = _fit_height(luma, state.target_lines)
    frame = Frame(luma=luma,
                line_rate=fs / period, lines=n_lines,
                standard=standard, locked=True,
                field_parity=int(n_fields) % 2)

    # Повільне уточнення періоду: різниця між прогнозованою і фактично
    # знайденою позицією, поділена на кількість польових періодів, що
    # минули з останнього надійного вимірювання, — пряма оцінка того,
    # наскільки поточний period відхилився від реального (тепловий
    # дрейф вільнонесучого генератора VTx). Береться лише малою часткою
    # (alpha), щоб один зашумлений кадр не хитав період так само різко,
    # як повний сліпий перерахунок; і тільки коли n_fields достатньо
    # велике, інакше похибка вимірювання самого t0 (одиниці відліків)
    # після ділення на малий n_fields дає нестабільно завищену поправку.
    if n_fields >= 4 and h_pll:
        _pll_nudge_period(state, period, t0, local_t0_pred, n_fields, standard)

    return frame, abs_start + t0


def _pll_nudge_period(state: DecodeState, period: float, t0: float,
                      local_t0_pred: float, n_fields: int,
                      standard: str) -> None:
    """Slow H-line period PLL. Not a sample-by-sample loop."""
    phase_err = t0 - local_t0_pred
    period_err_per_line = (phase_err / n_fields) / FIELD_LINES.get(
        standard, FIELD_LINES["?"])
    alpha = 0.05
    new_period = period + alpha * period_err_per_line
    if 0.5 * period < new_period < 1.5 * period:
        state.period = new_period


def decode(base: np.ndarray, fs: float, width: int = 640,
        max_lines: int = 288,
        state: DecodeState | None = None,
        abs_start: float = 0.0,
        auto_levels: bool = True,
        sharpen: float = 0.0,
        h_phase_frac: float = 0.0,
        crop_left_frac: float = CROP_LEFT_FRAC,
        crop_bottom_lines: int = CROP_BOTTOM_LINES,
        h_pll: bool = False) -> Frame | None:
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
        tracked = None
        for tol_frac in (0.55, 0.75, 0.95):
            tracked = _attempt_tracked(v, fs, width, max_lines, state, abs_start,
                                       tol_frac, auto_levels=auto_levels,
                                       sharpen=sharpen,
                                       h_phase_frac=h_phase_frac,
                                       h_pll=h_pll)
            if tracked is not None:
                break
        if tracked is not None:
            frame, abs_t0 = tracked
            state.abs_t0 = abs_t0
            state.lost = 0
            frame.luma = _h_crop(frame.luma, crop_left_frac, crop_bottom_lines)
            return frame
        state.lost += 1

    # Відому полярність пробуємо першою: на зриві трекінгу це часто
    # одразу дає поле і не ганяє другий повний сліпий прохід.
    signs = (1.0, -1.0)
    known = state is not None and state.period is not None
    if known:
        signs = (state.sign, -state.sign)
    best_s, best_f, best_t0, best_sign = 0.0, None, None, 1.0
    for sign in signs:
        s, f, t0 = _attempt(v * sign, fs, width, max_lines,
                            auto_levels=auto_levels, sharpen=sharpen,
                            h_phase_frac=h_phase_frac)
        if f is not None and s > best_s:
            best_s, best_f, best_t0, best_sign = s, f, t0, sign
            if known and best_s > 0.85:
                break
    if best_s <= 0.5:
        return None

    if state is not None:
        state.sign = best_sign
        state.period = fs / best_f.line_rate
        state.standard = best_f.standard
        state.abs_t0 = abs_start + best_t0
        state.lost = 0
        state.t0_err = None
        state.h_roll = None
        if state.target_lines is None:
            state.target_lines = max_lines
        best_f.luma = _fit_height(best_f.luma, state.target_lines)
    best_f.luma = _h_crop(best_f.luma, crop_left_frac, crop_bottom_lines)
    return best_f


def encode(frame: Frame, fmt: str = "webp", quality: int = 80,
           height: int | None = 480, method: int = 1) -> bytes:
    """Кадр у стиснений формат.

    WebP на сірій картинці дає приблизно вчетверо менший файл, ніж
    JPEG тієї ж візуальної якості, і підтримується всіма браузерами.
    Саме він іде і в веб-консоль, і в знімки.

    `method` — зусилля енкодера WebP (0 найшвидший, 6 найякісніший).
    На потоці 4 було ~36 мс/кадр; 0–1 знімає більшу частину цього.
    """
    luma = np.ascontiguousarray(frame.luma)
    if luma.dtype != np.uint8:
        luma = np.asarray(luma, dtype=np.uint8)
    img = Image.frombuffer("L", (int(luma.shape[1]), int(luma.shape[0])),
                           luma, "raw", "L", 0, 1)
    if height:
        img = img.resize((int(luma.shape[1]), int(height)), Image.BILINEAR)
    buf = BytesIO()
    f = fmt.lower()
    if f == "png":
        img.save(buf, "PNG", optimize=True)
    elif f in ("jpeg", "jpg"):
        img.save(buf, "JPEG", quality=quality)
    else:
        img.save(buf, "WEBP", quality=quality, method=int(method))
    return buf.getvalue()


def to_jpeg(frame: Frame, quality: int = 70, height: int = 480) -> bytes:
    return encode(frame, "jpeg", quality, height)