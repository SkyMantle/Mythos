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
        if self.lines < min_lines and not self.locked:
            return False
        if require_lock and not self.locked:
            return False
        if self.row_corr >= min_corr:
            return True
        # слабке, але зібране поле — не ховаємо (типове 3G3 на межі С/Ш)
        return bool(self.locked and self.lines >= min_lines)


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
    value = 0.45 * corr_n + 0.35 * locked_n + 0.20 * lines_n
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


def _sync_edges(v: np.ndarray, thr: float):
    below = v < thr
    edges = np.flatnonzero(below[1:] & ~below[:-1]) + 1
    return below, edges


def _genlock_starts(v: np.ndarray, t0: float, period: float, n_lines: int,
                    thr: float, max_corr_frac: float = 0.045) -> np.ndarray:
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
    out[h:] = luma[-1]
    return out


def _render(v: np.ndarray, starts: np.ndarray, period: float,
            a0_frac: float, a1_frac: float, width: int,
            auto_levels: bool = True, sharpen: float = 0.0,
            state: DecodeState | None = None) -> np.ndarray:
    """Вибирає активну частину рядків у растр + рівні + апертурна корекція.

    `starts` — позиції переднього фронту синхри кожного рядка (уже з
    генлоком). Далі: лінійна інтерполяція активної частини, авторівні
    (розтяг контрасту за перцентилями замість фіксованого відображення —
    прибирає «сірий» недоконтрастний вигляд) і горизонтальна апертурна
    корекція (компенсує завал ВЧ у демодуляторі й децимації — прибирає
    «мило» по горизонталі, робить текст різкішим).
    """
    a0 = a0_frac * period
    a1 = a1_frac * period
    offs = np.linspace(a0, a1, width, dtype=np.float32)
    idx = starts[:, None].astype(np.float32) + offs[None, :]
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
             state: DecodeState | None = None):
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
                   auto_levels=auto_levels, sharpen=sharpen, state=state)
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
                    auto_levels: bool = True, sharpen: float = 0.0):
    """Швидка спроба декодування зі знанням періоду й полярності.

    Шукає фронт кадрової синхри лише у вузькому вікні (±tol_frac·period)
    навколо прогнозованої позиції. При успіху також повільно уточнює
    state.period (вузькосмугова ФАПЧ за фазою) — див. коментар нижче.

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
                   auto_levels=auto_levels, sharpen=sharpen, state=state)
    if state.target_lines is None:
        state.target_lines = max_lines
    luma = _fit_height(luma, state.target_lines)
    frame = Frame(luma=luma,
                line_rate=fs / period, lines=n_lines,
                standard=standard, locked=True)

    # Повільне уточнення періоду: різниця між прогнозованою і фактично
    # знайденою позицією, поділена на кількість польових періодів, що
    # минули з останнього надійного вимірювання, — пряма оцінка того,
    # наскільки поточний period відхилився від реального (тепловий
    # дрейф вільнонесучого генератора VTx). Береться лише малою часткою
    # (alpha), щоб один зашумлений кадр не хитав період так само різко,
    # як повний сліпий перерахунок; і тільки коли n_fields достатньо
    # велике, інакше похибка вимірювання самого t0 (одиниці відліків)
    # після ділення на малий n_fields дає нестабільно завищену поправку.
    if n_fields >= 4:
        phase_err = t0 - local_t0_pred
        period_err_per_line = (phase_err / n_fields) / FIELD_LINES.get(standard, FIELD_LINES["?"])
        alpha = 0.05
        new_period = period + alpha * period_err_per_line
        if 0.5 * period < new_period < 1.5 * period:   # запобіжник від викиду
            state.period = new_period

    return frame, abs_start + t0


def decode(base: np.ndarray, fs: float, width: int = 640,
        max_lines: int = 288,
        state: DecodeState | None = None,
        abs_start: float = 0.0,
        auto_levels: bool = True,
        sharpen: float = 0.0) -> Frame | None:
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
                                       sharpen=sharpen)
            if tracked is not None:
                break
        if tracked is not None:
            frame, abs_t0 = tracked
            state.abs_t0 = abs_t0
            state.lost = 0
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
                            auto_levels=auto_levels, sharpen=sharpen)
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
        if state.target_lines is None:
            state.target_lines = max_lines
        best_f.luma = _fit_height(best_f.luma, state.target_lines)
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
        img.save(buf, "WEBP", quality=quality, method=int(method))
    return buf.getvalue()


def to_jpeg(frame: Frame, quality: int = 70, height: int = 480) -> bytes:
    return encode(frame, "jpeg", quality, height)