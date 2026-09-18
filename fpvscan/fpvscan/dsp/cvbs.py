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
import time

import numpy as np
from PIL import Image

from fpvscan.dsp.demod import standard_from_line_rate

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
ACTIVE_FIELD_LINES = {"PAL": 288, "NTSC": 240, "?": 288}
# One PAL/NTSC field. A 56 ms capture at 15.6 kHz is ~875 lines; shipping
# that (or 2000+) is why LOCK sat at 0.3 fps / ~2.7 s decode.
ONE_FIELD_LINES = 288
# Flat/near-zero luma: sticky predicted t0 painted sync/porch, not picture.
VISIBLE_LUMA_MEAN = 12.0
# Сліпий пошук мусить бути сталим за вартістю: 20 мс містять понад
# 280 аналогових рядків, а після прорідження до ~2 Мвідл/с FFT лишається
# малим навіть для 35 Мвідл/с та довгого LOCK-знімка.
LINE_HUNT_WINDOW_S = 0.020
LINE_HUNT_TARGET_FS = 2_000_000.0
LINE_HUNT_MAX_INPUT_SAMPLES = 65_536
LINE_HUNT_MAX_NFFT = 131_072


@dataclass
class Frame:
    luma: np.ndarray       # uint8, (висота, ширина)
    line_rate: float
    lines: int
    standard: str
    locked: bool
    field_parity: int | None = None  # 0/1 — яке поле; щоб не плести сусідні
    free_run: bool = False           # no H/V sync; snow / skewed analog
    field_t0: float | None = None    # local field-start sample, for sticky t0
    incomplete: bool = False         # geometry is diagnostic, never locked


@dataclass
class LineRateHint:
    """Один лінивий line-rate hunt, спільний для decode і fallback."""

    line_hz: float | None = None
    attempted: bool = False
    confirmed: bool = False
    elapsed_ms: float = 0.0
    input_samples: int = 0
    nfft: int = 0

    @classmethod
    def from_period(cls, period: float | None, fs: float) -> "LineRateHint":
        hz = (float(fs) / float(period)) if period and fs > 0.0 else 0.0
        if ANALOG_LINE_LO_HZ < hz < ANALOG_LINE_HI_HZ:
            return cls(line_hz=hz, attempted=True)
        return cls()

    def accept(self, line_hz: float | None) -> None:
        hz = float(line_hz or 0.0)
        if ANALOG_LINE_LO_HZ < hz < ANALOG_LINE_HI_HZ:
            # A multi-window adaptive comb is stronger than one block's
            # threshold-edge spacing.  Keep that anchor for decode+fallback;
            # an unconfirmed/shared hunt remains free to refine itself.
            if self.confirmed and self.line_hz is not None:
                return
            self.line_hz = hz
            self.attempted = True

    def resolve(self, base: np.ndarray, fs: float) -> float | None:
        if self.attempted:
            return self.line_hz
        self.attempted = True
        t0 = time.perf_counter()
        stats: dict[str, int] = {}
        hz = estimate_line_hz(base, fs, stats=stats)
        self.elapsed_ms += (time.perf_counter() - t0) * 1000.0
        self.input_samples = int(stats.get("input_samples", 0))
        self.nfft = int(stats.get("nfft", 0))
        self.line_hz = hz
        return hz


def cap_field_lines(n: int, max_lines: int = ONE_FIELD_LINES) -> int:
    """Never render more than one field, even if the caller asks for 2k."""
    hi = min(max(1, int(max_lines)), ONE_FIELD_LINES)
    return max(0, min(int(n), hi))


def published_lines(frame: Frame | None) -> int:
    """Footer ``рядків``: sliced field count, not capture length."""
    if frame is None:
        return 0
    n = int(frame.lines or 0)
    luma = getattr(frame, "luma", None)
    if luma is not None and getattr(luma, "ndim", 0) == 2:
        n = min(n, int(luma.shape[0]))
    return cap_field_lines(n)


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

    def is_analog(self, min_corr: float = 0.02, require_lock: bool = False,
                  min_lines: int = 32) -> bool:
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
    # A clearly coherent but incomplete diagnostic raster remains useful
    # picture evidence; reward it without claiming field lock.
    if not frame.locked and corr_n >= 0.50:
        value += 0.12
    if corr_n < 0.06:
        value *= 0.45
    return PictureScore(
        min(1.0, value), bool(frame.locked), int(frame.lines), float(corr),
    )


# Analog FPV often has H-sync / line rate without a clean V-sync pulse
# (OSD cameras). decode() then free-runs and the engine treated that as
# snow — no HTTP lock, no POST /api/snapshot. High row_corr + analog
# line rate is a picture. Snow is ~0.02–0.11; 3290 MHz / 16 MHz IF was 0.65.
ANALOG_LINE_LO_HZ = 14_000.0
ANALOG_LINE_HI_HZ = 17_500.0
USABLE_ROW_CORR = 0.28


def analog_usable(frame: Frame | None, *,
                  min_corr: float = USABLE_ROW_CORR,
                  pic: PictureScore | None = None) -> bool:
    """True when the raster looks like analog video even without V-sync."""
    if frame is None or frame.luma is None or frame.luma.size == 0:
        return False
    hz = float(frame.line_rate or 0.0)
    if not (ANALOG_LINE_LO_HZ < hz < ANALOG_LINE_HI_HZ):
        return False
    if int(frame.lines or 0) < 32:
        return False
    if raster_is_black(frame):
        return False
    if pic is not None:
        corr = float(pic.row_corr)
    else:
        corr = row_correlation(frame.luma)
    return corr >= float(min_corr)


def complete_field_geometry(
    frame: Frame | None,
    *,
    minimum_fraction: float = 0.95,
) -> bool:
    """True only for a nearly complete standard-specific active field."""
    if frame is None or frame.free_run or frame.incomplete:
        return False
    expected = ACTIVE_FIELD_LINES.get(frame.standard, ACTIVE_FIELD_LINES["?"])
    minimum = int(np.ceil(float(expected) * float(minimum_fraction)))
    rows = (
        int(frame.luma.shape[0])
        if frame.luma is not None and getattr(frame.luma, "ndim", 0) == 2 else 0
    )
    return int(frame.lines or 0) >= minimum and rows >= minimum


def field_is_framed(frame: Frame | None, *, slack_lines: int = 2) -> bool:
    """True when the raster is a full PAL/NTSC field, not a 234-line shear.

    ``complete_field_geometry(0.95)`` still accepts NTSC 234/240. AFC must
    not freeze on that wrap: the FM shelf is still off-centre.
    """
    if frame is None or frame.free_run or frame.incomplete:
        return False
    expected = ACTIVE_FIELD_LINES.get(frame.standard, ACTIVE_FIELD_LINES["?"])
    need = max(1, int(expected) - max(0, int(slack_lines)))
    rows = (
        int(frame.luma.shape[0])
        if frame.luma is not None and getattr(frame.luma, "ndim", 0) == 2 else 0
    )
    return int(frame.lines or 0) >= need and rows >= need


def luma_mean(frame: Frame | None) -> float:
    if frame is None or frame.luma is None or frame.luma.size == 0:
        return 0.0
    return float(frame.luma.mean())


def raster_is_black(frame: Frame | None, *,
                    min_mean: float = VISIBLE_LUMA_MEAN) -> bool:
    """True when luma is missing or near-zero (predicted-t0 black field)."""
    if frame is None or frame.luma is None or frame.luma.size == 0:
        return True
    return luma_mean(frame) < float(min_mean)


def choose_lock_raster(tracked: Frame | None,
                       fallback: Frame | None) -> Frame | None:
    """Never ship an all-black tracked field as the only LOCK raster.

    Sticky t0 with no H-edge used to return 288 black lines; engine then
    skipped free_run() because decode() was non-None. Prefer free_run
    (or last-good) over a dead lock; keep a non-black decode if fallback
    is missing.
    """
    if tracked is not None and analog_usable(tracked) and not raster_is_black(tracked):
        return tracked
    if fallback is not None:
        return fallback
    if tracked is not None and not raster_is_black(tracked):
        return tracked
    return None


def field_hold_ok(frame: Frame | None, *, pic: PictureScore | None = None) -> bool:
    """True when this raster is worth holding period + field-start.

    Gate is visible analog (line rate + luma + row corr), not a perfectly
    framed V-blank. Requiring no mid-frame trough refused sheared-but-real
    FPV, so every IQ block re-ran the blind hunt at 0.3 fps. Tracking still
    demands a real H-edge near predicted t0; a black/no-edge hit drops sticky.
    """
    return analog_usable(frame, pic=pic)


# Below this, a sticky field is a half-field / wrap (operator 173 lines),
# not a framed PAL field. Do not use for a well-sized raster with OSD.
SHORT_FIELD_FRAC = 0.72
# Mid/late trough = glued fields or wrap, not a top V-blank or early OSD.
WRAP_TROUGH_FRAC = 0.32


def field_needs_resnap(n_lines: int, max_lines: int = ONE_FIELD_LINES,
                       luma: np.ndarray | None = None,
                       vblank: int = 25) -> bool:
    """True when sticky t0 is mid-picture and must snap to V-blank.

    A framed field (V-blank at top, ~288 lines) must not re-hunt
    pulse/energy every IQ block. A short pack (~173) or a trough in
    the mid/late frame is the torn wrap — resnap. A dark burst in the
    upper third of an already-full field is not.
    """
    hi = cap_field_lines(max_lines)
    if int(n_lines) < int(hi * SHORT_FIELD_FRAC):
        return True
    if luma is None or getattr(luma, "ndim", 0) != 2:
        return False
    j = _v_blank_row(luma, vblank)
    if j is None:
        return False
    h = int(luma.shape[0])
    if h < 32:
        return False
    k = min(max(int(vblank), 8), max(8, h // 5))
    centre = float(j) + k / 2.0
    return centre >= WRAP_TROUGH_FRAC * h


def _extend_short_field_t0(t0: float, n_lines: int, period: float,
                           n_samples: int, max_lines: int) -> tuple[float, int]:
    """Keep H-phase but start earlier so a short pack can reach one field."""
    hi = cap_field_lines(max_lines)
    if int(n_lines) >= int(hi * SHORT_FIELD_FRAC) or period <= 1.0:
        return float(t0), int(n_lines)
    need = float(hi + 1) * float(period)
    end = float(t0) + need
    if end <= float(n_samples) or float(t0) < float(period):
        return float(t0), int(n_lines)
    n_back = int(np.ceil((end - float(n_samples)) / float(period)))
    early = float(t0) - n_back * float(period)
    if early < 0.0:
        return float(t0), int(n_lines)
    avail = (float(n_samples) - early - period) / period
    alt_n = cap_field_lines(int(min(hi, avail)), hi)
    if alt_n > int(n_lines) + 8:
        return early, alt_n
    return float(t0), int(n_lines)


def needs_lock_fallback(frame: Frame | None, *,
                        pic: PictureScore | None = None) -> bool:
    """True when LOCK must still run free_run().

    A non-None black raster is not success — that skip painted the 282-line
    empty pane. Visible analog (analog_usable) is the only skip.
    """
    if frame is None or raster_is_black(frame):
        return True
    return not analog_usable(frame, pic=pic)


def adopt_analog_picture(frame: Frame | None,
                         pic: PictureScore | None = None) -> bool:
    """Promote a high-corr analog raster to lock so snapshot/WS are not snow.

    Mutates ``frame.locked`` / ``free_run`` / ``standard``. Returns True
    when the frame is usable analog (already locked or just adopted).
    """
    if frame is None or not analog_usable(frame, pic=pic):
        return False
    frame.locked = True
    frame.free_run = False
    if str(frame.standard or "").strip() in ("", "?"):
        frame.standard = standard_from_line_rate(
            float(frame.line_rate), max_err_hz=None)
    return True


def should_save_still(frame: Frame | None,
                      pic: PictureScore | None = None) -> bool:
    """POST /api/snapshot: keep locked analog stills; skip snow.

    `free_run` is not the test: adopt_analog_picture already clears it on
    usable analog, and a locked PAL field with high row_corr is a picture
    even if a stale flag lingered. Snow is low corr / no analog line rate.
    """
    if frame is None or frame.luma is None or frame.luma.size == 0:
        return False
    if analog_usable(frame, pic=pic):
        return True
    if not bool(frame.locked):
        return False
    hz = float(frame.line_rate or 0.0)
    if not (ANALOG_LINE_LO_HZ < hz < ANALOG_LINE_HI_HZ):
        return False
    if int(frame.lines or 0) < 32:
        return False
    corr = float(pic.row_corr) if pic is not None else row_correlation(frame.luma)
    return corr >= 0.12


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
    h_edge_hist: tuple[int, ...] = ()
    sticky: bool = False             # hold field t0 after a framed analog hit
    coarse_hint_hz: float | None = None  # confirmed comb seeds period once


def analog_period_hint(state: DecodeState | None, fs: float) -> float | None:
    """Known analog line period from DecodeState, or None."""
    if state is None or state.period is None or fs <= 0.0:
        return None
    hz = float(fs) / float(state.period)
    if not (ANALOG_LINE_LO_HZ < hz < ANALOG_LINE_HI_HZ):
        return None
    return float(state.period)


def drop_dead_sticky(state: DecodeState | None) -> None:
    """Predicted t0 is dead. Keep line period; forget field-start."""
    if state is None:
        return
    state.sticky = False
    state.abs_t0 = None
    state.lost = max(int(state.lost), 3)


def hold_visible_lock(state: DecodeState | None, frame: Frame | None, *,
                      fs: float, abs_start: float = 0.0) -> None:
    """After a visible analog raster, keep period so the next IQ can track.

    Sets sticky only when field_t0 is known (decode/free_run found a start)
    and the picture is analog_usable. Does not invent t0.
    """
    if state is None or frame is None or fs <= 0.0:
        return
    if not analog_usable(frame) or raster_is_black(frame):
        return
    hz = float(frame.line_rate or 0.0)
    if not (ANALOG_LINE_LO_HZ < hz < ANALOG_LINE_HI_HZ):
        return
    state.period = float(fs) / hz
    std = str(frame.standard or "?")
    if std in ("", "?"):
        std = standard_from_line_rate(hz, max_err_hz=None)
    state.standard = std
    t0 = getattr(frame, "field_t0", None)
    if t0 is not None:
        state.abs_t0 = float(abs_start) + float(t0)
    state.lost = 0
    state.sticky = bool(field_hold_ok(frame) and state.abs_t0 is not None)


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
H_ROLL_DEAD_PX = 1
H_ROLL_ALPHA = 0.15
H_EDGE_STABLE_N = 3
H_EDGE_STABLE_SPREAD = 1
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
    d = np.diff(e)
    if d.size >= 8:
        meas = float(np.median(d))
        if 0.85 * period < meas < 1.15 * period:
            period = meas
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


def edge_col_stable(cols: list | tuple, *, need: int = H_EDGE_STABLE_N,
                    max_spread: int = H_EDGE_STABLE_SPREAD) -> bool:
    """True when recent H-sync edge columns agree within 1 px."""
    seq = [int(c) for c in cols]
    if len(seq) < int(need):
        return False
    recent = seq[-int(need):]
    return max(recent) - min(recent) <= int(max_spread)


def h_phase_should_nudge(frac: float, *, pic_locked: bool = False,
                         edge_stable: bool = True,
                         deadband: float = H_PHASE_DEADBAND) -> bool:
    """Residual H-phase after TBC. |frac|<0.01 (~1 sample / ~3.6°) is idle."""
    if pic_locked or not edge_stable:
        return False
    return abs(float(frac)) >= float(deadband)


def h_roll_step(prev: int | None, auto: int, width: int, *,
                pic_locked: bool = False,
                edge_stable: bool = True,
                dead_px: int = H_ROLL_DEAD_PX,
                alpha: float = H_ROLL_ALPHA) -> int:
    """Slow H-roll. Dead zone 1 px; freeze when locked or the edge is noisy.

    First park (prev is None) still applies so a mid-frame pit can land
    on the porch. `auto==0` means this raster is already parked — drop
    a stale roll rather than EMA back through the picture. After that,
    |Δ| ≤ 1 px and pic_locked leave the raster.
    """
    w = max(1, int(width))
    auto_i = int(auto) % w
    if prev is None:
        return auto_i
    if auto_i == 0:
        return 0
    prev_i = int(prev) % w
    delta = (auto_i - prev_i + w // 2) % w - w // 2
    if pic_locked or not edge_stable or abs(delta) <= int(dead_px):
        return prev_i
    return int(prev_i + float(alpha) * delta) % w


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
              h_phase_frac: float = 0.0,
              pic_locked: bool = False) -> np.ndarray:
    """Park H-blank on the left porch, then a deadbanded manual nudge.

    Phase only — crop is a later slice, never a roll into the middle.
    A large h_phase_frac that would land mid-frame is ignored (auto
    park stays); we do not pretend the slider applied.
    Auto-roll after the first park is slow, 1 px dead-zoned, and frozen
    when `pic_locked` or `h_sync_edge_col` jitters across lines.
    """
    if luma is None or luma.ndim != 2:
        return luma
    w = luma.shape[1]
    pit = h_phase_col(luma)
    auto = h_auto_roll(pit, w)
    if state is not None:
        hist = tuple(state.h_edge_hist) + (int(pit),)
        state.h_edge_hist = hist[-5:]
        stable = True if state.h_roll is None else edge_col_stable(state.h_edge_hist)
        auto = h_roll_step(
            state.h_roll, auto, w,
            pic_locked=pic_locked, edge_stable=stable)
        state.h_roll = auto
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
    # NTSC active field is 240. Cropping 6 footer lines made the LOCK
    # footer read 234 and ate the last OSD row. PAL 288 still allows it.
    bottom = min(bl, max(0, h - 24))
    if 220 <= h <= 252:
        bottom = 0
    out = luma[:, left:] if left else luma
    if bottom:
        out = out[:-bottom]
    return out


def _fit_height(luma: np.ndarray, target: int) -> np.ndarray:
    """Deterministic crop/black-pad for explicitly incomplete diagnostics.

    Repeating the last active row hid missing field geometry and produced
    short "locked" rasters.  Complete streaming fields never call this helper.
    """
    h, w = luma.shape
    if h == target:
        return luma
    if h > target:
        return luma[:target]
    out = np.empty((target, w), dtype=luma.dtype)
    out[:h] = luma
    out[h:] = 0
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
        frac = TBC_SEARCH_FRAC
    picked = edges[found] if found.any() else edges
    if picked.size >= 9:
        meas = float(np.median(np.diff(picked)))
        if 0.85 * period < meas < 1.15 * period and abs(meas - period) > 0.12:
            period = meas
            t00 = float(picked[0])
            starts = t00 + np.arange(len(starts), dtype=np.float64) * period
            edges, found = _tbc_line_edges(
                v, starts, period, thr, search_frac=frac, return_found=True)
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


def estimate_line_hz(v: np.ndarray, fs: float, *,
                     stats: dict[str, int] | None = None) -> float | None:
    """Analog line rate from H-sync pulse-train periodicity (14–17.5 kHz).

    Raw luma ACF is a broad sawtooth. The binary sync mask has a sharp
    comb; that peak is the wrap rate that stops free-run diagonal shear.
    """
    if v is None or fs <= 0.0:
        return None
    x = np.asarray(v, dtype=np.float32)
    if x.size < int(fs * 0.012):
        return None
    n_use = min(x.size, int(fs * LINE_HUNT_WINDOW_S))
    x = x[-n_use:]
    lo, hi = np.percentile(x[::8], [0.5, 99.5])
    if hi - lo < 1e-9:
        return None
    xn = (x - lo) / (hi - lo)
    dec = max(
        1,
        int(np.ceil(float(fs) / LINE_HUNT_TARGET_FS)),
        int(np.ceil(float(n_use) / LINE_HUNT_MAX_INPUT_SAMPLES)),
    )
    fs_d = float(fs) / float(dec)
    n_dec = int((n_use + dec - 1) // dec)
    lag_lo = max(2, int(np.floor(fs_d / 17500.0)))
    lag_hi = min(n_dec // 3, int(np.ceil(fs_d / 14000.0)))
    if lag_hi <= lag_lo + 4:
        return None
    best_hz = None
    best_ratio = 0.0
    nfft = int(2 ** int(np.ceil(np.log2(max(8, n_dec * 2)))))
    nfft = min(nfft, LINE_HUNT_MAX_NFFT)
    if stats is not None:
        stats["input_samples"] = n_dec
        stats["nfft"] = nfft
    for pulse in ((xn < 0.18), (xn > 0.82)):
        if int(pulse.sum()) < 16:
            continue
        xd = pulse.astype(np.float32)[::dec]
        if xd.size < 64:
            continue
        spec = np.fft.rfft(xd - float(xd.mean()), nfft)
        ac = np.fft.irfft(np.abs(spec) ** 2, nfft).real
        band = ac[lag_lo:lag_hi + 1]
        if band.size < 5:
            continue
        k = int(np.argmax(band))
        peak = float(band[k])
        floor = float(np.median(np.abs(band)))
        if peak <= 1.4 * max(floor, 1e-18):
            continue
        if 0 < k < band.size - 1:
            y0, y1, y2 = float(band[k - 1]), peak, float(band[k + 1])
            den = y0 - 2.0 * y1 + y2
            if abs(den) > 1e-18:
                k = k + 0.5 * (y0 - y2) / den
        lag = float(lag_lo) + float(k)
        if lag < 2.0:
            continue
        hz = fs_d / lag
        if not (14000.0 < hz < 17500.0):
            continue
        ratio = peak / max(floor, 1e-18)
        if ratio > best_ratio:
            best_ratio = ratio
            best_hz = float(hz)
    return best_hz


def _vsync_groups(frac: np.ndarray, win: int, vs_thr: float):
    vs = np.flatnonzero(frac > float(vs_thr))
    if vs.size < 3:
        return None
    brk = np.flatnonzero(np.diff(vs) > win)
    starts = np.concatenate(([vs[0]], vs[brk + 1])) if len(brk) else vs[:1]
    ends = np.concatenate((vs[brk], vs[-1:])) if len(brk) else vs[-1:]
    return starts, ends


def _pulse_field_t0(below: np.ndarray, period: float, vblank: int):
    """Classic PAL/NTSC serration: a 1-line window that stays mostly low.

    Weak 0.22 thresholds used to take the *first* group. A mid-field
    dark burst then locked t0 there and shipped a torn 288-line pack.
    Keep only groups that look like a few V-sync lines, and prefer the
    strongest peak (classic 0.55 first).
    """
    win = int(period)
    if below.size < win * 40 or win < 8:
        return None
    csum = np.cumsum(np.concatenate(([0], below.astype(np.int32))))
    frac = (csum[win:] - csum[:-win]) / win
    med = float(np.median(frac)) if frac.size else 0.0
    thresholds = (
        0.55,
        float(np.clip(max(0.28, med + 0.18), 0.28, 0.50)),
        float(np.clip(max(0.22, med + 0.12), 0.22, 0.40)),
    )
    min_span = max(2, win // 4)
    max_span = int(15 * win)
    best = None
    best_score = -1.0
    seen = set()
    for vs_thr in thresholds:
        key = round(float(vs_thr), 3)
        if key in seen:
            continue
        seen.add(key)
        g = _vsync_groups(frac, win, vs_thr)
        if g is None:
            continue
        group_starts, group_ends = g
        for gs, ge in zip(group_starts, group_ends):
            span = int(ge) - int(gs)
            if span < min_span or span > max_span:
                continue
            sl = frac[int(gs):int(ge) + 1]
            peak = float(sl.max()) if sl.size else 0.0
            score = peak + (0.15 if vs_thr >= 0.50 else 0.0)
            if score <= best_score:
                continue
            best_score = score
            end_vs = int(ge)
            start = int(end_vs + win * int(vblank))
            later = group_starts[group_starts > start + win]
            next_vs = int(later[0]) if len(later) else None
            best = (float(start), None if next_vs is None else float(next_vs))
        if best is not None and vs_thr >= 0.50:
            return best
    return best


def _energy_field_t0(v: np.ndarray, period: float, vblank: int,
                     field_lines: float = 312.5, t_hint: float = 0.0):
    """FPV often lacks a 0.55 V-sync pulse. V-blank is still the darkest run."""
    if period < 8.0 or v.size < int(period * 80):
        return None
    t0 = max(0.0, float(t_hint))
    n_scan = int(min((len(v) - t0) / period - 2, 420))
    k = max(8, int(vblank))
    if n_scan < k + 48:
        return None
    starts = t0 + np.arange(n_scan, dtype=np.float64) * period
    offs = np.linspace(0.18, 0.92, 10) * period
    idx = starts[:, None] + offs[None, :]
    np.clip(idx, 0, len(v) - 1, out=idx)
    means = v[idx.astype(np.int32)].mean(axis=1)
    c = np.cumsum(np.concatenate(([0.0], means.astype(np.float64))))
    roll = (c[k:] - c[:-k]) / k
    j = int(np.argmin(roll))
    floor = float(roll[j])
    mid = float(np.median(roll))
    # Bright FPV (3470 hillside) often has V-blank only ~0.04 below active.
    if mid - floor < 0.035:
        return None
    active = float(starts[min(j + k, n_scan - 1)])
    next_vs = None
    lo = j + max(int(0.55 * field_lines), k + 8)
    hi = min(len(roll), j + int(field_lines) + k)
    if hi > lo + 4:
        j2 = lo + int(np.argmin(roll[lo:hi]))
        if mid - float(roll[j2]) >= 0.05:
            next_vs = float(starts[j2])
    return active, next_vs


def _choose_field_t0(pulse, energy, period: float):
    """Prefer energy when a weak/false pulse is more than ~1/10 field off."""
    if pulse is None:
        return energy
    if energy is None:
        return pulse
    if abs(float(pulse[0]) - float(energy[0])) > 32.0 * max(float(period), 1.0):
        return energy
    return pulse


def _field_active_t0(v: np.ndarray, period: float, vblank: int, *,
                     below: np.ndarray | None = None,
                     field_lines: float = 312.5,
                     t_hint: float = 0.0):
    """Active-video t0 after V-blank, plus the next V-blank sample.

    Pulse and energy both run. A mid-field dark burst used to win the
    pulse path and pack two half-fields into 288 lines (3470 shear).
    Starting at the first H-sync does the same when both miss.
    """
    if below is None:
        below = v < 0.18
    pulse = _pulse_field_t0(below, period, vblank)
    energy = _energy_field_t0(
        v, period, vblank, field_lines=field_lines, t_hint=t_hint)
    return _choose_field_t0(pulse, energy, period)


def _v_blank_row(luma: np.ndarray, band: int = 25) -> int | None:
    """Start row of a mid-frame V-blank trough, or None if already framed.

    Absolute floor<=40 rejected bright hillside rasters (median ~160,
    trough ~50–80). Contrast is relative to the row median.
    """
    if luma is None or luma.ndim != 2:
        return None
    h = int(luma.shape[0])
    k = min(max(int(band), 8), max(8, h // 5))
    if h < k + 48:
        return None
    rowm = luma.astype(np.float32).mean(axis=1)
    med = float(np.median(rowm))
    c = np.cumsum(np.concatenate(([0.0], rowm)))
    roll = (c[k:] - c[:-k]) / k
    j = int(np.argmin(roll))
    floor = float(roll[j])
    if med - floor < max(8.0, 0.12 * max(med, 1.0)):
        return None
    centre = j + k / 2.0
    if not (0.14 * h <= centre <= 0.86 * h):
        return None
    return int(j)


def _v_unwrap(luma: np.ndarray, band: int = 25) -> np.ndarray:
    """Drop the pre-blank fragment. Do not circular-roll it onto the footer.

    Refuse a slice that would leave a short/black remainder — that is how
    a mid-frame trough wiped the active picture into a 282-line black field.
    """
    j = _v_blank_row(luma, band)
    if j is None:
        return luma
    rest = luma[j:]
    if rest.shape[0] < max(32, int(0.45 * luma.shape[0])):
        return luma
    if (float(rest.mean()) < VISIBLE_LUMA_MEAN
            and float(luma.mean()) >= VISIBLE_LUMA_MEAN):
        return luma
    return rest


def _resync_t0(luma: np.ndarray, t0: float, period: float, vblank: int,
               field_lines: float, n_samples: int, max_lines: int,
               next_vs: float | None = None):
    """If luma still has a mid-frame V-blank, move t0 to active video.

    Circular unwrap used to glue the previous field onto the footer
    (5068 black wedge / 3470 walking bar). Re-slice from the trough.
    """
    j = _v_blank_row(luma, vblank)
    if j is None or j < 12:
        return None
    # `_v_blank_row` returns the beginning of the blanking trough. Raster
    # rendering must resume after it; treating that row as active t0 held
    # sticky geometry one field-end (~287 lines) away from the true H/V edge.
    t0_alt = float(t0) + float(j + vblank) * float(period)
    avail = (float(n_samples) - t0_alt - period) / period
    if next_vs is not None:
        avail = min(avail, (float(next_vs) - t0_alt) / period - 1.0)
    n_alt = int(min(max_lines, avail))
    if n_alt < int(max_lines * 0.55):
        back = t0_alt - float(field_lines) * period
        if back >= 0.0:
            t0_alt = back
            avail = (float(n_samples) - t0_alt - period) / period
            n_alt = int(min(max_lines, avail))
    if n_alt < 32:
        return None
    return t0_alt, n_alt


def _attempt(v: np.ndarray, fs: float, width: int, max_lines: int,
             auto_levels: bool = True, sharpen: float = 0.0,
             state: DecodeState | None = None,
             h_phase_frac: float = 0.0,
             period_hint: float | None = None,
             line_hint: LineRateHint | None = None):
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
    max_lines = cap_field_lines(max_lines)

    thr = 0.18                        # між вершиною синхри і рівнем гасіння
    below, edges = _sync_edges(v, thr)
    score = 0.0
    period = None
    if len(edges) >= 12:
        d = np.diff(edges).astype(np.float64)
        # орієнтир — найдовший поширений інтервал (півкадрові імпульси коротші)
        # >= : a rock-steady H-sync has identical intervals, so '>' is empty
        # and we used to drop a perfectly regular analog raster.
        d_hi = d[d >= np.percentile(d, 40)] if d.size else d
        if d_hi.size:
            med = float(np.median(d_hi))
            keep = (d > med * 0.75) & (d < med * 1.25)
            score = float(keep.mean())
            if keep.sum() >= 8:
                period = float(np.median(d[keep]))
                e = edges.astype(np.float64)
                k = np.round((e - e[0]) / period)
                good_k = np.abs((e - e[0]) - k * period) < period * 0.2
                if good_k.sum() >= 8:
                    kk, ee = k[good_k], e[good_k]
                    A = np.vstack([kk, np.ones_like(kk)]).T
                    period, _ = np.linalg.lstsq(A, ee, rcond=None)[0]
                    period = float(period)
    line_rate = (fs / period) if period else 0.0
    anchored_hz = 0.0
    if line_hint is not None and line_hint.confirmed:
        anchored_hz = float(line_hint.line_hz or 0.0)
    if ANALOG_LINE_LO_HZ < anchored_hz < ANALOG_LINE_HI_HZ:
        # Multi-window comb confirmation is deliberately authoritative over
        # a single block's threshold edges.  Snow/porch edges can form a very
        # regular but false 16–17 kHz period.
        period = float(fs) / anchored_hz
        line_rate = anchored_hz
    # Skip the 50 ms ACF/FFT when H-sync already gave an analog period,
    # or when a previous visible analog frame already measured it.
    if period is None or not (14000 < line_rate < 17500):
        hint = period_hint
        if hint is None:
            hint = analog_period_hint(state, fs)
        hint_hz = (fs / hint) if hint else 0.0
        if hint is not None and 14000 < hint_hz < 17500:
            period = float(hint)
            line_rate = hint_hz
            if score < 0.25:
                score = 0.25
        else:
            est = (
                line_hint.resolve(v, fs)
                if line_hint is not None
                else estimate_line_hz(v, fs)
            )
            if est is not None:
                period = fs / est
                line_rate = est
                if score < 0.25:
                    score = 0.25
    if period is None or not (14000 < line_rate < 17500):
        return 0.0, None, None
    if line_hint is not None:
        line_hint.accept(line_rate)
    # No Hz cap here: 14–17.5 kHz already proved analog. '?' geometry
    # (287.5 lines) breaks PAL tracking when the camera is >100 Hz off.
    standard = standard_from_line_rate(line_rate, max_err_hz=None)
    a0_frac, a1_frac, vblank = STD_GEOM[standard]

    # Field start: broad V-sync if present, else darkest 25-line run.
    # First H-sync in the capture is mid-field on FPV and packs two
    # half-fields into 288 lines (diagonal black bar, stacked picture).
    field_lines = FIELD_LINES.get(standard, FIELD_LINES["?"])
    hit = _field_active_t0(
        v, period, vblank, below=below, field_lines=field_lines)
    locked = hit is not None
    next_vs_start = None
    if hit is not None:
        start = int(hit[0])
        if hit[1] is not None:
            next_vs_start = int(hit[1])
    else:
        start = int(edges[0]) if len(edges) else 0

    if len(edges):
        nxt = edges[edges >= start]
        if len(nxt) == 0:
            nxt = edges
        t0 = float(nxt[0])
    else:
        t0 = float(start)

    a0 = a0_frac * period
    a1 = a1_frac * period
    avail = (len(v) - t0 - period) / period
    if next_vs_start is not None:
        avail = min(avail, (next_vs_start - t0) / period - 1.0)
    n_lines = cap_field_lines(int(min(max_lines, avail)), max_lines)
    if a1 <= a0 or n_lines < 32:
        return 0.0, None, None

    starts = _genlock_starts(v, float(t0), period, n_lines, thr)
    luma = _render(v, starts, period, a0_frac, a1_frac, width,
                   auto_levels=auto_levels, sharpen=sharpen, state=state,
                   h_phase_frac=h_phase_frac, thr=thr, tbc_locked=False)
    snap = _resync_t0(
        luma, t0, period, vblank, field_lines, len(v), max_lines,
        next_vs_start)
    if snap is not None:
        t0, n_lines = snap
        n_lines = cap_field_lines(n_lines, max_lines)
        locked = True
        starts = _genlock_starts(v, float(t0), period, n_lines, thr)
        luma = _render(v, starts, period, a0_frac, a1_frac, width,
                       auto_levels=auto_levels, sharpen=sharpen, state=state,
                       h_phase_frac=h_phase_frac, thr=thr, tbc_locked=False)
    h_before = int(luma.shape[0])
    luma = _v_unwrap(luma, band=vblank)
    dropped = h_before - int(luma.shape[0])
    if dropped > 0:
        t0 = float(t0) + float(dropped) * float(period)
        n_lines = max(32, n_lines - dropped)
    n_lines = cap_field_lines(min(n_lines, int(luma.shape[0])), max_lines)
    if state is not None:
        if state.target_lines is None:
            state.target_lines = max_lines
        luma = _fit_height(luma, state.target_lines)
    expected_lines = ACTIVE_FIELD_LINES.get(standard, ACTIVE_FIELD_LINES["?"])
    complete = n_lines >= int(np.ceil(0.95 * expected_lines))
    return score, Frame(
        luma=luma,
        line_rate=line_rate, lines=n_lines,
        standard=standard, locked=bool(locked and complete),
        field_parity=0 if locked and complete else None,
        field_t0=t0, incomplete=not complete), t0


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
    max_lines = cap_field_lines(max_lines)
    hold = bool(state.sticky)
    # Ring snapshot often starts a few lines after t0. Shift to a field
    # start that actually sits in this block instead of failing track.
    if local_t0_pred < 0:
        local_t0_pred += field_period
        n_fields += 1
    elif local_t0_pred >= len(v) - 8 and n_fields > 0:
        local_t0_pred -= field_period
        n_fields -= 1
    tol = tol_frac * period
    lo = int(max(0, local_t0_pred - tol))
    hi = int(min(len(v), local_t0_pred + tol))
    if hi - lo < 8:
        if not hold or not (0.0 <= local_t0_pred < len(v)):
            return None
        lo = int(max(0, local_t0_pred - period))
        hi = int(min(len(v), local_t0_pred + period))
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
        # Sticky used to paint predicted t0 with no H-edge → black field,
        # locked=True, 282 lines, sync gone. Re-blind-hunt instead.
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
    a0_frac, a1_frac, vblank = STD_GEOM[standard]
    field_lines = FIELD_LINES.get(standard, FIELD_LINES["?"])
    a0 = a0_frac * period
    a1 = a1_frac * period
    avail = (len(vv) - t0 - period) / period
    n_lines = cap_field_lines(int(min(max_lines, avail)), max_lines)
    next_vs = None
    # Sticky + already framed (V-blank at top, ~288): skip pulse/energy.
    # Sticky + short/torn (~173 mid-field): still snap to V-blank.
    want_snap = (not hold) or field_needs_resnap(n_lines, max_lines)
    if want_snap:
        hit = _field_active_t0(
            vv, period, vblank, field_lines=field_lines,
            t_hint=max(0.0, t0 - 4 * period))
        if hit is not None:
            snap_t0, next_vs = float(hit[0]), hit[1]
            if not hold:
                vb_start = snap_t0 - float(vblank) * period
                if vb_start > t0 + 4 * period:
                    n_clip = int((vb_start - t0) / period - 1.0)
                    if n_clip >= 32:
                        n_lines = min(n_lines, n_clip)
            alt_avail = (len(vv) - snap_t0 - period) / period
            if next_vs is not None:
                alt_avail = min(alt_avail, (float(next_vs) - snap_t0) / period - 1.0)
            alt_n = cap_field_lines(int(min(max_lines, alt_avail)), max_lines)
            if alt_n > n_lines + 8:
                t0 = snap_t0
                n_lines = alt_n
        if hold:
            t0, n_lines = _extend_short_field_t0(
                t0, n_lines, period, len(vv), max_lines)
    if a1 <= a0 or n_lines < 32:
        return None

    starts = _genlock_starts(vv, float(t0), period, n_lines, thr)
    luma = _render(vv, starts, period, a0_frac, a1_frac, width,
                   auto_levels=auto_levels, sharpen=sharpen, state=state,
                   h_phase_frac=h_phase_frac, thr=thr, tbc_locked=True)
    want_resync = (not hold) or field_needs_resnap(
        n_lines, max_lines, luma, vblank)
    if want_resync:
        snap = _resync_t0(
            luma, t0, period, vblank, field_lines, len(vv), max_lines,
            next_vs)
        if snap is not None:
            t0, n_lines = snap
            n_lines = cap_field_lines(n_lines, max_lines)
            starts = _genlock_starts(vv, float(t0), period, n_lines, thr)
            luma = _render(vv, starts, period, a0_frac, a1_frac, width,
                           auto_levels=auto_levels, sharpen=sharpen, state=state,
                           h_phase_frac=h_phase_frac, thr=thr, tbc_locked=True)
    h_before = int(luma.shape[0])
    luma = _v_unwrap(luma, band=vblank)
    dropped = h_before - int(luma.shape[0])
    if dropped > 0:
        t0 = float(t0) + float(dropped) * float(period)
        n_lines = max(32, n_lines - dropped)
    n_lines = cap_field_lines(min(n_lines, int(luma.shape[0])), max_lines)
    if state.target_lines is None:
        state.target_lines = max_lines
    luma = _fit_height(luma, state.target_lines)
    expected_lines = ACTIVE_FIELD_LINES.get(standard, ACTIVE_FIELD_LINES["?"])
    complete = n_lines >= int(np.ceil(0.95 * expected_lines))
    frame = Frame(luma=luma,
                line_rate=fs / period, lines=n_lines,
                standard=standard, locked=complete,
                field_parity=int(n_fields) % 2,
                field_t0=t0, incomplete=not complete)

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
        h_pll: bool = False,
        line_hint: LineRateHint | None = None) -> Frame | None:
    """Декодує напівкадр.

    Без `state` (або на першому виклику) — точнісінько як раніше:
    повний перебір полярності й сліпий пошук синхри. Якщо переданий
    `state` уже містить період з попереднього успішного кадру, спершу
    пробує трекінг у вузькому вікні. Sticky t0 тримається лише коли є
    справжній H-фронт біля прогнозу І видима яскравість; інакше t0
    скидається одразу і цей блок іде в сліпий пошук (engine далі може
    викликати free_run). Немає малювання поля без фронту.
    """
    if len(base) < int(fs * 0.02):        # менше 20 мс — нема сенсу
        return None
    v = base.astype(np.float32)
    max_lines = cap_field_lines(max_lines)

    anchored_hz = 0.0
    if line_hint is not None and line_hint.confirmed:
        anchored_hz = float(line_hint.line_hz or 0.0)
    if (state is not None
            and ANALOG_LINE_LO_HZ < anchored_hz < ANALOG_LINE_HI_HZ
            and (
                state.coarse_hint_hz is None
                or abs(float(state.coarse_hint_hz) - anchored_hz) > 1.0
            )):
        previous_hz = (
            float(fs) / float(state.period)
            if state.period is not None and state.period > 0.0 else 0.0
        )
        state.period = float(fs) / anchored_hz
        state.coarse_hint_hz = anchored_hz
        state.standard = standard_from_line_rate(anchored_hz, max_err_hz=None)
        if previous_hz and abs(previous_hz - anchored_hz) > 50.0:
            # A stale/spurious period also invalidates its extrapolated field
            # phase.  Keep the trusted period but reacquire V/H position.
            state.abs_t0 = None
            state.sticky = False
            state.t0_err = None
            state.h_roll = None
            state.h_edge_hist = ()

    lost_lim = 8 if (state is not None and state.sticky) else 3
    if state is not None and state.period is not None and state.lost < lost_lim:
        # Sticky: one narrow window. Three tols + blind every block is
        # why decode sat at ~2.7 s after a good 3430 flash.
        tols = (0.55,) if state.sticky else (0.55, 0.75, 0.95)
        tracked = None
        for tol_frac in tols:
            tracked = _attempt_tracked(v, fs, width, max_lines, state, abs_start,
                                       tol_frac, auto_levels=auto_levels,
                                       sharpen=sharpen,
                                       h_phase_frac=h_phase_frac,
                                       h_pll=h_pll)
            if tracked is not None:
                break
        if tracked is not None:
            frame, abs_t0 = tracked
            if raster_is_black(frame) or not analog_usable(frame):
                # Dead sticky: H-edge painted sync/porch or snow. Drop t0,
                # keep period, re-blind this block. Never ship this raster
                # as success (engine used to skip free_run on non-None).
                drop_dead_sticky(state)
            else:
                state.abs_t0 = abs_t0
                state.lost = 0
                state.sticky = bool(field_hold_ok(frame))
                frame.luma = _h_crop(frame.luma, crop_left_frac, crop_bottom_lines)
                frame.lines = published_lines(frame)
                return frame
        else:
            # No H-edge at predicted t0 — do not paint. Drop sticky now.
            drop_dead_sticky(state)

    # Відому полярність пробуємо першою: на зриві трекінгу це часто
    # одразу дає поле і не ганяє другий повний сліпий прохід.
    signs = (1.0, -1.0)
    known = state is not None and state.period is not None
    period_hint = analog_period_hint(state, fs)
    if line_hint is None:
        line_hint = LineRateHint.from_period(period_hint, fs)
    elif period_hint is not None:
        line_hint.accept(fs / period_hint)
    if known:
        signs = (state.sign, -state.sign)
    best_s, best_f, best_t0, best_sign = 0.0, None, None, 1.0
    for sign in signs:
        s, f, t0 = _attempt(v * sign, fs, width, max_lines,
                            auto_levels=auto_levels, sharpen=sharpen,
                            h_phase_frac=h_phase_frac,
                            period_hint=period_hint,
                            line_hint=line_hint)
        if f is not None and s > best_s:
            best_s, best_f, best_t0, best_sign = s, f, t0, sign
            if known and best_s > 0.85:
                break
    # Polarity pick uses 0.5. A weaker raster still has lines — emit it
    # (engine used to go black / 0 fps when this returned None).
    if best_f is None:
        return None
    line_hint.accept(best_f.line_rate)
    usable = analog_usable(best_f)
    complete = complete_field_geometry(best_f)
    if usable and complete:
        best_f.locked = True
        best_f.free_run = False
    # Messy FPV H-sync often scores ≤0.5 even on a framed landscape.
    # Used to skip seeding → every IQ block re-ran full blind hunt.
    # Seed on analog_usable even when the H-interval score is weak.
    weak = best_s <= 0.5 and not (usable and complete)
    if weak:
        best_f.locked = False
    elif state is not None:
        state.sign = best_sign
        state.period = fs / best_f.line_rate
        state.standard = best_f.standard
        state.abs_t0 = abs_start + best_t0
        state.lost = 0
        state.t0_err = None
        state.h_roll = None
        state.h_edge_hist = ()
        state.sticky = field_hold_ok(best_f)
        if state.target_lines is None:
            state.target_lines = max_lines
        best_f.luma = _fit_height(best_f.luma, state.target_lines)
    if best_f.field_t0 is None:
        best_f.field_t0 = best_t0
    best_f.luma = _h_crop(best_f.luma, crop_left_frac, crop_bottom_lines)
    best_f.lines = published_lines(best_f)
    return best_f


def free_run(base: np.ndarray, fs: float, width: int = 640,
             max_lines: int = 288, line_hz: float = 15625.0,
             period_hint: float | None = None,
             line_hint: LineRateHint | None = None) -> Frame | None:
    """Wrap FM baseband at the hunted analog line rate.

    When decode() finds no sync, LOCK still has RF. Analog then shows a
    raster instead of a black pane. Wrapping at 15625 when the camera is
    even ~100 Hz off shears the whole field; hunt the period first.
    Skip that hunt when a previous visible analog frame already measured it.
    """
    if base is None or fs <= 0.0:
        return None
    if len(base) < int(fs * 0.02):
        return None
    max_lines = cap_field_lines(max_lines)
    hunted = None
    if (line_hint is not None and line_hint.confirmed
            and ANALOG_LINE_LO_HZ
            < float(line_hint.line_hz or 0.0)
            < ANALOG_LINE_HI_HZ):
        hunted = float(line_hint.line_hz)
    elif period_hint is not None and period_hint > 0.0:
        hint_hz = float(fs) / float(period_hint)
        if 14000.0 < hint_hz < 17500.0:
            hunted = hint_hz
            if line_hint is not None:
                line_hint.accept(hunted)
    if hunted is None:
        hunted = (
            line_hint.resolve(base, fs)
            if line_hint is not None
            else estimate_line_hz(base, fs)
        )
    if hunted is not None:
        line_hz = hunted
    if line_hz <= 0.0:
        return None
    period = float(fs) / float(line_hz)
    v = np.asarray(base, dtype=np.float32)
    lo, hi = np.percentile(v[::8], [0.5, 99.5])
    if hi - lo < 1e-9:
        return None
    v = (v - lo) / (hi - lo)
    standard = standard_from_line_rate(line_hz, max_err_hz=None)
    a0_frac, a1_frac, vblank = STD_GEOM[standard]
    field_lines = FIELD_LINES[standard]
    t0 = 0.0
    next_vs = None
    hit = _field_active_t0(v, period, vblank, field_lines=field_lines)
    if hit is not None:
        t0, next_vs = float(hit[0]), hit[1]
    avail = (len(v) - t0 - period) / period
    if next_vs is not None:
        avail = min(avail, (float(next_vs) - t0) / period - 1.0)
    n_lines = cap_field_lines(int(min(max_lines, avail)), max_lines)
    if n_lines < 32:
        t0 = 0.0
        n_lines = cap_field_lines(int(min(max_lines, (len(v) / period) - 1)), max_lines)
    if n_lines < 32 or period < 4.0:
        return None
    starts = _genlock_starts(v, t0, period, n_lines, 0.18)
    luma = _render(v, starts, period, a0_frac, a1_frac, width,
                   auto_levels=True, sharpen=0.0, state=None,
                   h_phase_frac=0.0, tbc_locked=False)
    snap = _resync_t0(
        luma, t0, period, vblank, field_lines, len(v), max_lines,
        next_vs)
    if snap is not None:
        t0, n_lines = snap
        n_lines = cap_field_lines(n_lines, max_lines)
        starts = _genlock_starts(v, t0, period, n_lines, 0.18)
        luma = _render(v, starts, period, a0_frac, a1_frac, width,
                       auto_levels=True, sharpen=0.0, state=None,
                       h_phase_frac=0.0, tbc_locked=False)
    luma = _v_unwrap(luma, band=vblank)
    n_lines = cap_field_lines(min(n_lines, int(luma.shape[0])), max_lines)
    return Frame(
        luma=luma, line_rate=float(line_hz), lines=n_lines,
        standard=standard, locked=False, free_run=True,
        field_t0=t0, incomplete=True,
    )


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