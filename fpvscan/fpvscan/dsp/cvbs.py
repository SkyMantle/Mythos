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
from pathlib import Path
import json as _json
import time as _time
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch as _welch

# #region agent log
_DBG_LOGS = [
    r"D:\projects\Mythos\debug-85d685.log",
    str(Path(__file__).resolve().parents[3] / "debug-85d685.log"),
    str(Path(__file__).resolve().parents[2] / "debug-85d685.log"),
]
_DBG_INGEST = "http://127.0.0.1:7301/ingest/4fe89a75-6982-4b18-b3b4-713f945eb0cd"

def _agent_log(hypothesis_id: str, location: str, message: str, data: dict):
    payload = {
        "sessionId": "85d685",
        "timestamp": int(_time.time() * 1000),
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
    }
    line = _json.dumps(payload, ensure_ascii=False) + "\n"
    for p in _DBG_LOGS:
        try:
            with open(p, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass
    try:
        from urllib.request import Request, urlopen
        req = Request(_DBG_INGEST, data=line.encode("utf-8"),
                      headers={"Content-Type": "application/json",
                               "X-Debug-Session-Id": "85d685"})
        urlopen(req, timeout=0.2).read()
    except Exception:
        pass
# #endregion
 
_dbg_last = [0.0]   # тротлінг діагностичного друку, спільний на процес
 
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
 
 
def _fft_line_rate(v: np.ndarray, fs: float,
                    lo_hz: float = 14000.0, hi_hz: float = 17500.0):
    """Спектральна оцінка рядкової частоти — той самий принцип, що й
    `demod.classify_video()`, але напряму на сигналі, який вже є в
    `_attempt()`.
 
    Знахідка 02.09.2026 ("у смузі PAL/NTSC=X/Y", ~5-8% замість ~0%):
    `classify_video()` (FFT з довгим накопиченням за десятки-сотні
    періодів рядка) впевнено бачить рядкову лінію 15.6кГц навіть тоді,
    коли пошук по сусідніх фронтах нижче в `_attempt()` губиться в
    шумі — жоден сусідній інтервал не домінує (score 0.13-0.18), хоча
    справжній період і присутній серед сирих даних. FFT стійкий до
    джиттеру й неглибокого занурення синхроімпульсу саме тому, що не
    залежить від одного порогового перетину на рядок — накопичує
    потужність по всьому знімку одразу.
 
    Друга знахідка, того ж дня, ПІСЛЯ впровадження вище: перша версія
    рахувала один-єдиний неусереднений періодограм (`|FFT|²` з одного
    вікна Ганна) і порівнювала пік із порогом 6дБ над медіаною фону.
    Перевірка на чистому гаусовому шумі (без жодної реальної
    періодичності) показала 199 впевнених "піків" >6дБ зі 200
    випробувань — тобто поріг практично нічого не відсіював. Причина —
    статистика екстремумів: у вікні 14-17.5кГц вміщується ~140-150
    незалежних частотних кошиків, кожен з яких для шуму має
    експоненційний розподіл (варіація потужності одного кошика ~100%
    від його ж середнього); максимум зі ~145 таких кошиків систематично
    відхиляється від медіани на 6-8дБ просто за рахунок їх кількості —
    задовго до того, як з'явиться справжня вузька спектральна лінія.
 
    Фікс — усереднення Уелча (`scipy.signal.welch`, вікна, що
    перекриваються на 50%): замість однієї шумної оцінки на кошик
    рахуємо кілька (K≈6-7 при типовій довжині захоплення) і беремо їхнє
    середнє. Дисперсія оцінки падає ~вдвічі на кожне подвоєння K, тому
    той самий екстремум-із-багатьох-кошиків стає значно менш
    імовірним. Перевірено емпірично (Монте-Карло, 2000-5000 випробувань
    чистого шуму різної довжини захоплення 20-80мс): максимальний
    "підйом" після усереднення Уелча ще ЖОДНОГО разу не перевищив
    ~6.3дБ — тому поріг підняли до 8дБ (як і в `classify_video()`) із
    запасом. Справжня рядкова синхра, навіть при SNR аж до -3дБ, дає
    підйом ~28-30дБ — тобто запас між шумом і сигналом величезний,
    поріг 8дБ нічим не ризикує з боку чутливості.
 
    Повертає (line_rate_hz, prominence_db) або (None, 0.0), якщо у
    вікні `lo_hz`-`hi_hz` немає впевненого піка над фоном.
    """
    dec = max(1, int(fs / 400e3))
    x = v.astype(np.float32)
    if dec > 1:
        c = np.cumsum(np.concatenate(([0.0], x), dtype=np.float64))
        x = ((c[dec:] - c[:-dec]) / dec)[::dec].astype(np.float32)
    fs2 = fs / dec
    if len(x) < 4096:
        return None, 0.0
 
    x = x - x.mean()
    # Сегмент навмисно короткий відносно всього захоплення (~len/4,
    # аж до 8192 відліків) — це і є усереднення Уелча: краще кілька
    # шумніших-по-роздільності, але усереднених оцінок, ніж одна
    # високороздільна й статистично ненадійна (див. docstring вище).
    nperseg = 1 << int(np.floor(np.log2(max(256, len(x) // 4))))
    nperseg = min(nperseg, 8192, 1 << int(np.floor(np.log2(len(x)))))
    freqs, psd = _welch(x, fs=fs2, window="hann", nperseg=nperseg,
                        noverlap=nperseg // 2, detrend=False)
 
    sel = (freqs > lo_hz) & (freqs < hi_hz)
    if not np.any(sel):
        return None, 0.0
    idx = np.where(sel)[0]
    peak_i = idx[np.argmax(psd[idx])]
    peak_f = float(freqs[peak_i])
    peak_p = float(psd[peak_i])

    # Субінна (параболічна) інтерполяція вершини — фікс 03.09.2026.
    #
    # Досі частота піка бралась як ЦЕНТР біна, тобто квантувалась із
    # кроком fs2/nperseg (при типових параметрах 50.9Гц, тобто ±25.4Гц
    # похибки). Для ВИЯВЛЕННЯ лінії цього досить, але період рядка з
    # такою похибкою (0.16%) за 288 рядків екстраполяції `t0+k·period`
    # накопичує зсув до половини рядка — це і є діагональний «шер» на
    # картинці, і саме він ламав нумерацію рядків у `_refine_period()`
    # приблизно з 123-го рядка (далі МНК уточнював період лише по
    # першій третині кадру й тому не рятував).
    #
    # Вершина параболи по трьох сусідніх бінах у ЛОГАРИФМІЧНІЙ шкалі —
    # стандартний спосіб для віконного спектра (в лог-шкалі вершина
    # близька до параболи; на лінійній оцінка зміщена, і саме тому
    # попередня спроба цього фіксу свого часу не спрацювала).
    # Виміряно на реальному записі cap_5003.cf32: похибка рядкової
    # частоти 40.7Гц -> 1.8Гц, накопичений зсув за кадр 419 -> 19
    # відліків, різкість вертикальних структур 14.4 -> 53.6 при
    # еталоні 54.1 (див. bench_period.py).
    if 0 < peak_i < len(psd) - 1:
        lp = np.log(psd[peak_i - 1:peak_i + 2] + 1e-30)
        denom = lp[0] - 2.0 * lp[1] + lp[2]
        if abs(denom) > 1e-30:
            delta = 0.5 * (lp[0] - lp[2]) / denom
            # Вершина за визначенням лежить у межах ±пів-біна; вихід за
            # ці межі означає, що трійка не схожа на пік (шум/плато) —
            # тоді краще лишити центр біна, ніж екстраполювати навмання.
            if -0.5 <= delta <= 0.5:
                peak_f = float(freqs[peak_i] + delta * (freqs[1] - freqs[0]))
 
    bg_sel = (freqs > 10e3) & (freqs < 25e3)
    bg = float(np.median(psd[bg_sel])) + 1e-20
    prominence_db = 10 * np.log10(peak_p / bg)
    return peak_f, prominence_db
 
 
def _refine_period(e: np.ndarray, period: float, tol_frac: float = 0.2,
                    max_iters: int = 10):
    """Ітеративно уточнює період рядка через МНК по фронтах, що узгоджуються
    з періодом на цілій кількості рядків (`good_k`).
 
    Знахідка 02.09.2026 (скріншоти з діагональним "зсувом"/"шером"
    картинки на підтверджених реальних сигналах — `classify_video()`
    впевнений на 100%, а `cvbs` все одно застрягав на `score` 0.20-0.46):
    ОДИН прохід МНК (як було раніше) уточнює період лише по фронтах, що
    вже потрапили в допуск ВІДНОСНО ГРУБОЇ стартової FFT-оцінки. На
    реальному сигналі серед фронтів є домішка "хибних" (де вміст картинки
    теж пірнає нижче порогу `thr` — не лише синхроімпульс), і при грубій
    стартовій оцінці періоду цей допуск відсіює забагато СПРАВЖНІХ фронтів
    синхри, а не тільки хибні. Період уточнюється по невеликій, усіченій
    вибірці (лише перші десятки рядків, поки накопичена похибка не вилізе
    за допуск), лишається неточним на частку відсотка — і ця похибка
    накопичується за 288 рядків екстраполяції у помітний горизонтальний
    зсув: кожен наступний рядок сфотографований трохи не там, де мав би,
    тому пряма вертикальна лінія на картинці виходить діагональною смугою.
 
    Фікс — ітерація: після кожного МНК-уточнення період уже точніший, тож
    повторний відбір `good_k` із НОВИМ періодом ловить більше справжніх
    фронтів (включно з тими, що раніше випадали за допуск лише через
    грубість стартової оцінки), а МНК по більшій і чистішій вибірці дає ще
    точніший період — і так далі, поки кількість "good" фронтів не
    перестане зростати. Перевірено симуляцією (реалістична домішка ~60%
    хибних фронтів + стартова похибка періоду ~0.15%, типова для
    роздільності Уелч-спектру): один прохід — score 0.45, накопичений
    зсув ~334 відліки за 288 рядків (майже половина періоду рядка, точно
    той "шер", що видно на скріншотах); після збіжності ітерацій (~10
    проходів — дешево, це лише numpy по кількох тисячах чисел) — score
    0.78, зсув ~18 відліків (у ~20 разів менше, вже непомітно на око).
 
    Повертає (уточнений_період, good_k_маска_з_останньої_ітерації).
    """
    good_k = np.zeros(len(e), dtype=bool)
    e0 = e[0]
    prev_count = -1
    counts = []
    period0 = float(period)
    for _ in range(max_iters):
        k = np.round((e - e0) / period)
        good_k = np.abs((e - e0) - k * period) < period * tol_frac
        count = int(good_k.sum())
        counts.append(count)
        if count < 8 or count == prev_count:
            break
        prev_count = count
        kk, ee = k[good_k], e[good_k]
        A = np.vstack([kk, np.ones_like(kk)]).T
        period, _ = np.linalg.lstsq(A, ee, rcond=None)[0]
        period = float(period)
    # #region agent log
    _agent_log("G", "cvbs.py:_refine_period", "refine-period", {
        "period0": period0, "period": float(period),
        "dp": float(period) - period0, "counts": counts,
        "n_edges": int(len(e)), "good": int(good_k.sum()),
        "count_dropped": bool(counts and max(counts) > counts[-1]),
    })
    # #endregion
    return period, good_k
 
 
_drift_dbg_last = [0.0]   # тротлінг окремо від інших діагностичних принтів
 
 
def _log_drift_diag(e: np.ndarray, t0: float, period: float, n_lines: int,
                     tol_frac: float = 0.2, tag: str = ""):
    """ТИМЧАСОВА діагностика (02.09.2026): чи є СПРАВЖНІЙ дрейф періоду
    в межах кадру, а не просто шум вимірювання.
 
    Контекст: користувач повідомив, що "шер" на картинці не змінився
    після фіксу через `_refine_period()` (ітеративне МНК-уточнення
    ЄДИНОГО середнього періоду на весь кадр). Власна перевірка
    симуляцією показала дві речі:
 
    1) `_refine_period()` сама по собі іноді (не завжди) збігається
       до трохи зміщеного значення періоду, коли серед фронтів є
       сторонні (від вмісту картинки) — ітерації можуть "сповзати"
       вбік, а не строго покращуватись. Це вже підстраховано (див.
       історію функції) відбором НАЙКРАЩОЇ ітерації за кількістю
       узгоджених фронтів, а не сліпим використанням останньої.
 
    2) Спроба одразу піти далі й зробити "маховик" синхронізації
       (по-рядкове прилипання до реально знайдених фронтів замість
       екстраполяції за єдиним періодом) НЕ підтвердилась власною
       симуляцією — в змодельованих умовах (частка сторонніх фронтів,
       пропусків) вона іноді покращувала результат, а іноді робила
       гірше, залежно від параметрів, які точно не відомі для
       реального сигналу. Наосліп таке не переноситься в прод,
       особливо після "накосячила" — тож перш ніж пробувати
       по-рядковий маховик, тут просто ДРУКУЄТЬСЯ те, що насправді
       відбувається з реальним сигналом.
 
    Що саме друкує: медіанний залишок (реальний_фронт − прогноз за
    t0+k·period) окремо для першої й останньої чверті кадру. Якщо
    залишок помітно росте від початку до кінця кадру — це системний
    ДРЕЙФ (потрібен по-рядковий маховик або інший підхід); якщо
    залишок майже однаковий в обох чвертях — дрейфу нема, і "шер"
    треба шукати в чомусь іншому (наприклад, у визначенні t0/кадрової
    синхри, а не в period).
    """
    now = _time.time()
    if now - _drift_dbg_last[0] < 2.0:
        return
    k = np.round((e - t0) / period)
    resid = (e - t0) - k * period
    good = (np.abs(resid) < period * tol_frac) & (k >= 0) & (k < n_lines)
    if int(good.sum()) < 16:
        return
    kk, rr = k[good], resid[good]
    q = max(1, n_lines // 4)
    first = rr[kk < q]
    last = rr[kk >= n_lines - q]
    if len(first) < 3 or len(last) < 3:
        return
    _drift_dbg_last[0] = now
    shift = float(np.median(last) - np.median(first))
    verdict = "СХОЖЕ НА ДРЕЙФ" if abs(shift) > 0.05 * period else "без вираженого тренду"
    print(f"[drift{('/' + tag) if tag else ''}] залишок(t0,period): "
          f"1-ша чверть медіана={np.median(first):+.2f} (n={len(first)}), "
          f"остання чверть медіана={np.median(last):+.2f} (n={len(last)}), "
          f"зсув={shift:+.2f} відл із {period:.2f} — {verdict}", flush=True)
    # #region agent log
    _agent_log("A", "cvbs.py:_log_drift_diag", "h-sync residual vs t0+k*period", {
        "tag": tag, "period": float(period), "n_lines": int(n_lines),
        "t0": float(t0), "med_first": float(np.median(first)),
        "med_last": float(np.median(last)), "shift_samp": shift,
        "shift_frac_line": shift / period if period else None,
        "shear_px_if_width640": (shift / period * 640.0) if period else None,
        "n_first": int(len(first)), "n_last": int(len(last)),
        "n_good": int(good.sum()), "verdict": verdict,
    })
    # #endregion
 
 
def _row_correlation(luma: np.ndarray) -> float:
    """Медіанна кореляція сусідніх рядків готової картинки.
 
    Останній рубіж перевірки кадру, ПІСЛЯ того, як синхро-метрики
    (score/line_rate) вже визнали період прийнятним. Знахідка
    02.09.2026: після переходу на FFT+кратність score почав впевнено
    "зіходитись" (0.36-0.43, майже поріг) навіть там, де синхро-період
    насправді хибний — сталий сторонній тон близько 22-26кГц
    (найімовірніше завада, напр. від імпульсного блока живлення, а не
    справжня рядкова синхра 15625Гц) виявився достатньо СТАБІЛЬНИМ,
    щоб пройти перевірку на кратність фронтів, хоча синхрою не є.
    Результат — кадр, зібраний із самого шуму: рядки одна з одною не
    корелюють (картинка виглядає як суцільний "сніг" без жодної
    структури), хоча формально "score" високий.
 
    Справжнє відео дає стабільно високу кореляцію сусідніх рядків
    (картинка гладка по вертикалі); некорельований шум — близьку до
    нуля. Це незалежний від синхро-метрик сигнал, тому ловить саме
    цей клас хибних локів.
    """
    if luma.shape[0] < 3:
        return 1.0
    a = luma[:-1].astype(np.float32)
    b = luma[1:].astype(np.float32)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1)) + 1e-6
    corr = num / den
    return float(np.median(corr))
 
 
def _smooth(v: np.ndarray, fs: float, presmooth_us: float) -> np.ndarray:
    """Коротке бокскар-згладжування виключно для пошуку фронтів синхри.
 
    На слабших сигналах (SNR ~12-17дБ, ще нижче типового порогу, де
    досі спостерігали впевнений декод) поодинокі шумові викиди в
    демодульованому сигналі перетинають поріг `thr` частіше, ніж
    справжні синхроімпульси — звідси `line_rate`, що вилітає на
    порядки за межі PAL/NTSC (350000-5833333 Гц замість ~15625 Гц).
    Вікно навмисно вузьке — на порядок коротше за сам синхроімпульс
    (SYNC_US=4.7мкс) — тому фронт синхри не розмивається помітно, а
    поодинокі шумові піки шириною в один-два відліки притлумлюються.
    Застосовується лише до копії, за якою шукають фронти; яскравість
    рядка й далі семплюється з несмугованого сигналу, щоб не втрачати
    горизонтальну роздільність картинки.
    """
    win = max(1, int(round(presmooth_us * fs)))
    if win <= 1:
        return v
    return uniform_filter1d(v, size=win, mode="nearest").astype(np.float32)
 
 
def _attempt(v: np.ndarray, fs: float, width: int, max_lines: int,
             presmooth_us: float = 0.0, fft_hint: tuple[float | None, float] | None = None,
             min_row_corr: float = 0.5):
    """Одна спроба сліпого декодування за заданої полярності.
 
    Повертає (оцінка_якості, Frame|None, t0|None, причина). `t0` —
    локальний індекс використаного фронту кадрової синхри, потрібен
    викликачу, щоб засіяти DecodeState. `причина` — рядок діагностики
    для друку в decode(), на логіку не впливає. `presmooth_us` — див.
    `_smooth()`.
 
    Оцінка періоду рядка (02.09.2026, третя ітерація): період бере
    ВИКЛЮЧНО спектральна оцінка `_fft_line_rate()` (та сама ідея, що
    й у `demod.classify_video()` — усереднений Уелчем спектр, стійкий
    до джиттеру й неглибокого занурення синхроімпульсу). Резервного
    шляху через домінантний сусідній інтервал більше нема — прибрано
    того ж дня, коли з'ясувалось, що він ніколи не давав надійного
    результату на реальних даних (лише правдоподібно виглядаючий
    шум), див. коментар прямо над використанням `fft_prom` нижче.
    Якщо FFT не бачить впевненого піка (>8дБ) — `_attempt()` одразу
    звітує відсутність сигналу, а не вгадує. Обраний період потім
    ІТЕРАТИВНО перевіряється й уточнюється через кратність (`good_k`,
    МНК) — див. `_refine_period()` — доти, доки кількість фронтів, що
    узгоджуються з періодом на цілій кількості рядків, не перестане
    зростати. `score` — фінальна частка таких фронтів (коректно
    враховує і пропущені рядки, де синхроімпульс не перетнув поріг, і
    зайві фронти від вмісту картинки, що теж пірнає нижче порогу).
 
    `min_row_corr` — останній рубіж, ПІСЛЯ того, як период визнаний
    прийнятним: медіанна кореляція сусідніх рядків готової картинки
    (див. `_row_correlation()`). Ловить хибний лок на сторонній, але
    випадково стабільний тон (напр. завада ~22-26кГц), що проходить
    перевірку на кратність фронтів, хоча синхрою не є — score тоді
    формально високий, а картинка розсипається в некорельований шум.
 
    `fft_hint` — необов'язкова пара (line_rate_hz, prominence_db) від
    `_fft_line_rate()`, порахована один раз викликачем. FFT не
    залежить від полярності (`v` чи `-v` дають однаковий спектр
    потужності), тому `decode()` рахує її один раз замість двічі (по
    разу на кожен знак) — економія половини накладних витрат на FFT.
    Якщо не передано, `_attempt()` порахує сама (напр. при виклику
    напряму, поза `decode()`).
    """
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi - lo < 1e-9:
        return 0.0, None, None, "плаский сигнал (hi-lo<1e-9)"
    v = (v - lo) / (hi - lo)          # вершина синхри ≈ 0, білий ≈ 1
 
    thr = 0.18                        # між вершиною синхри і рівнем гасіння
    v_edge = _smooth(v, fs, presmooth_us) if presmooth_us > 0 else v
    below, edges = _sync_edges(v_edge, thr)
    if len(edges) < 24:
        return 0.0, None, None, f"мало фронтів синхри ({len(edges)}<24)"
 
    d = np.diff(edges).astype(np.float64)
    # Діагностика (не впливає на вибір періоду нижче): скільки сирих
    # інтервалів між СУСІДНІМИ фронтами взагалі потрапляють у діапазон
    # справжньої рядкової частоти PAL/NTSC (14-17.5кГц). Знахідка
    # 02.09.2026: тут зазвичай ~5-8%, не ~0% — тобто правильний період
    # присутній серед сирих даних, просто ніколи не домінує серед
    # СУСІДНІХ пар (звідси й перехід на FFT + кратність нижче).
    band_lo, band_hi = fs / 17500.0, fs / 14000.0
    in_band = int(((d > band_lo) & (d < band_hi)).sum())
 
    fft_rate, fft_prom = fft_hint if fft_hint is not None else _fft_line_rate(v, fs)
    # Знахідка 02.09.2026, третя ітерація: резервний шлях через
    # "домінантний сусідній інтервал" (старий, доFFT-ний підхід) тут
    # ПРИБРАНО остаточно. Причина не статистична, а емпірична — на
    # жодному реальному захопленні за всю цю сесію діагностики він не
    # дав надійного результату (score 0.13-0.18 навіть при SNR
    # 40-65дБ за classify_video(), див. розділ "Діагностика у смузі
    # PAL/NTSC=X/Y" в doc.md), а коли FFT/Уелч чесно не бачить
    # впевненого піка (сигналу для декодування або нема, або він надто
    # слабкий), цей резерв все одно "знаходив" якийсь домінантний сусідній
    # інтервал — просто найстійкіший шматок шуму — і видавав правдоподібно
    # виглядаючий, але безглуздий line_rate (десятки-сотні кГц), який лише
    # плутав діагностику й забивав консоль. Тепер, якщо FFT не впевнений,
    # `_attempt()` чесно каже "сигналу нема", замість вгадувати.
    if fft_rate is None or fft_prom <= 8.0:
        return 0.0, None, None, (f"немає впевненого піка рядкової частоти "
                f"(FFT-підйом={fft_prom:.1f}дБ<=8.0дБ"
                + (f", найближчий кандидат {fft_rate:.0f}Гц" if fft_rate else "")
                + f", у смузі PAL/NTSC={in_band}/{len(d)})")
    period = fs / fft_rate
    src = f"fft(підйом {fft_prom:.1f}дБ)"
 
    e = edges.astype(np.float64)
    period, good_k = _refine_period(e, period)
    score = float(good_k.mean())
    line_rate = fs / period
    if not (14000 < line_rate < 17500):
        return score, None, None, (f"рядкова поза діапазоном (score={score:.2f}, "
                f"line_rate={line_rate:.0f}Гц, період={period:.1f}відл, "
                f"джерело={src}, у смузі PAL/NTSC={in_band}/{len(d)})")
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
        return score, None, None, f"замало рядків (score={score:.2f}, n_lines={n_lines}<32)"
    _log_drift_diag(e, t0, period, n_lines, tag="сліпий")
    # #region agent log
    k64 = np.arange(n_lines, dtype=np.float64)
    last_f64 = float(t0 + k64[-1] * period)
    last_f32 = float(np.float32(t0) + np.float32(n_lines - 1) * np.float32(period))
    _agent_log("B", "cvbs.py:_attempt", "raster timebase", {
        "path": "blind", "fs": float(fs), "fft_rate": float(fft_rate),
        "period": float(period), "line_rate": float(line_rate),
        "score": float(score), "t0": float(t0), "n_lines": int(n_lines),
        "locked_vsync": bool(locked), "standard": standard,
        "float32_last_err_samp": last_f32 - last_f64,
        "len_v": int(len(v)),
    })
    # #endregion
 
    offs = np.linspace(a0, a1, width, dtype=np.float32)
    idx = (t0 + np.arange(n_lines, dtype=np.float32)[:, None] * period
           + offs[None, :])
    idx = np.clip(idx, 0, len(v) - 2)
    i0 = idx.astype(np.int32)
    fr = idx - i0
    samp = v[i0] * (1 - fr) + v[i0 + 1] * fr      # лінійна інтерполяція
 
    luma = np.clip((samp - 0.30) / 0.70, 0, 1)    # гасіння -> чорний
    luma_u8 = (luma * 255).astype(np.uint8)
    row_corr = _row_correlation(luma_u8)
    if row_corr < min_row_corr:
        return score, None, None, (f"кадр не корелює по рядках (corr={row_corr:.2f}"
                f"<{min_row_corr:.2f}) — ймовірно хибний синхро-період "
                f"(score={score:.2f}, line_rate={line_rate:.0f}Гц, джерело={src})")
    return score, Frame(
        luma=luma_u8,
        line_rate=line_rate, lines=n_lines,
        standard=standard, locked=locked), t0, ""
 
 
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
                      tol_frac: float = 0.55,
                      presmooth_us: float = 0.0,
                      min_row_corr: float = 0.5):
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
    vv_edge = _smooth(vv, fs, presmooth_us) if presmooth_us > 0 else vv
    window = vv_edge[lo:hi]
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
    if _time.time() - _drift_dbg_last[0] >= 2.0:
        below_full = vv_edge < thr
        edges_full = (np.flatnonzero(below_full[1:] & ~below_full[:-1]) + 1).astype(np.float64)
        _log_drift_diag(edges_full, t0, period, n_lines, tag="трекінг")
    # #region agent log
    k64 = np.arange(n_lines, dtype=np.float64)
    last_f64 = float(t0 + k64[-1] * period)
    last_f32 = float(np.float32(t0) + np.float32(n_lines - 1) * np.float32(period))
    _agent_log("C", "cvbs.py:_attempt_tracked", "tracked raster", {
        "path": "tracked", "period": float(period), "t0": float(t0),
        "t0_pred": float(local_t0_pred), "t0_err": float(t0 - local_t0_pred),
        "n_lines": int(n_lines), "standard": standard,
        "float32_last_err_samp": last_f32 - last_f64,
        "state_lost": int(state.lost), "abs_start": float(abs_start),
    })
    # #endregion
 
    offs = np.linspace(a0, a1, width, dtype=np.float32)
    idx = (t0 + np.arange(n_lines, dtype=np.float32)[:, None] * period
           + offs[None, :])
    idx = np.clip(idx, 0, len(vv) - 2)
    i0 = idx.astype(np.int32)
    fr = idx - i0
    samp = vv[i0] * (1 - fr) + vv[i0 + 1] * fr
 
    luma = np.clip((samp - 0.30) / 0.70, 0, 1)
    luma_u8 = (luma * 255).astype(np.uint8)
    # Той самий останній рубіж, що й у _attempt() (див. _row_correlation) —
    # захист від дрейфу трекінгу на сторонній стабільний тон, якщо сигнал
    # ослаб/зник посеред уже встановленого лока.
    if _row_correlation(luma_u8) < min_row_corr:
        return None
    frame = Frame(luma=luma_u8,
                  line_rate=fs / period, lines=n_lines,
                  standard=standard, locked=True)
    return frame, abs_start + t0
 
 
def decode(base: np.ndarray, fs: float, width: int = 640,
           max_lines: int = 288,
           state: DecodeState | None = None,
           abs_start: float = 0.0,
           presmooth_us: float = 0.0,
           min_score: float = 0.5,
           min_row_corr: float = 0.5) -> Frame | None:
    """Декодує напівкадр.
 
    Без `state` (або на першому виклику) — точнісінько як раніше:
    повний перебір полярності й сліпий пошук синхри. Якщо переданий
    `state` уже містить період з попереднього успішного кадру, спершу
    пробує трекінг у вузькому вікні; лише після трьох поспіль
    невдалих спроб трекінгу («lost») відкочується на сліпий пошук і
    засіює `state` заново.
 
    `presmooth_us` (0 = вимкнено, типово задається з video.presmooth_us
    у конфізі) — ширина бокскар-згладжування перед пошуком фронтів
    синхри, див. `_smooth()`. Потрібен на слабких сигналах, де шумові
    викиди дають хибні фронти і `line_rate` вилітає за межі PAL/NTSC.
 
    `min_score` (типово з video.sync_score_threshold, дефолт 0.5) —
    поріг якості сліпого пошуку. На слабких сигналах з presmooth_us
    оцінка іноді підповзає впритул до 0.5 (спостережено 0.44-0.50), не
    переростаючи його — вартий експерименту параметр, який можна
    тюнити без перезбирання файлу.
 
    `min_row_corr` (типово з video.min_row_corr у конфізі, дефолт 0.5)
    — див. `_row_correlation()`: останній рубіж ПІСЛЯ синхро-метрик,
    відкидає кадр, зібраний на сторонньому, але випадково стабільному
    тоні (не справжня рядкова синхра), навіть якщо `score`/`line_rate`
    формально пройшли.
    """
    if len(base) < int(fs * 0.02):        # менше 20 мс — нема сенсу
        return None
    v = base.astype(np.float32)
 
    if state is not None and state.period is not None and state.lost < 3:
        tracked = _attempt_tracked(v, fs, width, max_lines, state, abs_start,
                                    presmooth_us=presmooth_us, min_row_corr=min_row_corr)
        if tracked is not None:
            frame, abs_t0 = tracked
            state.abs_t0 = abs_t0
            state.lost = 0
            # #region agent log
            _agent_log("C", "cvbs.py:decode", "decode path", {
                "path": "tracked", "line_rate": float(frame.line_rate),
                "lines": int(frame.lines), "standard": frame.standard,
                "locked": bool(frame.locked), "abs_start": float(abs_start),
            })
            # #endregion
            return frame
        state.lost += 1
        # #region agent log
        _agent_log("C", "cvbs.py:decode", "track miss", {"lost": int(state.lost)})
        # #endregion
 
    # FFT-оцінка рядкової частоти не залежить від полярності (|FFT(v)|
    # == |FFT(-v)|, а percentile-нормалізація в _attempt() лише
    # відзеркалює AC-частину) — рахуємо один раз тут замість двічі
    # нижче (по разу на кожен знак), економлячи половину накладних
    # витрат на FFT (див. docstring _attempt()).
    lo, hi = np.percentile(v, [0.5, 99.5])
    fft_hint = (_fft_line_rate((v - lo) / (hi - lo), fs)
                if hi - lo > 1e-9 else (None, 0.0))
 
    best_s, best_f, best_t0, best_sign = 0.0, None, None, 1.0
    reasons = []
    for sign in (1.0, -1.0):
        s, f, t0, reason = _attempt(v * sign, fs, width, max_lines,
                                     presmooth_us=presmooth_us, fft_hint=fft_hint,
                                     min_row_corr=min_row_corr)
        reasons.append(f"sign={sign:+.0f}: {reason or f'ok, але якість замала (score={s:.2f})'}")
        if f is not None and s > best_s:
            best_s, best_f, best_t0, best_sign = s, f, t0, sign
    if best_s <= min_score:
        # Тротлимо: без цього при повторних невдалих спробах на
        # утримуваному каналі консоль заллє друком кожні ~100-200мс.
        now = _time.time()
        if now - _dbg_last[0] > 1.0:
            _dbg_last[0] = now
            print(f"[cvbs] сліпий пошук не зійшовся (поріг {min_score:.2f}): "
                  + "; ".join(reasons), flush=True)
        return None
 
    if state is not None:
        state.sign = best_sign
        state.period = fs / best_f.line_rate
        state.standard = best_f.standard
        state.abs_t0 = abs_start + best_t0
        state.lost = 0
    # #region agent log
    _agent_log("A", "cvbs.py:decode", "decode path", {
        "path": "blind", "score": float(best_s),
        "line_rate": float(best_f.line_rate), "lines": int(best_f.lines),
        "standard": best_f.standard, "locked": bool(best_f.locked),
        "sign": float(best_sign), "abs_start": float(abs_start),
    })
    # #endregion
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
 
 




