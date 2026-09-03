"""ЧМ-демодуляція та розпізнавання аналогового відео.
 
Ключова ідея класифікатора: широка смуга сама по собі нічого не
доводить — так само виглядає Wi-Fi, LTE чи стрибуча завада. А от
рядкова розгортка дає у демодульованому сигналі дуже вузьку
спектральну лінію на 15 625 Гц (PAL) або 15 734 Гц (NTSC) з
десятками гармонік. Саме її ми й шукаємо: це майже безпомилкова
ознака аналогового композитного відео.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal
 
LINE_PAL = 15625.0
LINE_NTSC = 15734.264
 
 
def shift(iq: np.ndarray, offset_hz: float, fs: float) -> np.ndarray:
    """Переносить ділянку спектра на нуль."""
    if abs(offset_hz) < 1.0:
        return iq
    n = np.arange(len(iq), dtype=np.float32)
    lo = np.exp(-2j * np.pi * offset_hz * n / fs).astype(np.complex64)
    return iq * lo
 
 
def channelize(iq: np.ndarray, fs: float, offset_hz: float,
               out_bw_hz: float, fast: bool = True) -> tuple[np.ndarray, float]:
    """Зсув на нуль + децимація до потрібної смуги. Повертає (iq, fs_нов).
 
    Швидкий шлях — прямокутне усереднення через reshape+sum замість
    КІХ-фільтра. Його нулі стоять рівно на кратних новій частоті
    дискретизації, тобто саме там, звідки завертаються дзеркала.
    Пригнічення бічних пелюсток гірше, ніж у чесного фільтра, але
    для ЧМ-відео цього досить, а коштує воно на порядок дешевше.
    """
    x = shift(iq, offset_hz, fs)
    dec = max(1, int(fs / out_bw_hz))
    if dec == 1:
        return x.astype(np.complex64), fs
 
    if not fast:
        return (signal.decimate(x, dec, ftype="fir",
                                zero_phase=False).astype(np.complex64),
                fs / dec)
 
    n = (len(x) // dec) * dec
    if n == 0:
        return x.astype(np.complex64), fs
    # Два прямокутні вікна поспіль дають трикутне — удвічі крутіший
    # спад, майже задарма. Розкладаємо децимацію на два множники.
    d1 = int(np.sqrt(dec))
    while d1 > 1 and dec % d1:
        d1 -= 1
    d2 = dec // d1
    y = x[:n].reshape(-1, d1).sum(axis=1) if d1 > 1 else x[:n]
    if d2 > 1:
        m = (len(y) // d2) * d2
        y = y[:m].reshape(-1, d2).sum(axis=1)
    return (y / dec).astype(np.complex64), fs / dec
 
 
def fm_demod(iq: np.ndarray, fs: float, deviation_hz: float = 4e6) -> np.ndarray:
    """Квадратурний частотний дискримінатор.
 
    Найгарячіша петля всього проєкту. У numpy це arctan2 по всьому
    масиву; на Pi 5 саме звідси беруться основні мілісекунди.
    """
    d = iq[1:] * np.conj(iq[:-1])
    inst = np.arctan2(d.imag, d.real).astype(np.float32)
    return inst * (fs / (2 * np.pi * deviation_hz))
 
def inst_freq_hz(iq: np.ndarray, fs: float) -> np.ndarray:
    """Миттєва частота у герцах відносно нуля смуги."""
    d = iq[1:] * np.conj(iq[:-1])
    return (np.arctan2(d.imag, d.real).astype(np.float32) * (fs / (2 * np.pi)))
 
 
def freq_error_hz(iq: np.ndarray, fs: float) -> float:
    """Наскільки несуча зміщена від центру смуги.
 
    У ЧМ-відео миттєва частота гуляє між вершиною синхроімпульсу і
    рівнем білого. Середина цього розмаху і є центр девіації; якщо
    вона не на нулі — приймач стоїть збоку від каналу.
 
    Беремо саме перцентилі, а не середнє: середнє тягне за собою
    вміст картинки (темний кадр зсуває його не гірше за розстройку),
    а краї розмаху задані рівнями гасіння і білого, тобто самим
    стандартом.
    """
    f = inst_freq_hz(iq, fs)
    if f.size < 1024:
        return 0.0
    lo, hi = np.percentile(f[::4], [2.0, 98.0])
    return float((lo + hi) / 2.0)
 
def deemphasis(v: np.ndarray, fs: float, tau: float = 0.5e-6) -> np.ndarray:
    a = np.exp(-1.0 / (fs * tau))
    return signal.lfilter([1 - a], [1, -a], v).astype(np.float32)
 
 
@dataclass
class VideoScore:
    is_video: bool
    line_rate: float       # виміряна рядкова частота, Гц
    standard: str          # "PAL" | "NTSC" | "?"
    confidence: float      # 0..1
    prominence_db: float = 0.0   # наскільки лінія вища за локальний фон
    harmonics: int = 0           # скільки гармонік підтвердилось
    reason: str = ""          # 0..1
 
 
def classify_video(base: np.ndarray, fs: float,
                tol_hz: float = 150.0,
                min_prominence_db: float = 8.0,
                min_conf: float = 0.45) -> VideoScore:
    """Шукає рядкову лінію в спектрі демодульованого сигналу."""
    # Досить смуги до ~200 кГц — рядкова та кілька її гармонік.
    # Просте прорідження тут неприпустиме: воно завернуло б увесь
    # спектр яскравості на ту саму ділянку. Ставимо перед ним
    # ковзне середнє — дешевий ФНЧ з нулями кратно частоті прорідження.
    dec = max(1, int(fs / 400e3))
    x = base.astype(np.float32)
    if dec > 1:
        c = np.cumsum(np.concatenate(([0.0], x), dtype=np.float64))
        x = ((c[dec:] - c[:-dec]) / dec)[::dec].astype(np.float32)
    fs2 = fs / dec
    if len(x) < 8192:      # менше ~20 мс ефіру — рядкову не виміряти
        return VideoScore(False, 0.0, "?", 0.0)
 
    x = x - x.mean()
    nfft = 1 << int(np.floor(np.log2(len(x))))
    nfft = min(nfft, 1 << 16)
    w = np.hanning(nfft).astype(np.float32)
    sp = np.abs(np.fft.rfft(x[:nfft] * w)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1 / fs2)
 
    # шукаємо максимум у вікні 15.0–16.2 кГц
    sel = (freqs > 15.0e3) & (freqs < 16.2e3)
    if not np.any(sel):
        return VideoScore(False, 0.0, "?", 0.0)
    idx = np.where(sel)[0]
    peak_i = idx[np.argmax(sp[idx])]
    peak_f = float(freqs[peak_i])
    peak_p = float(sp[peak_i])
 
    # локальний фон: медіана в 10–25 кГц без самої лінії
    bg_sel = (freqs > 10e3) & (freqs < 25e3)
    bg = float(np.median(sp[bg_sel])) + 1e-20
    prominence_db = 10 * np.log10(peak_p / bg)
 
    # перевіряємо 2-у та 3-ю гармоніки — вони мають бути теж помітні
    harm = 0
    for h in (2, 3):
        hf = peak_f * h
        if hf >= freqs[-1]:
            break
        hi = int(hf / (fs2 / nfft))
        win = sp[max(0, hi - 3):hi + 4]
        if len(win) and 10 * np.log10(win.max() / bg) > 6:
            harm += 1
 
    std = "?"
    if abs(peak_f - LINE_PAL) < tol_hz:
        std = "PAL"
    elif abs(peak_f - LINE_NTSC) < tol_hz:
        std = "NTSC"
 
    conf = min(1.0, max(0.0,
               (prominence_db - min_prominence_db) / 18)) * (0.6 + 0.2 * harm)
    if std == "?":
        conf *= 0.5        # знижуємо, але не відкидаємо: буває нестандарт
 
    reason = ""
    if conf <= min_conf:
        reason = (f"впевненість {conf:.2f}: підйом {prominence_db:.1f} дБ, "
                f"гармонік {harm}, стандарт {std}")
    return VideoScore(conf > min_conf, peak_f, std, round(conf, 2),
                    round(prominence_db, 1), harm, reason)
 

