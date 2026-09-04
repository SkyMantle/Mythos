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
    """Переносить ділянку спектра на нуль.

    Гетеродин рахуємо через `cos`/`sin` у float32, а не `np.exp` від
    комплексного аргументу. Скалярний множник `-2j*np.pi*...` у старому
    варіанті піднімав увесь масив фази до complex128, і `exp` по
    мільйонах відліків повного (недецимованого) потоку був найгарячішою
    ланкою всього тракту (≈90 мс на кадр проти ≈10 мс тепер).
    """
    if abs(offset_hz) < 1.0:
        return iq
    # Ціле-числовий фазовий акумулятор. Дробову частину фази тримаємо у
    # молодших 32 бітах int64: `n*K` рахується точно (без накопичення
    # похибки float на мільйонах відліків), а згортання періоду — це
    # просто маска `& 0xFFFFFFFF`. Виходить і швидше за float64-рампу, і
    # точніше (крок фази 2⁻³²). Далі `cos`/`sin` у float32 — цього досить.
    n = len(iq)
    k = (offset_hz / fs) % 1.0
    K = np.int64(round(k * (1 << 32)))
    phi = (np.arange(n, dtype=np.int64) * K) & np.int64(0xFFFFFFFF)
    ph = phi.astype(np.float32) * np.float32(2 * np.pi / (1 << 32))
    lo = np.empty(n, dtype=np.complex64)
    lo.real = np.cos(ph)
    lo.imag = -np.sin(ph)
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
        if x.dtype != np.complex64:
            x = x.astype(np.complex64)
        return x, fs

    if not fast:
        return (signal.decimate(x, dec, ftype="fir",
                                zero_phase=False).astype(np.complex64),
                fs / dec)

    n = (len(x) // dec) * dec
    if n == 0:
        return x.astype(np.complex64), fs
    # dec=2 одним прямокутником 2× майже не глушить дзеркало: на
    # 20 Мвідл/с → 10 МГц синхроімпульси розсипаються і decode()
    # втрачає PAL, хоча рядкова лінія в спектрі ще є. Тривідлікове
    # вікно + кожен 2-й — дешево і тримає фронти.
    if dec == 2:
        xx = x[:n].astype(np.complex64)
        acc = xx[:-2] + xx[1:-1] + xx[2:]
        return (acc[::2] / 3).astype(np.complex64), fs / 2
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


def freq_error_from_demod(base: np.ndarray, deviation_hz: float) -> float:
    """Зсув несучої з уже демодульованого ЧМ — без другого arctan2.

    `fm_demod` нормує миттєву частоту на deviation_hz, тож середина
    розмаху, помножена на deviation, дає герци. На живому LOCK це
    знімає ~20 мс, які раніше йшли на повторний inst_freq_hz.
    """
    if base.size < 1024:
        return 0.0
    lo, hi = np.percentile(base[::8], [2.0, 98.0])
    return float((lo + hi) / 2.0) * deviation_hz


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
    lo, hi = np.percentile(f[::8], [2.0, 98.0])
    return float((lo + hi) / 2.0)


def blob_offset_hz(iq: np.ndarray, fs: float,
                   dc_notch_hz: float = 300e3) -> float:
    """Центр енергії ЧМ-плями відносно DC — дешеве FFT, без демодуляції.

    На спідниці каналу ЧМ-похибка (перцентилі синхра/білий) бреше, а
    спектральна «пляма» все ще показує, куди зсувати вікно. На вже
    зведеному каналі ~0 означає, що AFC влучила в центр.
    """
    n = min(len(iq), 4096)
    n = 1 << int(np.floor(np.log2(max(n, 2))))
    if n < 256:
        return 0.0
    x = iq[:n]
    sp = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / fs))
    mask = np.abs(freqs) > dc_notch_hz
    if not np.any(mask):
        return 0.0
    w = sp[mask]
    s = float(w.sum())
    if s < 1e-20:
        return 0.0
    return float(np.dot(freqs[mask], w) / s)

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
                min_conf: float = 0.45,
                min_harmonics: int = 1) -> VideoScore:
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
        return VideoScore(False, 0.0, "?", 0.0, reason="закороткий буфер")

    x = x - x.mean()
    nfft = 1 << int(np.floor(np.log2(len(x))))
    nfft = min(nfft, 1 << 16)
    w = np.hanning(nfft).astype(np.float32)
    sp = np.abs(np.fft.rfft(x[:nfft] * w)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1 / fs2)

    # шукаємо максимум у вікні 15.0–16.2 кГц
    sel = (freqs > 15.0e3) & (freqs < 16.2e3)
    if not np.any(sel):
        return VideoScore(False, 0.0, "?", 0.0, reason="нема лінії 15–16 кГц")
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

    # Гармоніки обов'язкові: одиночний пік дає будь-яка вузька завада,
    # а гребінець із кратних частот — тільки рядкова розгортка.
    if harm < min_harmonics:
        return VideoScore(False, peak_f, std, round(conf, 2),
                          round(prominence_db, 1), harm,
                          f"гармонік {harm} < {min_harmonics}")
    if conf <= min_conf:
        return VideoScore(False, peak_f, std, round(conf, 2),
                          round(prominence_db, 1), harm,
                          f"впевненість {conf:.2f}: підйом {prominence_db:.1f} дБ, "
                          f"гармонік {harm}, стандарт {std}")
    return VideoScore(True, peak_f, std, round(conf, 2),
                      round(prominence_db, 1), harm, "")
