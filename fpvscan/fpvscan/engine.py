"""Рушій сканування.
 
Три режими роботи, які перемикає одна робоча нитка:
 
  SWEEP   — суцільний прохід 400 МГц … 6 ГГц кроками по смузі
            дискретизації. На кожному кроці — спектр, пошук зайнятих
            ділянок, груба відсіювання за шириною смуги.
  INSPECT — кандидат демодулюється на 2–3 мс і перевіряється на
            рядкову частоту. Це відсіює Wi-Fi, LTE та завади.
  LOCK    — утримання каналу: періодичні захоплення, декодування
            кадрів, віддача картинки в веб.
 
Уся важка арифметика — в цій нитці, asyncio її не блокує.
"""
from __future__ import annotations
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from queue import Queue, Empty
from .iqbuffer import IQRingBuffer
 
import numpy as np
 
from .bands import PRIORITY_BANDS, band_of, nearest_channel
from .dsp import spectrum, demod, cvbs
from . import paths
from .recorder import VideoRecorder, FfmpegMissing
 
 
@dataclass
class Detection:
    freq_hz: float
    bandwidth_hz: float
    snr_db: float
    standard: str = "?"
    confidence: float = 0.0
    channel: str | None = None
    band: str = "—"
    first_seen: float = 0.0
    last_seen: float = 0.0
    hits: int = 1
 
 
 
@dataclass
class EngineState:
    mode: str = "SWEEP"
    tuned_hz: float = 0.0
    sweep_pos_hz: float = 0.0
    sweeps_done: int = 0
    detections: dict[int, Detection] = field(default_factory=dict)
    lock_target: float | None = None
    auto: bool = False
    auto_until: float = 0.0
 
 
class Engine:
    def __init__(self, source, cfg: dict, events: Queue):
        self.src = source
        self.cfg = cfg
        self.events = events            # чим годуємо веб-сокети
        self.state = EngineState()
        self._stop = threading.Event()
        self._cmd: Queue = Queue()
        self._thread: threading.Thread | None = None
        self._rec: VideoRecorder | None = None
        self._snap = False
        self._peeked: dict[int, float] = {}   # частоти, які вже бачилися в цьому проході
        self._lock_tuned: float | None = None  # на що вже перебудовані
        self._ring: IQRingBuffer | None = None       # кільцевий буфер IQ для LOCK
        self._reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()
        self._reader_err: Exception | None = None     # помилка з нитки читання
        self._lock_state = None                        # cvbs.DecodeState | None
        self._lock_dec: int | None = None              # коефіцієнт децимації минулого виклику
        self._lock_n = 0
        self._acc: np.ndarray | None = None
        self._afc = 0.0
        self._hw_afc_offset = 0.0   # частина AFC, вже "запечена" в реальну
                                     # перебудову приймача (див. фікс 03.09.2026
                                     # бага _lock_tuned/channelize нижче)
        self._sweep_i = 0
        self._timings: dict[str, float] = {}   # ковзне середнє по етапах, мс
        self._frame_ts: float | None = None    # час минулого відданого кадру
        self._fps_ema = 0.0
        self._last_ch_bw = 0.0                 # для діагностики через /api/state
        self._last_afc_lim = 0.0
        self._logged_afc_lim: float | None = None   # щоб не спамити консоль щоцикл
        self._err_dbg_last = 0.0   # тротлінг діагностики сирого freq_error_hz()
        self._recenter_count = 0   # поспільні recenter без жодного кадру
        self._last_classify_dbg = 0.0   # тротлінг crosscheck-друку нижче
 
    # ---------- зовнішнє API ----------
 
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
 
    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
 
    def command(self, name: str, **kw):
        self._cmd.put((name, kw))
 
    # ---------- внутрішнє ----------
 
    def _mark(self, stage: str, t0: float) -> float:
        """Ковзне середнє часу етапу (мс). Легке — тільки арифметика,
        жодних додаткових захоплень чи алокацій на гарячому шляху."""
        t1 = time.perf_counter()
        dt_ms = (t1 - t0) * 1000
        prev = self._timings.get(stage)
        self._timings[stage] = dt_ms if prev is None else prev * 0.8 + dt_ms * 0.2
        return t1
 
    def _log_classify_crosscheck(self, base, fs_ch):
        """Незалежна перевірка на ТОМУ Ж capture, коли cvbs.decode() не
        зійшовся: чи бачить ``demod.classify_video()`` (окремий,
        встановлений детектор свіп/inspect-етапу — не той самий код, що
        `_fft_line_rate()`, і вже підтверджений зовнішнім сканером на
        реальному відео) хоч якийсь натяк на рядкову лінію 15625/15734Гц
        у цьому ж `base`.
 
        Знахідка 02.09.2026 (лист користувача "накосячила ти знатно
        ніхріна не працює"): після переходу `_fft_line_rate()` на Уелч і
        прибирання старого резервного шляху ``cvbs.decode()`` перестав
        видавати кадри взагалі — і це саме собою ще нічого не каже: чи
        Уелч-поріг тепер справедливо відкидає капчі БЕЗ реального відео,
        чи на стенді просто немає зараз активного передавача. Цей
        crosscheck дає пряму відповідь у наступному ж логу: якщо
        `classify_video()` теж мовчить (`is_video=False`,
        `confidence`≈0) — сигналу справді нема, питання до заліза/ефіру,
        не до `cvbs.py`. Якщо ж `classify_video()` впевнено бачить лінію
        (`is_video=True`) на тому самому `base`, а `cvbs.decode()` — ні,
        це вказує на розбіжність між двома детекторами (напр. надто
        короткий capture для Уелч-усереднення в `_fft_line_rate()`
        порівняно з тим, що бачить `classify_video()`), і треба звужувати
        далі саме там.
 
        Тротлиться так само, як і власний друк `cvbs.decode()` — інакше
        при утримуваному каналі без сигналу лог заллє дублями що ~100мс.
 
        Друга знахідка 02.09.2026 (лог одразу після прибирання
        резервного шляху): на кількох каналах (`4538.8МГц` двічі) сам
        `classify_video()` під час LOCK показує `confidence`≈0.00-0.01,
        хоча ТОЙ САМИЙ канал за кілька секунд до того на етапі INSPECT
        (окреме, чисте `retune_and_read()` без AFC/автогейну) дав
        `confidence` 0.77 при SNR 35.4дБ. Це вказує, що проблема — не в
        порозі `_fft_line_rate()`/`classify_video()`, а десь у самому
        шляху захоплення/підготовки сигналу під час LOCK (цифрова AFC-
        корекція `self._afc`, автопідбір gain у `_autolevel_gain()`,
        або щось у безперервному кільцевому читачі), яка гасить сигнал,
        що INSPECT ловить без проблем. Тому друк тепер несе й `afc`
        (поточний цифровий зсув), `gain` (поточний gain_db приймача) і
        `clip_frac` — щоб побачити, чи AFC зʼїхала на щось нетривіальне,
        чи gain після автопідбору сів значно нижче базового, чи є
        кліпування. Порівнювати варто з gain_db/bias_tee_gain_offset_db
        з конфігу (базовий рівень, який використовує і INSPECT) і з
        afc_lim_hz із сусіднього `[afc]`-рядка.
        """
        now = time.time()
        if now - self._last_classify_dbg < 1.0:
            return
        self._last_classify_dbg = now
        try:
            sc = self.cfg["scan"]
            score = demod.classify_video(
                base, fs_ch,
                tol_hz=float(sc.get("line_tol_hz", 150)),
                min_prominence_db=float(sc.get("line_prominence_db", 8)),
                min_conf=float(sc.get("min_confidence", 0.45)))
            gain = getattr(self.src, "gain_db", None)
            gain_s = f"{gain:.1f}дБ" if gain is not None else "?"
            print(f"[crosscheck] classify_video() на тому ж base: "
                  f"is_video={score.is_video} line_rate={score.line_rate:.0f}Гц "
                  f"standard={score.standard} confidence={score.confidence:.2f} "
                  f"prominence={score.prominence_db:.1f}дБ harmonics={score.harmonics} "
                  f"| afc={self._afc/1e3:.1f}кГц gain={gain_s} "
                  f"clip_frac={float(getattr(self.src, 'clip_frac', 0.0)):.4f}",
                  flush=True)
        except Exception as e:
            print(f"[crosscheck] classify_video() впав: {e}", flush=True)
 
    def _emit(self, kind: str, payload):
        """Кладе подію в чергу до веб-шару.
 
        Черга обмежена, і без підключеного клієнта вона забивається
        спектрами. Кадри при цьому губитись не мають, тому під них
        місце звільняється за рахунок найстаріших подій.
        """
        ev = {"type": kind, "data": payload}
        try:
            self.events.put_nowait(ev)
        except Exception:
            if kind != "frame":
                return
            for _ in range(8):
                try:
                    self.events.get_nowait()
                except Empty:
                    break
            try:
                self.events.put_nowait(ev)
            except Exception:
                pass
 
    def _drain_commands(self):
        while True:
            try:
                name, kw = self._cmd.get_nowait()
            except Empty:
                return
            try:
                self._handle_command(name, kw)
            except Exception as e:
                self._emit("notice", {"level": "error",
                                "text": f"команда «{name}»: {e}"})
                print(f"[рушій] команда «{name}» впала: {e}", flush=True)
 
    def _handle_command(self, name: str, kw: dict):
        if name == "lock":
            self._acc = None
            self._afc = 0.0
            self._hw_afc_offset = 0.0
            self._lock_tuned = None
            # fps/мітка кадру — з попереднього лока, інакше /api/state
            # бреше про поточну ціль (показує привид старої частоти).
            self._frame_ts = None
            self._fps_ema = 0.0
            self._recenter_count = 0
            req = float(kw["freq_hz"])
            # Знахідка 02.09.2026: якщо запит влучає в ту саму "МГц-
            # комірку" (той самий int(freq_hz/1e6)), що й уже знайдена
            # детекція в self.state.detections — беремо ТОЧНУ детековану
            # freq_hz замість того, що надіслала панель. Панель ідентифікує
            # канал саме за цим цілим МГц (те число, що показане в UI,
            # напр. "5010"), і схоже, що саме його й шле назад при виборі
            # каналу — тобто ручний вибір міг втрачати до ~0.5-1МГц
            # точності порівняно з автовідтворенням (`_maybe_auto_peek()`
            # нижче завжди бере `det.freq_hz` напряму, без округлення).
            # На межі "втягування" AFC (`freq_error_hz()` — percentile-
            # оцінка, надійна лише при відносно малій розстройці) така
            # різниця цілком може пояснювати, чому ручний вибір каналу
            # іноді зависає (AFC ніколи не сходиться), а вибір ТІЄЇ Ж
            # частоти під час активного автовідтворення — працює: останнє
            # стартує з точної частоти й ніколи не виходить за межі
            # надійного діапазону оцінки. Якщо збігу немає (панель і так
            # слала точне значення, або канал не з детекцій) — поведінка
            # не змінюється, це чистий safety net, не гадання.
            det = self.state.detections.get(int(req / 1e6))
            self.state.lock_target = det.freq_hz if det is not None else req
            self.state.mode = "LOCK"
            self.state.auto = False
        elif name == "sweep":
            self.state.lock_target = None
            self.state.mode = "SWEEP"
            self.state.auto = False
            self._stop_reader()
        elif name == "clear":
            self._peeked.clear()
            self._sweep_i = 0
            self.state.detections.clear()
        elif name == "snapshot":
            self._snap = True
        elif name == "rec_start":
            self._rec_start()
        elif name == "rec_stop":
            self._rec_stop()
        elif name == "bias_tee":
            on = bool(kw.get("on", True))
            if hasattr(self.src, "set_bias_tee"):
                ok = self.src.set_bias_tee(on)
                if ok:
                    self._apply_bias_tee_gain(on)
                self._emit("notice", {"level": "ok" if ok else "error", "text":
                           f"bias-tee: {'увімкнено' if on and ok else 'вимкнено' if ok else 'не підтримується платою'}"})
            else:
                self._emit("notice", {"level": "error", "text": "джерело не підтримує bias-tee"})
        if name in ("sweep", "lock") and self._rec is not None:
            self._rec_stop()      # ролик прив'язаний до одного каналу
        # ---------- фото і відео ----------
 
    def _rec_start(self):
        if self._rec is not None:
            return
        if self.state.mode != "LOCK":
            # У режимі свіпу кадрів немає — писати нічого.
            self._emit("notice", {"level": "error",
                                "text": "спершу стань на канал"})
            return
        v = self.cfg.get("video", {})
        name = paths.stamped("rec", "mp4", self.state.lock_target)
        try:
            r = VideoRecorder(paths.ensure(paths.VIDEO / name),
        int(v.get("width", 640)),
                            int(v.get("rec_height", 288)),
                            fps=int(v.get("rec_fps", 5)),
                            crf=int(v.get("rec_crf", 24)),
                            preset=str(v.get("rec_preset", "veryfast")),
                            exe=v.get("ffmpeg_path") or None)
            r.start()
        except FfmpegMissing as e:
            self._emit("notice", {"level": "error", "text": str(e)})
            return
        self._rec = r
        self._emit("notice", {"level": "ok", "text": f"запис: {name}"})
 
    def _rec_stop(self):
        if self._rec is None:
            return
        st = self._rec.stop()
        self._rec = None
        self._emit("notice", {"level": "ok", "text":
                   f"{Path(st['path']).name}: {st['bytes']/1e6:.2f} МБ, "
                   f"{st['seconds']} с, {st['kbps']} кбіт/с"})
 
    def _apply_bias_tee_gain(self, on: bool):
        """LNA на bias-tee додає власне підсилення поверх gain_db приймача.
 
        Без компенсації сумарний рівень сигналу заходить у кліп АЦП, і
        CVBS-декодер видає смуги/блоки замість картинки — саме це
        побачили на увімкненому LNA. Тому при увімкненні підрізаємо
        gain_db приймача на bias_tee_gain_offset_db (типово LNA дає
        ~15-20 дБ), а при вимкненні повертаємо як було.
        """
        base_gain = float(self.cfg["sdr"].get("gain_db", 30))
        offset = float(self.cfg["sdr"].get("bias_tee_gain_offset_db", 18))
        target = base_gain - offset if on else base_gain
        try:
            self.src.set_gain(target)
        except Exception as e:
            self._emit("notice", {"level": "error", "text": f"gain при bias-tee: {e}"})
 
    def _save_photo(self, frame):
        name = paths.stamped("shot", "webp", self.state.lock_target)
        dst = paths.ensure(paths.PHOTOS / name)
        dst.write_bytes(cvbs.encode(frame, "webp", 90, height=576))
        self._emit("notice", {"level": "ok", "text":
                   f"знімок: {name} ({dst.stat().st_size/1024:.1f} КБ)"})
 
    def _run(self):
        scan = self.cfg["scan"]
        fs = float(scan["sample_rate"])
        self.src.open()
        if getattr(self.src, "fixed_freq", False):
            fs = self.src.sample_rate         # запис диктує смугу
            self.cfg["scan"]["sample_rate"] = fs
            self.cfg["video"]["sample_rate"] = fs
        self.src.set_sample_rate(fs)
        self.src.set_gain(float(self.cfg["sdr"].get("gain_db", 30)))
        if hasattr(self.src, "set_bias_tee"):
            try:
                bt_on = bool(self.cfg["sdr"].get("bias_tee", False))
                self.src.set_bias_tee(bt_on)
                self._apply_bias_tee_gain(bt_on)
            except Exception as e:
                self._emit("notice", {"level": "error", "text": f"bias-tee: {e}"})
 
        if self.cfg["sdr"].get("quick_tune") and hasattr(self.src, "prime_quick_tune"):
            pts = self._sweep_plan()
            ok = self.src.prime_quick_tune(sorted(set(int(p) for p in pts)))
            print(f"[quick tune] знято профілів: {ok} з {len(set(map(int, pts)))}"
                  + ("" if ok else "  — плата/бібліотека не підтримує, працюємо звичайно"))
 
        try:
            fails = 0
            while not self._stop.is_set():
                self._drain_commands()
                try:
                    if self.state.mode == "LOCK" and self.state.lock_target:
                        self._do_lock()
                    else:
                        self._do_sweep()
                    fails = 0
                except Exception as e:
                    # Одиничний зрив USB не привід валити весь сервер:
                    # драйвер уміє перезапустити потік сам. Здаємось
                    # лише коли помилки йдуть підряд.
                    fails += 1
                    self._emit("notice", {"level": "error",
                                          "text": f"приймач: {e}"})
                    print(f"[рушій] помилка {fails}/8: {e}", flush=True)
                    if fails >= 8:
                        raise
                    # Пауза росте: якщо плата переперелічується на USB,
                    # їй потрібні секунди, а не мілісекунди.
                    time.sleep(min(5.0, 0.5 * fails))
        finally:
            self._stop_reader()
            self._rec_stop()
            self.src.close()
 
    # ---------- SWEEP ----------
 
    def _sweep_plan(self) -> list[float]:
        """Точки перебудови: суцільний прохід плюс повторний обхід
        пріоритетних діапазонів, щоб борти ловились швидше."""
        scan = self.cfg["scan"]
        fs = float(scan["sample_rate"])
        if getattr(self.src, "fixed_freq", False):
            return [self.src.center_freq]
 
        # Крок не можна брати рівним смузі: канал шириною 20 МГц, що
        # ліг на стик двох кроків, у кожному з них видно лише наполовину
        # і він може не набрати порогу. Тому від корисної смуги
        # віднімаємо половину очікуваної ширини каналу.
        ch_bw = float(scan.get("channel_bw_hz", 20e6))
        step = float(scan.get("step_hz", 0)) or max(fs * 0.25,
                                                    fs * 0.9 - ch_bw / 2)
        pts = list(np.arange(float(scan["start_hz"]) + fs / 2,
                             float(scan["stop_hz"]), step))
        if scan.get("priority_bands", True):
            prio = []
            for b in PRIORITY_BANDS:
                prio += list(np.arange(b.start_hz + fs / 2, b.stop_hz, step))
            # чергуємо: 1 крок суцільного скану на 1 пріоритетний
            merged = []
            for i, p in enumerate(pts):
                merged.append(p)
                if prio:
                    merged.append(prio[i % len(prio)])
            return merged
        return pts
 
    def _grab(self, f: float, n: int):
        fast = getattr(self.src, "retune_and_read_fast", None)
        return fast(f, n) if fast else self.src.retune_and_read(f, n)
 
    def _do_sweep(self):
        scan = self.cfg["scan"]
        fs = float(scan["sample_rate"])
        if not getattr(self.src, "fixed_freq", False) and \
            abs(self.src.sample_rate - fs) > 1.0:
            self.src.set_sample_rate(fs)
        # fs = float(scan["sample_rate"])
        nfft = int(scan.get("fft_size", 4096))
        avg = int(scan.get("averages", 8))
        need = nfft * avg
 
        plan = self._sweep_plan()
        if self._sweep_i >= len(plan):
            self._sweep_i = 0
            self.state.sweeps_done += 1
 
        while self._sweep_i < len(plan):
            if self._stop.is_set():
                return
            self._drain_commands()
            if self.state.mode == "LOCK":
                return
 
            f = plan[self._sweep_i]
            self._sweep_i += 1
            if self._stop.is_set():
                return
            self._drain_commands()
            if self.state.mode == "LOCK":
                return
 
            self._lock_tuned = None
            iq = self._grab(f, need)
            psd = spectrum.psd_db(iq, nfft, avg)
            self.state.tuned_hz = f
            self.state.sweep_pos_hz = f
 
            self._emit("spectrum", {
                "center_hz": f, "span_hz": fs,
                "bins": spectrum.downsample_for_display(psd, 384),
                "floor_db": round(spectrum.noise_floor_db(psd), 1),
            })
 
            occ = spectrum.find_occupied(
                psd, f, fs,
                threshold_db=float(scan.get("threshold_db", 8)),
                min_bw_hz=float(scan.get("min_bw_hz", 4e6)),
                dc_notch_hz=float(scan.get("dc_notch_hz", 200e3)))
 
            for o in occ:
                # Нижню межу тримаємо низько: реальний передавач на
                # стенді дав 4.3 МГц за порогом 6 дБ, і жорсткі 4 МГц
                # відкидали його при найменшій зміні шумової підлоги.
                if not (2.5e6 <= o.bandwidth_hz <= 35e6):
                    continue          # аналогове відео не буває вужчим/ширшим
                self._inspect(iq, f, fs, o)
                self._lock_tuned = None
 
 
 
    # ---------- INSPECT ----------
 
    def _inspect(self, _iq, center_hz, fs, occ):
        """Підтвердження кандидата за рядковою частотою.
 
        Свіповий буфер для цього закороткий: щоб побачити лінію
        15.7 кГц, потрібні десятки її періодів, тобто ~20+ мс ефіру.
        Тому тут робиться окреме, довше захоплення.
        """
        insp_s = float(self.cfg["scan"].get("inspect_ms", 25)) / 1000
        try:
            iq = self.src.retune_and_read(occ.center_hz, int(fs * insp_s))
            ch, fs2 = demod.channelize(
                iq, fs, 0.0,
                out_bw_hz=max(occ.bandwidth_hz  + 2 * self.MERGE_TOL_HZ, 8e6),
                fast=bool(self.cfg["scan"].get("fast_channelizer", False)))
            base = demod.fm_demod(ch, fs2, deviation_hz=occ.bandwidth_hz / 5)
            sc = self.cfg["scan"]
            score = demod.classify_video(
                base, fs2,
                tol_hz=float(sc.get("line_tol_hz", 150)),
                min_prominence_db=float(sc.get("line_prominence_db", 8)),
                min_conf=float(sc.get("min_confidence", 0.45)))
        except Exception as e:
            if self.cfg["scan"].get("debug_candidates"):
                self._emit("candidate", {"freq_hz": occ.center_hz,
            "reason": f"збій: {e}"})
            return
 
        # Кожен кандидат, що пройшов спектральний відбір, віддається
        # назовні разом із причиною рішення. Без цього неможливо
        # відрізнити «поріг завищений» від «сигналу немає».
        if self.cfg["scan"].get("debug_candidates"):
            self._emit("candidate", {
                "freq_hz": occ.center_hz,
                "bandwidth_hz": occ.bandwidth_hz,
                "snr_db": round(occ.snr_db, 1),
                "line_rate": round(score.line_rate, 1),
                "standard": score.standard,
                "confidence": score.confidence,
                "prominence_db": score.prominence_db,
                "harmonics": score.harmonics,
                "accepted": score.is_video,
                "reason": score.reason,
            })
        if not score.is_video:
            return
 
        now = time.time()
        det = Detection(
            freq_hz=occ.center_hz,
            bandwidth_hz=occ.bandwidth_hz,
            snr_db=round(occ.snr_db, 1),
            standard=score.standard,
            confidence=score.confidence,
            channel=nearest_channel(occ.center_hz),
            band=band_of(occ.center_hz),
            first_seen=now, last_seen=now,
        )
        self._merge(det)
 
    MERGE_TOL_HZ = 6e6
 
    def _merge(self, det: Detection):
        """Один передавач ловиться на кількох перекритих кроках свіпу
        і щоразу дає трохи інший центр. Зливаємо такі знахідки в одну,
        інакше список перетворюється на кашу з дублів."""
        for k, old in list(self.state.detections.items()):
            if abs(old.freq_hz - det.freq_hz) <= self.MERGE_TOL_HZ:
                det.first_seen = old.first_seen
                det.hits = old.hits + 1
                if old.snr_db > det.snr_db:      # тримаємось кращої оцінки
                    det.freq_hz = old.freq_hz
                    det.bandwidth_hz = old.bandwidth_hz
                    det.snr_db = old.snr_db
                del self.state.detections[k]
                break
        self.state.detections[int(det.freq_hz / 1e6)] = det
        # Одноразовий спалах у шумі не показуємо: справжній передавач
        # нікуди не подінеться і підтвердиться наступним проходом.
        if det.hits >= int(self.cfg["scan"].get("confirm_hits", 2)):
            self._emit("detection", asdict(det))
            self._maybe_peek(det)
 
    def _maybe_peek(self, det: Detection):
        """Автоматично зазирнути на щойно знайдений канал.
 
        Сенс режиму: оператор не має встигати клікати. Знайшли —
        показали кілька секунд картинки — пішли шукати далі. Ручне
        утримання це не чіпає: якщо на канал стали руками, свіп не
        відновлюється, доки не натиснуть «Сканувати».
        """
        sc = self.cfg["scan"]
        if not sc.get("auto_peek", True) or self.state.mode == "LOCK":
            return
        if det.confidence < float(sc.get("auto_peek_min_conf", 0.6)):
            return
        key = int(det.freq_hz / 1e6)
        now = time.time()
        # Не повертатись на той самий канал щопроходу.
        if now - self._peeked.get(key, 0) < float(sc.get("auto_peek_cooldown_s", 60)):
            return
        self._peeked[key] = now
        self._lock_tuned = None
        self.state.lock_target = det.freq_hz
        self.state.mode = "LOCK"
        self.state.auto = True
        self.state.auto_until = now + float(sc.get("auto_peek_secs", 8))
        self._emit("notice", {"level": "ok", "text":
                f"дивлюсь {det.freq_hz/1e6:.1f} МГц "
                f"({sc.get('auto_peek_secs', 8)} с)"})
 
    # ---------- LOCK ----------
 
    def _autolevel_gain(self, want: float, fs: float):
        """Автопідбір gain під конкретний сигнал перед стартом читача.
 
        Фіксований gain_db (навіть з підрізкою під Bias-T) підходить
        лише для сигналів середньої сили: борт на короткій дистанції
        все одно заганяє АЦП у кліп (бачили clip_frac 0.35 у /api/state
        під час "лок є, а відео не йде") — decode() тоді не знаходить
        синхру майже ніколи, і LOCK лише зрідка ловить випадковий кадр.
        Слабший сигнал, навпаки, недоотримує чутливості від тієї самої
        константи.
 
        Тому перед кожним новим читачем (новий канал або afc-recenter,
        усе йде через _start_reader) стартуємо з базового gain_db (з
        урахуванням Bias-T-офсету) і, якщо АЦП все одно клипить,
        підрізаємо gain ступінчасто, поки clip_frac не впаде нижче
        безпечного порогу. Калібрування не накопичується між локами —
        щоразу починається заново з базового рівня, інакше рушій рано
        чи пізно "з'їде" в мінімальний gain і застрягне там на слабких
        сигналах.
        """
        sc = self.cfg["sdr"]
        if not sc.get("gain_autolevel", True):
            return
        target = float(sc.get("clip_target", 0.02))
        step = float(sc.get("gain_step_db", 3.0))
        floor = float(sc.get("gain_floor_db", -10.0))
        max_steps = int(sc.get("gain_autolevel_steps", 6))
 
        bt_on = bool(getattr(self.src, "bias_tee", False))
        self._apply_bias_tee_gain(bt_on)     # старт завжди з базового рівня
 
        probe_n = max(4096, int(fs * 0.005))
        for _ in range(max_steps):
            try:
                self.src.retune_and_read(want, probe_n)
            except Exception:
                return      # хай далі це ловить звичайний шлях помилок
            clip = float(getattr(self.src, "clip_frac", 0.0))
            if clip <= target:
                return
            gain = float(getattr(self.src, "gain_db", 30.0)) - step
            if gain < floor:
                return
            try:
                self.src.set_gain(gain)
            except Exception:
                return
 
    def _lock_bw(self, freq_hz: float, default_bw: float) -> float:
        """Ширина каналу для утримання.
 
        Беремо зміряну під час свіпу — передавачі відрізняються
        девіацією в рази, і константа з конфігу тут або зріже сигнал,
        або впустить половину сусіднього діапазону.
 
        Маржа НЕ перевикористовує MERGE_TOL_HZ (це поріг злиття
        детекцій під час свіпу — 6МГц, зовсім інша мета). Побачили на
        реальних сигналах: додавання 2×MERGE_TOL_HZ (12МГц) до широких
        детекцій (bandwidth_hz 5-6+МГц) при fs=20МГц підганяло ch_bw
        впритул до fs і з'їдало майже весь safe_lim у _do_lock() —
        AFC лишалась або без запасу (постійні recenter, "стрибає
        підстройка"), або взагалі з safe_lim=0 (AFC вимкнена, дрейф
        несучої нічим не компенсується — "lock є, відео немає").
        Тепер окрема, менша маржа (`video.lock_bw_margin_hz`) плюс
        жорсткий стеля в частці fs — щоб один широкий сигнал не зжирав
        увесь запас Найквіста.
        """
        vcfg = self.cfg["video"]
        margin = float(vcfg.get("lock_bw_margin_hz", 3e6))
        cap = float(vcfg.get("sample_rate", 20e6)) * 0.7
        for d in self.state.detections.values():
            if abs(d.freq_hz - freq_hz) < 6e6:
                return min(cap, max(8e6, d.bandwidth_hz + 2 * margin))
        return min(cap, max(8e6, default_bw))
 
    def _start_reader(self, want: float, fs: float, ring_seconds: float):
        """Запускає нитку безперервного читання IQ у кільцевий буфер.
 
        Раніше кожен виклик _do_lock() сам читав IQ блоками: поки йде
        обробка попереднього блоку, приймач простоює, а наступний блок
        читається «з нуля» — звідси й ривки, і втрата фази синхри між
        блоками (п.1, п.2 розділу «Що далі» в технічному описі). Тепер
        приймач читає безперервно в окремій нитці, а _do_lock() лише
        бере знімки з кільцевого буфера — коли завгодно, без блокування
        приймача на час декодування.
        """
        self._stop_reader()
        self._autolevel_gain(want, fs)
        first = self.src.retune_and_read(want, max(2048, int(fs * 0.01)))
        self._ring = IQRingBuffer(capacity=max(len(first), int(fs * ring_seconds)))
        self._ring.write(first)
        self._reader_stop = threading.Event()
        self._reader_err = None
        self._reader_thread = threading.Thread(
            target=self._reader_loop, args=(fs,), daemon=True)
        self._reader_thread.start()
        self._lock_tuned = want
        self._lock_state = cvbs.DecodeState()
        self._lock_dec = None
 
    def _stop_reader(self):
        if self._reader_thread is not None:
            self._reader_stop.set()
            self._reader_thread.join(timeout=2)
        self._reader_thread = None
        self._ring = None
 
    def _reader_loop(self, fs: float):
        chunk = max(1024, int(fs * 0.005))     # 5 мс за раз
        while not self._reader_stop.is_set():
            try:
                iq = self.src.read(chunk)
            except Exception as e:
                self._reader_err = e
                return
            ring = self._ring
            if ring is None:
                return
            ring.write(iq)
 
    def _log_freq_err(self, f, base_iq, fs_ch, ch_bw, err, afc_on=True):
        """Тротлена діагностика розстройки (02.09.2026).
 
        Друкує СИРИЙ `freq_error_hz()` (до клампу й до накопичення в
        `self._afc`) поруч із тим, де сигнал стоїть НАСПРАВДІ —
        спектральний пік і центроїд потужності `base_iq`.
 
        Навіщо саме центроїд: у логах користувача `err` тримався
        -12.14МГц, сталий до ±9кГц, і НЕ змінювався після фізичної
        перебудови на 5.4-5.8МГц (справжня розстройка зсунулась би
        рівно на стільки). Власна симуляція показала, що перцентильна
        оцінка сама по собі такого не породжує: широкосмугове ЧМ-відео
        з девіацією до 20МГц у нефільтрованій смузі дає зміщення
        щонайбільше ±4.5МГц, і воно спадає до нуля з падінням SNR.
        Центроїд же на синтетиці з ВІДОМИМ зсувом відновлює його з
        похибкою <60кГц (перевірено на 0/+5.8/-12.14МГц), а пік
        зміщений на девіацію — тому читати треба центроїд, пік лише
        як довідку.
 
        Розрізняє однозначно: центроїд ≈0 при великому `err` — бреше
        оцінювач; центроїд ≈`err` — сигнал справді не там, де вважає код.
 
        Працює і при `video.afc=false` (там викликається з окремої
        гілки) — інакше вимкнення AFC заодно глушило б єдину
        діагностику, здатну перевірити гіпотезу «винна AFC».
        """
        if abs(err) <= 300e3:
            return
        self._err_dbg_last = time.time()
        try:
            m = min(len(base_iq), 1 << 15)
            sp = np.abs(np.fft.fftshift(np.fft.fft(base_iq[:m] * np.hanning(m)))) ** 2
            fr = np.fft.fftshift(np.fft.fftfreq(m, 1 / fs_ch))
            pk = float(fr[int(np.argmax(sp))])
            cen = float((fr * sp).sum() / max(sp.sum(), 1e-30))
            spec_s = (f"пік={pk/1e6:+.2f}МГц центроїд={cen/1e6:+.2f}МГц "
                      f"fs_ch={fs_ch/1e6:.1f}МГц")
        except Exception as ex:
            spec_s = f"спектр не порахувався: {ex}"
        print(f"[afc/err] {f/1e6:.1f}МГц: сирий freq_error_hz()={err/1e3:+.0f}кГц "
              f"(ch_bw={ch_bw/1e6:.1f}МГц, накопичено afc={self._afc/1e3:+.0f}кГц, "
              f"hw_afc_offset={self._hw_afc_offset/1e3:+.0f}кГц"
              + ("" if afc_on else ", AFC ВИМКНЕНА — лише вимірювання") + ") | "
              + spec_s, flush=True)
 
    def _do_lock(self):
        vcfg = self.cfg["video"]
        fs = float(vcfg.get("sample_rate", 20e6))
        f = self.state.lock_target
 
        if self.src.sample_rate != fs:
            self._stop_reader()          # нитка читала на старій fs — перезапуск
            self.src.set_sample_rate(fs)
 
        off = float(vcfg.get("lo_offset_hz", 0.0))
        bw = float(vcfg.get("channel_bw_hz", 20e6))
        if off and (abs(off) + bw / 2) > fs * 0.45:
            off = 0.0
 
        capture_s = float(vcfg.get("capture_ms", 120)) / 1000
        n_full = int(fs * capture_s)
        n = n_full
        if (self._lock_state is not None and self._lock_state.period is not None
                and self._lock_state.lost == 0 and self._lock_dec):
            margin = float(vcfg.get("track_window_margin", 1.7))
            field_lines = cvbs.FIELD_LINES.get(self._lock_state.standard,
                                                cvbs.FIELD_LINES["?"])
            n_track = int(self._lock_state.period * field_lines
                         * self._lock_dec * margin)
            n = max(int(fs * 0.02), min(n_full, n_track))
 
        ring_s = float(vcfg.get("ring_seconds", max(0.3, capture_s * 3)))
 
        if self._reader_err is not None:
            err, self._reader_err = self._reader_err, None
            self._stop_reader()
            raise err
 
        # Фізична перебудова приймача — лише на реальну зміну каналу
        # (off тут фіксований конфігом, БЕЗ self._afc). Раніше afc
        # входив прямо у want, і будь-яка, навіть мінімальна, AFC-
        # корекція перебудовувала приймач і скидала DecodeState() у
        # _start_reader() — фазове трекання губилося щоразу, коли VTx
        # (вільнонесучий генератор) трохи «пливе» по частоті, а він
        # пливе постійно. Тепер AFC компенсується нижче цифровим
        # зсувом каналайзера, а фізична перебудова лишається рідкісною
        # подією (див. afc_recenter_frac нижче).
        # Фікс 03.09.2026: `want` мусить включати вже "запечену" в залізо
        # частину AFC (self._hw_afc_offset), інакше цей сторож після
        # ПЕРШОГО ЖЕ recenter побачить розбіжність між номінальним f+off
        # і реально застосованою частотою й одразу ж перебудує приймач
        # НАЗАД на нескориговане f+off — звівши recenter нанівець.
        want = f + off + self._hw_afc_offset
        if self._ring is None or self._lock_tuned != want:
            self._start_reader(want, fs, ring_s)
 
        iq, abs_start_iq = self._ring.snapshot(n)
        if len(iq) < n:
            time.sleep(float(vcfg.get("idle_ms", 120)) / 1000)
            return
 
        t = time.perf_counter()
        self._lock_n += 1
 
        every = max(1, int(vcfg.get("spectrum_every", 8)))
        if self._lock_n % every == 1:
            psd = spectrum.psd_db(iq, 4096, 4)
            self._emit("spectrum", {
                "center_hz": f + off + self._hw_afc_offset, "span_hz": fs,
                "bins": spectrum.downsample_for_display(psd, 384),
                "floor_db": round(spectrum.noise_floor_db(psd), 1),
            })
 
        iq = iq - np.mean(iq)
 
        ch_bw = self._lock_bw(f, bw)
        # Цифрова AFC-корекція: зсуваємо вікно каналайзера на self._afc
        # замість перебудови приймача (див. коментар вище про want).
        #
        # ВАЖЛИВО (перевірено синтетичним тестом 03.09.2026, verify_fix_
        # synthetic.py): тут НЕ потрібно віднімати self._hw_afc_offset.
        # IQ з приймача завжди відносні до того, куди залізо ФІЗИЧНО
        # зараз налаштоване — а після recenter воно вже там, і
        # self._afc щоразу заново стартує з 0 та вимірює лише СВІЖИЙ
        # залишок відносно вже скоригованого гетеродина. Перша версія
        # цього фіксу помилково віднімала hw_afc_offset і тут теж — це
        # робило гірше (постійно повертало похибку до повного історичного
        # зсуву замість збіжності до нуля), підтверджено симуляцією.
        base_iq, fs_ch = demod.channelize(iq, fs, -off + self._afc, ch_bw)
        t = self._mark("channelize", t)
 
        dec = max(1, int(fs / ch_bw))
        if self._lock_dec != dec:
            self._lock_dec = dec
            self._lock_state = cvbs.DecodeState()
        abs_start_ch = abs_start_iq / dec
 
        if not vcfg.get("afc", True):
            # Знахідка 02.09.2026: діагностика мусить працювати і при
            # video.afc=false. Інакше виходить пастка — щоб перевірити
            # гіпотезу "AFC заважає", користувач вимикає AFC, а разом з
            # нею замовкає і єдиний друк, який міг би цю гіпотезу
            # підтвердити чи спростувати (весь блок був усередині
            # того самого if). Тут AFC не вмикається — лише вимірюється
            # й друкується те саме, що й у робочому режимі.
            if time.time() - self._err_dbg_last > 1.0:
                self._log_freq_err(f, base_iq, fs_ch, ch_bw,
                                    demod.freq_error_hz(base_iq, fs_ch),
                                    afc_on=False)
 
        if vcfg.get("afc", True):
            err = demod.freq_error_hz(base_iq, fs_ch)
            self._log_freq_err(f, base_iq, fs_ch, ch_bw, err, afc_on=True)
            lim = float(vcfg.get("afc_limit_hz", 8e6))
            # Цифровий зсув має лишатися в межах смуги Найквіста сирого
            # потоку разом із фіксованим off і половиною ch_bw — інакше
            # channelize() або накладеться сама на себе, або зачепить
            # сусідню ділянку спектра. Той самий запобіжник, що вже
            # стоїть для статичного off вище, тепер застосований і до
            # динамічної частини (afc).
            safe_lim = max(0.0, fs * 0.45 - abs(off) - ch_bw / 2)
            lim = min(lim, safe_lim)
            self._last_ch_bw = ch_bw
            self._last_afc_lim = lim
            if self._logged_afc_lim != round(lim):
                self._logged_afc_lim = round(lim)
                print(f"[afc] {f/1e6:.1f}МГц: ch_bw={ch_bw/1e6:.1f}МГц "
                      f"afc_lim={lim/1e3:.0f}кГц"
                      + ("  !! AFC де-факто вимкнена (lim=0)" if lim <= 0 else ""),
                      flush=True)
            dead = float(vcfg.get("afc_deadband_hz", 150e3))
            if abs(err) > dead:
                gain = float(vcfg.get("afc_gain", 0.5))
                self._afc = max(-lim, min(lim, self._afc + err * gain))
 
            # Накопичений цифровий дрейф час від часу варто «звільнити»
            # фізичною перебудовою — інакше при наближенні до ліміту
            # канал зʼїжджає до краю вікна каналайзера і SNR просідає.
            # Це той самий retune, що раніше був на кожному afc-кроці,
            # але тепер рідкісний, тож повʼязаний з ним скид
            # DecodeState() майже не заважає трекінгу (а якщо і
            # зачепить один кадр — lost-лічильник у cvbs.decode() сам
            # відновиться протягом ≤3 викликів).
            # Запобіжник: якщо recenter спрацьовує знову й знову без
            # жодного вдалого кадру між ними (типово на аномально
            # широких/зашумлених ділянках, де freq_error_hz() ловить
            # не зсув несучої, а сусідній сигнал чи шум і ніколи не
            # сходиться) — фізичні перебудови зупиняються, а цифрова
            # afc лишається затиснутою на межі. Інакше _start_reader()
            # щоцикл скидає DecodeState() і decode() ніколи не встигає
            # накопичити фазу. Лічильник скидається на будь-якому
            # успішно декодованому кадрі (нижче).
            recenter_max = int(vcfg.get("afc_recenter_max", 4))
            recenter_frac = float(vcfg.get("afc_recenter_frac", 0.75))
            want_recenter = ((lim > 0 and abs(self._afc) > lim * recenter_frac)
                              or (lim <= 0 and self._afc != 0.0))
            if want_recenter and self._recenter_count >= recenter_max:
                if self._recenter_count == recenter_max:
                    print(f"[afc] {f/1e6:.1f}МГц: {recenter_max} recenter поспіль "
                          f"без кадру — зупиняю фізичні перебудови (канал завеликий "
                          f"чи зашумлений, ch_bw={ch_bw/1e6:.1f}МГц)", flush=True)
                    self._recenter_count += 1   # щоб print не повторювався щоцикл
            elif lim > 0 and abs(self._afc) > lim * recenter_frac:
                self._recenter_count += 1
                print(f"[afc] recenter {f/1e6:.1f}МГц: afc={self._afc/1e3:.0f}кГц "
                      f"lim={lim/1e3:.0f}кГц ch_bw={ch_bw/1e6:.1f}МГц "
                      f"(#{self._recenter_count})", flush=True)
                # Фікс 03.09.2026: складаємо поточну цифрову корекцію в
                # ПОСТІЙНИЙ hw_afc_offset (а не відкидаємо її бухгалтерію
                # разом з обнулінням self._afc) і перебудовуємось на
                # реально потрібну частоту f+off+hw_afc_offset. НЕ
                # перезаписуємо self._lock_tuned значенням `want` зі
                # старого (до цього recenter) стану — _start_reader()
                # сам коректно виставляє self._lock_tuned у ту частоту,
                # на яку щойно фізично перебудувався приймач; старий
                # рядок `self._lock_tuned = want` це затирав і саме
                # через це приймач замерзав на зсунутій частоті до
                # кінця сесії LOCK (див. аналіз гілок main/test-branch).
                self._hw_afc_offset += self._afc
                self._start_reader(f + off + self._hw_afc_offset, fs, ring_s)
                self._afc = 0.0
            elif lim <= 0 and self._afc != 0.0:
                # Запасу цифрового зсуву взагалі нема (вузький fs чи
                # широкий канал) — одразу віддаємо корекцію в
                # перебудову, інакше clamp занулить afc і дрейф
                # перестане компенсуватись.
                self._recenter_count += 1
                print(f"[afc] lim<=0, форсована перебудова {f/1e6:.1f}МГц: "
                      f"afc={self._afc/1e3:.0f}кГц ch_bw={ch_bw/1e6:.1f}МГц "
                      f"(AFC де-факто вимкнена на цьому каналі, #{self._recenter_count})",
                      flush=True)
                # Той самий фікс 03.09.2026, що й у гілці вище.
                self._hw_afc_offset += self._afc
                self._start_reader(f + off + self._hw_afc_offset, fs, ring_s)
                self._afc = 0.0
        t = self._mark("afc", t)
 
        base = demod.fm_demod(base_iq, fs_ch, deviation_hz=ch_bw / 4)
        t = self._mark("demod_fm", t)
        base = demod.deemphasis(base, fs_ch)
        t = self._mark("deemphasis", t)
 
        frame = cvbs.decode(base, fs_ch, width=int(vcfg.get("width", 640)),
                            state=self._lock_state, abs_start=abs_start_ch + 1,
                            presmooth_us=float(vcfg.get("presmooth_us", 1.0e-6)),
                            min_score=float(vcfg.get("sync_score_threshold", 0.5)),
                            min_row_corr=float(vcfg.get("min_row_corr", 0.5)))
        t = self._mark("decode", t)
        if frame is None:
            self._log_classify_crosscheck(base, fs_ch)
 
 
        if frame is not None:
            min_lines = int(vcfg.get("min_lines", 250))
            if frame.lines < min_lines:
                frame = None
 
        if frame is not None:
            self._recenter_count = 0   # вдалий кадр — запобіжник знову "довіряє" AFC
            k = float(vcfg.get("average", 0.0))
            if k > 0:
                cur = frame.luma.astype(np.float32)
                if self._acc is None or self._acc.shape != cur.shape:
                    self._acc = cur
                else:
                    a = 1.0 / max(1.0, k)
                    self._acc = self._acc * (1 - a) + cur * a
                frame.luma = np.clip(self._acc, 0, 255).astype(np.uint8)
 
            if self._rec is not None:
                self._rec.push(frame.luma)
 
            if self._snap:
                self._snap = False
                self._save_photo(frame)
 
            img = cvbs.encode(frame, "webp", int(vcfg.get("stream_quality", 75)), height=None)
            t = self._mark("encode", t)
 
            self._emit("frame", {
                "freq_hz": f,
                "standard": frame.standard,
                "line_rate": round(frame.line_rate, 1),
                "lines": frame.lines,
                "locked": frame.locked,
                "afc_hz": round(self._afc, 0),
                "hw_afc_offset_hz": round(self._hw_afc_offset, 0),
                "img": img,
            })
 
            now = time.perf_counter()
            prev_ts = self._frame_ts
            self._frame_ts = now
            if prev_ts is not None:
                dt = now - prev_ts
                if dt > 0:
                    inst = 1.0 / dt
                    if self._fps_ema == 0:
                        self._fps_ema = inst
                    else:
                        self._fps_ema = self._fps_ema * 0.8 + inst * 0.2
 
        # шпаруватість: даємо процесору видихнути між знімками
        time.sleep(float(vcfg.get("idle_ms", 120)) / 1000)
 
        # Автоматичний перегляд обмежений у часі — далі шукаємо інших.
        if self.state.auto and time.time() >= self.state.auto_until:
            self.state.auto = False
            self.state.lock_target = None
            self.state.mode = "SWEEP"
            self._acc = None
            self._afc = 0.0
            self._hw_afc_offset = 0.0
            self._stop_reader()        # звільняємо src перед _do_sweep()
 
    def snapshot(self) -> dict:
        return {
            "mode": self.state.mode,
            "tuned_hz": self.state.tuned_hz,
            "sweeps_done": self.state.sweeps_done,
            "lock_target": self.state.lock_target,
            "auto": self.state.auto,
            "source": self.src.name,
            "recording": self._rec is not None,
            "bias_tee": bool(getattr(self.src, "bias_tee", False)),
            "rec_seconds": (round(time.time() - self._rec.started_at, 1)
            if self._rec else 0),
            "fps": round(self._fps_ema, 2),
            "timings_ms": {k: round(v, 1) for k, v in self._timings.items()},
            "overflows": int(getattr(self.src, "overflows", 0)),
            "clip_frac": round(float(getattr(self.src, "clip_frac", 0.0)), 4),
            "ch_bw_hz": round(self._last_ch_bw, 0),
            "afc_lim_hz": round(self._last_afc_lim, 0),
            "hw_afc_offset_hz": round(self._hw_afc_offset, 0),
            "detections": [asdict(d) for d in
                        sorted(self.state.detections.values(),
                        key=lambda x: -x.snr_db)
                        if d.hits >= int(self.cfg["scan"].get(
                        "confirm_hits", 2))],
        }
 
 
 
 
 
 
 

