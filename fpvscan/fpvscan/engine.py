
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
    pic_score: float = 0.0      # score_picture — чим зливаємо, не SNR
    line_rate: float = 0.0
    row_corr: float = 0.0
    pic_locked: bool = False
 
 
 
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
        self._lock_gen = 0                         # покоління LOCK (скидає hunt)
        self._acc: np.ndarray | None = None
        self._afc = 0.0                            # лише цифровий зсув каналайзера
        self._last_err = 0.0
        self._hunt_th: threading.Thread | None = None
        self._hunt_out: float | None = None
        self._hunt_note: tuple[float, float, float] | None = None
        self._insp_dbg: list[dict] = []
        self._sweep_i = 0
        self._timings: dict[str, float] = {}   # ковзне середнє по етапах, мс
        self._frame_ts: float | None = None    # час минулого відданого кадру
        self._fps_ema = 0.0
 
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
            self._last_err = 0.0
            self._hunt_out = None
            self._hunt_note = None
            self._lock_tuned = None
            self._lock_n = 0
            self._lock_gen += 1
            self.state.lock_target = float(kw["freq_hz"])
            self.state.mode = "LOCK"
            self.state.auto = False
        elif name == "sweep":
            self.state.lock_target = None
            self.state.mode = "SWEEP"
            self.state.auto = False
            self._afc = 0.0
            self._lock_gen += 1
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
                # Нижню межу тримаємо ~4 МГц: реальний передавач на
                # стенді дав 4.3 МГц, а 2.7 МГц — типові шпори/гармоніки.
                if not (6.0e6 <= o.bandwidth_hz <= 35e6):
                    continue          # 4–5 МГц на 4 ГГц — шпори, не FPV
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
        sc = self.cfg["scan"]
        try:
            iq = self.src.retune_and_read(occ.center_hz, int(fs * insp_s))
            # Вікно класифікації = ширина зайнятості, з стелею inspect_bw.
            # Раніше додавали 2·MERGE_TOL (12 МГц) — шпора 12 МГц ставала
            # вікном 24 МГц і легше «бачила» рядкову лінію сусіда.
            insp_bw = float(sc.get("inspect_bw_hz", 12e6))
            out_bw = min(max(occ.bandwidth_hz, 8e6), insp_bw, fs * 0.9)
            ch, fs2 = demod.channelize(
                iq, fs, 0.0, out_bw_hz=out_bw,
                fast=bool(sc.get("fast_channelizer", False)))
            base = demod.fm_demod(ch, fs2, deviation_hz=occ.bandwidth_hz / 5)
            score = demod.classify_video(
                base, fs2,
                tol_hz=float(sc.get("line_tol_hz", 150)),
                min_prominence_db=float(sc.get("line_prominence_db", 8)),
                min_conf=float(sc.get("min_confidence", 0.45)),
                min_harmonics=int(sc.get("min_harmonics", 1)))
        except Exception as e:
            if self.cfg["scan"].get("debug_candidates"):
                self._emit("candidate", {"freq_hz": occ.center_hz,
            "reason": f"збій: {e}"})
            return

        accepted = score.is_video
        pic = None
        best_off = 0.0
        # Димова перевірка CVBS на кількох цифрових зсувах: центр
        # зайнятості часто є спідницею/шпорою, а картинка — на 1–2 МГц
        # осторонь. Обходу по spectral-впевненості немає (ним пролазив
        # 5018). Шум відсікає кореляція рядків, не «чи є кадрова».
        if accepted and sc.get("inspect_decode", True):
            ok, pic, best_off = self._inspect_offsets(iq, fs, out_bw, occ, sc)
            if not ok:
                accepted = False

        # Кожен кандидат, що пройшов спектральний відбір, віддається
        # назовні разом із причиною рішення. Без цього неможливо
        # відрізнити «поріг завищений» від «сигналу немає».
        if self.cfg["scan"].get("debug_candidates"):
            reason = score.reason
            if score.is_video and not accepted and pic is not None:
                reason = (f"димова: corr={pic.row_corr:.2f} "
                          f"locked={pic.locked} lines={pic.lines}")
            self._emit("candidate", {
                "freq_hz": occ.center_hz,
                "bandwidth_hz": occ.bandwidth_hz,
                "snr_db": round(occ.snr_db, 1),
                "line_rate": round(score.line_rate, 1),
                "standard": score.standard,
                "confidence": score.confidence,
                "prominence_db": score.prominence_db,
                "harmonics": score.harmonics,
                "accepted": accepted,
                "reason": reason,
                "row_corr": None if pic is None else round(pic.row_corr, 3),
                "pic_locked": None if pic is None else pic.locked,
            })
        if not accepted:
            self._note_insp(occ, score, pic, False)
            return
 
        now = time.time()
        det = Detection(
            freq_hz=occ.center_hz + best_off,
            bandwidth_hz=occ.bandwidth_hz,
            snr_db=round(occ.snr_db, 1),
            standard=score.standard,
            confidence=score.confidence,
            channel=nearest_channel(occ.center_hz),
            band=band_of(occ.center_hz),
            first_seen=now, last_seen=now,
            pic_score=0.0 if pic is None else round(pic.value, 3),
            line_rate=score.line_rate,
            row_corr=0.0 if pic is None else round(pic.row_corr, 3),
            pic_locked=False if pic is None else bool(pic.locked),
        )
        self._merge(det)

    def _decode_confirm(self, base: np.ndarray, fs: float):
        """Коротке cvbs.decode + score_picture для INSPECT.

        Повертає (чи схоже на аналогове відео, оцінка). Той самий
        рахунок, що й _freq_hunt — критерії не розходяться.
        """
        sc = self.cfg["scan"]
        vcfg = self.cfg.get("video", {})
        fr = cvbs.decode(demod.deemphasis(base, fs), fs,
                         width=int(vcfg.get("width", 640)),
                         state=None,
                         auto_levels=bool(vcfg.get("auto_levels", True)),
                         sharpen=0.0)
        pic = cvbs.score_picture(fr)
        if fr is None:
            return False, pic
        tol = float(sc.get("line_tol_hz", 150))
        sane = (abs(fr.line_rate - demod.LINE_PAL) <= tol
                or abs(fr.line_rate - demod.LINE_NTSC) <= tol)
        if not sane:
            return False, pic
        ok = pic.is_analog(
            min_corr=float(sc.get("inspect_min_row_corr", 0.12)),
            require_lock=bool(sc.get("inspect_require_lock", False)),
            min_lines=int(sc.get("inspect_min_lines", 80)))
        return ok, pic

    def _note_insp(self, occ, score, pic, accepted: bool):
        """Кілька останніх рішень INSPECT — щоб бачити, чому список порожній."""
        if occ.snr_db < 10 and not (score and score.is_video):
            return
        self._insp_dbg.append({
            "f": round(occ.center_hz / 1e6, 2),
            "bw": round(occ.bandwidth_hz / 1e6, 1),
            "snr": round(occ.snr_db, 1),
            "vid": bool(score.is_video),
            "std": score.standard,
            "conf": score.confidence,
            "acc": accepted,
            "corr": None if pic is None else round(pic.row_corr, 3),
            "lock": None if pic is None else pic.locked,
            "why": "" if accepted else (score.reason or "димова"),
        })
        self._insp_dbg = self._insp_dbg[-12:]

    def _inspect_offsets(self, iq, fs, out_bw, occ, sc):
        """Шукає відео-кращий цифровий зсув у вже знятому INSPECT-IQ.

        Повертає (ok, pic, offset_hz). Без повторної перебудови приймача.
        """
        trials = [0.0]
        for m in sc.get("inspect_offsets_mhz") or [1.0, 2.0]:
            hz = abs(float(m)) * 1e6
            trials.append(hz)
            trials.append(-hz)
        best_ok, best_pic, best_off = False, None, 0.0
        for mix in trials:
            ch, fs2 = demod.channelize(iq, fs, mix, out_bw_hz=out_bw)
            base = demod.fm_demod(ch, fs2, deviation_hz=max(occ.bandwidth_hz, 8e6) / 5)
            ok, pic = self._decode_confirm(base, fs2)
            if best_pic is None or pic.value > best_pic.value:
                best_ok, best_pic, best_off = ok, pic, mix
                if ok and pic.value >= 0.55:
                    break
        return best_ok, best_pic, best_off

    MERGE_TOL_HZ = 8e6

    def _same_tx(self, a: Detection, b: Detection) -> bool:
        """Один передавач vs два сусіди (Raceband ~19 МГц).

        Близькі центри — завжди одне. Далі, до merge_smear_hz, зливаємо
        лише якщо хоча б один хіт — слабка картинка (спідниця/шпора
        5018 поруч із живим 4988). Два живих відео поряд не чіпаємо.
        """
        dist = abs(a.freq_hz - b.freq_hz)
        if dist <= self.MERGE_TOL_HZ:
            return True
        smear = float(self.cfg["scan"].get("merge_smear_hz", 32e6))
        if dist > smear:
            return False
        # перекриття зайнятостей з запасом на крок свіпу
        if dist <= (a.bandwidth_hz + b.bandwidth_hz) * 0.5 + 8e6:
            return True
        weak = 0.40
        return (a.pic_score < weak) or (b.pic_score < weak)

    def _merge(self, det: Detection):
        """Один передавач ловиться на кількох перекритих кроках свіпу.

        Центр лишаємо відео-кращий (pic_score), не найсильніший SNR:
        інакше шпора 5018 (61 дБ) перебивала живий 4988.
        """
        for k, old in list(self.state.detections.items()):
            if not self._same_tx(old, det):
                continue
            det.first_seen = old.first_seen
            det.hits = old.hits + 1
            keep_old = False
            if old.pic_score > det.pic_score + 0.04:
                keep_old = True
            elif abs(old.pic_score - det.pic_score) <= 0.04:
                if old.confidence > det.confidence + 0.05:
                    keep_old = True
                elif abs(old.confidence - det.confidence) <= 0.05 and old.snr_db > det.snr_db:
                    keep_old = True
            if keep_old:
                det.freq_hz = old.freq_hz
                det.bandwidth_hz = old.bandwidth_hz
                det.snr_db = max(old.snr_db, det.snr_db)
                det.pic_score = max(old.pic_score, det.pic_score)
                det.confidence = max(old.confidence, det.confidence)
                det.line_rate = old.line_rate or det.line_rate
                det.row_corr = max(old.row_corr, det.row_corr)
                det.pic_locked = old.pic_locked or det.pic_locked
                det.standard = old.standard if old.pic_score >= det.pic_score else det.standard
                det.channel = old.channel or det.channel
                det.band = old.band or det.band
            else:
                det.snr_db = max(old.snr_db, det.snr_db)
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
        self._afc = 0.0
        self._lock_n = 0
        self._lock_gen += 1
        self.state.lock_target = det.freq_hz
        self.state.mode = "LOCK"
        self.state.auto = True
        self.state.auto_until = now + float(sc.get("auto_peek_secs", 8))
        self._emit("notice", {"level": "ok", "text":
                f"дивлюсь {det.freq_hz/1e6:.1f} МГц "
                f"({sc.get('auto_peek_secs', 8)} с)"})
 
    # ---------- LOCK ----------
 
    def _lock_bw(self, freq_hz: float, default_bw: float) -> float:
        """Ширина каналу для утримання.
 
        Беремо зміряну під час свіпу — передавачі відрізняються
        девіацією в рази, і константа з конфігу тут або зріже сигнал,
        або впустить половину сусіднього діапазону.
        """
        for d in self.state.detections.values():
            if abs(d.freq_hz - freq_hz) < 6e6:
                # виміряна смуга + невеликий запас на крила. Раніше
                # додавали 2·MERGE_TOL (12 МГц) — 6 МГц канал ставав
                # 18 МГц і скасовував децимацію.
                return max(8e6, d.bandwidth_hz + 1.5e6)
        return max(8e6, default_bw)
 
    def _start_reader(self, want: float, fs: float, ring_seconds: float,
                      keep_state: bool = False):
        """Запускає нитку безперервного читання IQ у кільцевий буфер.

        Раніше кожен виклик _do_lock() сам читав IQ блоками: поки йде
        обробка попереднього блоку, приймач простоює, а наступний блок
        читається «з нуля» — звідси й ривки, і втрата фази синхри між
        блоками. Тепер приймач читає безперервно в окремій нитці, а
        _do_lock() лише бере знімки з кільцевого буфера.

        keep_state: лишити період/полярність після перезапуску кільця.
        Цифрова AFC кільце не чіпає — DecodeState живе між кадрами сам.
        """
        old = self._lock_state if keep_state else None
        old_dec = self._lock_dec if keep_state else None
        self._stop_reader()
        first = self.src.retune_and_read(want, max(2048, int(fs * 0.01)))
        self._ring = IQRingBuffer(capacity=max(len(first), int(fs * ring_seconds)))
        self._ring.write(first)
        self._reader_stop = threading.Event()
        self._reader_err = None
        self._reader_thread = threading.Thread(
            target=self._reader_loop, args=(fs,), daemon=True)
        self._reader_thread.start()
        self._lock_tuned = want
        if old is not None and old.period is not None:
            old.abs_t0 = None
            old.lost = 3
            old.t0_err = None
            self._lock_state = old
            self._lock_dec = old_dec
        else:
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
 
    def _digital_afc_lim(self, fs: float, off: float, ch_bw: float,
                         vcfg: dict) -> float:
        """Стеля |цифрового AFC|: запас Найквіста. Далі — стоп, не RF."""
        lim = float(vcfg.get("afc_limit_hz", 20e6))
        nyq = max(0.0, fs * 0.45 - abs(off) - ch_bw / 2)
        return min(lim, nyq)

    def _video_ok_for_afc(self, frame: cvbs.Frame | None) -> bool:
        """AFC лише коли вже видно відео — на снігу freq_error бреше."""
        if frame is None or not frame.locked:
            return False
        return 14_000.0 < float(frame.line_rate) < 17_500.0

    def _apply_afc(self, base, fs: float, off: float, ch_bw: float,
                   deviation: float, vcfg: dict):
        """Щокадрова дешева AFC: лише цифровий зсув каналайзера.

        lock_target після «Стати» — якір оператора. Не складаємо AFC
        в якір і не перебудовуємо RF: freq_error (середина перцентилів
        ЧМ-відео) зміщена в бік синхри/спідниці і з'їжджала з картинки
        (3080→3075, той самий клас що 4989). На межі Найквіста — стоп.
        """
        err = demod.freq_error_from_demod(base, deviation)
        self._last_err = err
        dead = float(vcfg.get("afc_deadband_hz", 80e3))
        if abs(err) <= dead:
            return
        dig_lim = self._digital_afc_lim(fs, off, ch_bw, vcfg)
        if dig_lim < 50e3:
            return
        max_step = float(vcfg.get("afc_max_step_hz", 0.8e6))
        gain = float(vcfg.get("afc_gain", 0.7))
        step = float(np.clip(err * gain, -max_step, max_step))
        self._afc = max(-dig_lim, min(dig_lim, self._afc + step))

    def _take_hunt_result(self, fs: float, off: float, ch_bw: float):
        """Цифровий зсув з фонової нитки. lock_target — якір, не чіпаємо."""
        delta = self._hunt_out
        note = self._hunt_note
        self._hunt_out = None
        self._hunt_note = None
        if not delta:
            return
        vcfg = self.cfg["video"]
        safe = self._digital_afc_lim(fs, off, ch_bw, vcfg)
        self._afc = max(-safe, min(safe, self._afc + delta))
        if note and abs(delta) >= 0.25e6:
            d, cur_v, best_v = note
            self._emit("notice", {"level": "ok", "text":
                f"пошук частоти {d/1e6:+.2f} МГц "
                f"(оцінка {cur_v:.2f}→{best_v:.2f})"})

    def _kick_hunt(self, iq: np.ndarray, fs: float, off: float,
                   ch_bw: float, vcfg: dict, frame: cvbs.Frame | None):
        """Запустити цифровий пошук у фоні — не на гарячому кадрі.

        Повний перебір офсетів коштував 110–270 мс і вбивав fps.
        Тут копіюємо короткий зріз і рахуємо в іншій нитці; LOCK
        продовжує декодувати.
        """
        if not vcfg.get("hunt", True):
            return
        if frame is None:
            return
        if self._hunt_th is not None and self._hunt_th.is_alive():
            return
        skip = float(vcfg.get("hunt_skip_if_score", 0.60))
        if cvbs.score_picture(frame).value >= skip:
            return
        hunt_every = max(8, int(vcfg.get("hunt_every", 20)))
        if self._lock_n <= int(vcfg.get("hunt_after_lock", 16)):
            hunt_every = max(6, hunt_every // 2)
        if self._lock_n < 3 or self._lock_n % hunt_every != 0:
            return
        n_h = min(len(iq), max(int(fs * 0.018), 1))
        iq_h = np.array(iq[-n_h:], copy=True)
        mix0 = -off + self._afc
        gen = self._lock_gen
        self._hunt_th = threading.Thread(
            target=self._freq_hunt_worker,
            args=(gen, iq_h, fs, mix0, off, ch_bw, vcfg, self._last_err, self._afc),
            daemon=True, name="fpv-hunt")
        self._hunt_th.start()

    def _freq_hunt_worker(self, gen: int, iq_h: np.ndarray, fs: float,
                          mix0: float, off: float, ch_bw: float, vcfg: dict,
                          err: float, afc0: float):
        try:
            delta, cur_v, best_v = self._freq_hunt_compute(
                iq_h, fs, mix0, off, ch_bw, vcfg, err, afc0)
        except Exception:
            return
        if gen != self._lock_gen or not delta:
            return
        self._hunt_out = delta
        self._hunt_note = (delta, cur_v, best_v)

    def _freq_hunt_compute(self, iq_h: np.ndarray, fs: float, mix0: float,
                           off: float, ch_bw: float, vcfg: dict, err: float,
                           afc0: float):
        """Дрібний цифровий пошук (±0.25 МГц). lock_target не рухаємо.

        Лише якщо поточна оцінка слабка і кандидат явно кращий.
        """
        skip = float(vcfg.get("hunt_skip_if_score", 0.60))
        need = float(vcfg.get("hunt_min_gain", 0.10))
        offsets_mhz = vcfg.get("hunt_offsets_mhz") or [0.25]
        sign = 1.0 if err >= 0 else -1.0
        trials = []
        for m in offsets_mhz:
            hz = abs(float(m)) * 1e6
            trials.append(sign * hz)
        for m in offsets_mhz[:2]:
            trials.append(-sign * abs(float(m)) * 1e6)

        width = min(320, int(vcfg.get("width", 640)))
        auto_lv = bool(vcfg.get("auto_levels", True))
        safe = self._digital_afc_lim(fs, off, ch_bw, vcfg)

        def score_mix(mix_hz: float) -> cvbs.PictureScore:
            ch, fs_ch = demod.channelize(iq_h, fs, mix_hz, ch_bw)
            base = demod.fm_demod(ch, fs_ch, deviation_hz=ch_bw / 4)
            base = demod.deemphasis(base, fs_ch)
            fr = cvbs.decode(base, fs_ch, width=width, state=None,
                             auto_levels=auto_lv, sharpen=0.0)
            return cvbs.score_picture(fr)

        cur = score_mix(mix0)
        best_off = 0.0
        best = cur
        for trial in trials:
            if abs(afc0 + trial) > safe:
                continue
            pic = score_mix(mix0 + trial)
            if pic.value > best.value:
                best = pic
                best_off = trial
                if pic.value >= skip:
                    break
        if best_off != 0.0 and best.value >= cur.value + need:
            return best_off, cur.value, best.value
        return 0.0, cur.value, best.value

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
 
        # RF стоїть на якорі lock_target (клік / ±0.5 МГц). Цифровий
        # _afc у want не входить: його компенсує зсув каналайзера.
        # Складати AFC в lock_target — це 3080→3075.
        want = f + off
        if self._ring is None or self._lock_tuned != want:
            self._start_reader(want, fs, ring_s)
 
        iq, abs_start_iq = self._ring.snapshot(n)
        if len(iq) < n:
            # кільце ще не наповнилось — коротка пауза, не idle_ms
            # (той тепер 0 і інакше закрутить порожній цикл)
            time.sleep(0.004)
            return
 
        t = time.perf_counter()
        self._lock_n += 1
 
        every = max(1, int(vcfg.get("spectrum_every", 16)))
        did_spec = False
        if self._lock_n % every == 1:
            # LOCK-спектр лише індикація: короткий зріз, не повний знімок.
            nfft = 2048
            sl = iq[:nfft * 2] if len(iq) >= nfft * 2 else iq
            psd = spectrum.psd_db(sl, nfft, 2)
            self._emit("spectrum", {
                "center_hz": f + off, "span_hz": fs,
                "bins": spectrum.downsample_for_display(psd, 384),
                "floor_db": round(spectrum.noise_floor_db(psd), 1),
            })
            did_spec = True
 
        iq = iq - np.mean(iq)
 
        ch_bw = min(self._lock_bw(f, bw), fs * 0.9)
        # На ~20 Мвідл/с децимація 2× (вікно 8–10 МГц) ламає PAL-синхру
        # у decode(), хоча рядкова лінія в спектрі ще є. Тримаємо не
        # вужче 12 МГц — тоді dec=1 і поле збирається.
        ch_bw = max(ch_bw, min(12e6, fs * 0.9))
        # Запас Найквіста під цифрову AFC: інакше канал 12–18 МГц при
        # 20 Мвідл/с зануляє safe_lim і підстройка вічно стоїть на 0.
        headroom = float(vcfg.get("afc_digital_headroom_hz", 1.5e6))
        max_bw = 2.0 * max(6e6, fs * 0.45 - abs(off) - headroom)
        ch_bw = min(ch_bw, max_bw)
        ch_bw = max(ch_bw, min(12e6, fs * 0.9))
        # Цифрова AFC-корекція: зсуваємо вікно каналайзера на self._afc
        # замість перебудови приймача (див. коментар вище про want).
        base_iq, fs_ch = demod.channelize(iq, fs, -off + self._afc, ch_bw)
        t = self._mark("channelize", t)
 
        dec = max(1, int(fs / ch_bw))
        if self._lock_dec != dec:
            self._lock_dec = dec
            self._lock_state = cvbs.DecodeState()
        abs_start_ch = abs_start_iq / dec
 
        deviation = ch_bw / 4
        base = demod.fm_demod(base_iq, fs_ch, deviation_hz=deviation)
        t = self._mark("demod_fm", t)
        fm_base = base  # AFC міряє до деемфазису

        base = demod.deemphasis(base, fs_ch)
        t = self._mark("deemphasis", t)

        frame = cvbs.decode(base, fs_ch, width=int(vcfg.get("width", 640)),
                            state=self._lock_state, abs_start=abs_start_ch + 1,
                            auto_levels=bool(vcfg.get("auto_levels", True)),
                            sharpen=float(vcfg.get("sharpen", 0.0)))
        t = self._mark("decode", t)
 
        if frame is not None:
            # Поле рендериться до наступної кадрової синхри (один польовий
            # прохід), тож для NTSC це ~230-240 активних рядків, для
            # PAL — ~250-288. Поріг лишаємо низьким, щоб відсіювати лише
            # явний брак, а не коректні поля коротшого стандарту.
            min_lines = int(vcfg.get("min_lines", 200))
            if frame.lines < min_lines:
                frame = None

        self._take_hunt_result(fs, off, ch_bw)
        if vcfg.get("afc", True) and self._video_ok_for_afc(frame):
            self._apply_afc(fm_base, fs, off, ch_bw, deviation, vcfg)
        t = self._mark("afc", t)

        self._kick_hunt(iq, fs, off, ch_bw, vcfg, frame)

        if frame is not None:
            k = float(vcfg.get("average", 0.0))
            if k > 0:
                cur = frame.luma.astype(np.float32)
                if self._acc is None or self._acc.shape != cur.shape:
                    self._acc = cur
                else:
                    # Рухомо-адаптивне часове усереднення. Проста EMA
                    # рівномірно змішувала кадри й давала два артефакти,
                    # добре видні на реальних записах: змазування руху
                    # (рухомий об'єкт лишає «хвіст») і роздвоєння по
                    # вертикалі на нерухомому тексті (сусідні знімки —
                    # це протилежні поля черезрядкового відео, зсунуті на
                    # пів-рядка). Тому усереднюємо СИЛЬНО лише там, де
                    # кадри збігаються (нерухомий фон — виграш С/Ш), і
                    # майже не чіпаємо ділянки, що змінились (рух, краї
                    # тексту лишаються різкими).
                    a_static = 1.0 / max(1.0, k)
                    thr = float(vcfg.get("motion_thresh", 24.0))
                    diff = np.abs(cur - self._acc)
                    w = np.minimum(diff / max(1.0, thr), 1.0)   # 0 нерухомо .. 1 рух
                    a = a_static + w * (1.0 - a_static)
                    self._acc = self._acc * (1 - a) + cur * a
                frame.luma = np.clip(self._acc, 0, 255).astype(np.uint8)
 
            if self._rec is not None:
                self._rec.push(frame.luma)
 
            if self._snap:
                self._snap = False
                self._save_photo(frame)
 
            img = cvbs.encode(frame, str(vcfg.get("stream_fmt", "webp")),
                              int(vcfg.get("stream_quality", 75)),
                              height=None,
                              method=int(vcfg.get("stream_method", 0)))
            t = self._mark("encode", t)
 
            self._emit("frame", {
                "freq_hz": f,
                "standard": frame.standard,
                "line_rate": round(frame.line_rate, 1),
                "lines": frame.lines,
                "locked": frame.locked,
                "afc_hz": round(self._afc, 0),
                "freq_err_hz": round(self._last_err, 0),
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
 
        # шпаруватість: даємо процесору видихнути між знімками.
        # 0 — без штучної стелі fps (раніше 10 мс різали все, що вище ~15 к/с).
        idle = float(vcfg.get("idle_ms", 0))
        if idle > 0:
            time.sleep(idle / 1000)
 
        # Автоматичний перегляд обмежений у часі — далі шукаємо інших.
        if self.state.auto and time.time() >= self.state.auto_until:
            self.state.auto = False
            self.state.lock_target = None
            self.state.mode = "SWEEP"
            self._acc = None
            self._afc = 0.0
            self._lock_gen += 1
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
            "afc_hz": round(self._afc, 0),
            "freq_err_hz": round(self._last_err, 0),
            "lock_tuned_hz": self._lock_tuned,
            "inspect_dbg": list(self._insp_dbg),
            "detections": [asdict(d) for d in
                        sorted(self.state.detections.values(),
                        key=lambda x: -x.snr_db)
                        if d.hits >= int(self.cfg["scan"].get(
                        "confirm_hits", 2))],
        }
 


