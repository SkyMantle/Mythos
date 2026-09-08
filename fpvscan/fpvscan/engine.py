
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
from .dsp.field_blend import blend_same_field
from . import auto_mgc, paths, scan_gate, scan_hits, scan_view
from .web.coalesce import enqueue_live_event
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
    pic_lines: int = 0
    last_picture_at: float = 0.0
 
 
 
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
        self._lock_score_peak = 0.0                # hunt лише коли оцінка впала
        self._insp_dbg: list[dict] = []
        self._sweep_i = 0
        self._timings: dict[str, float] = {}   # ковзне середнє по етапах, мс
        self._frame_ts: float | None = None    # час минулого відданого кадру
        self._fps_ema = 0.0
        self._last_video: dict | None = None
        self._last_frame_ref: str | None = None
        self._last_spectrum: dict | None = None
        self._recent_hz: list[float] = []
        self._acc_parity: int | None = None
        self._spec_ts: float | None = None
        self._spec_rate = 0.0
        self._next_hz: float | None = None
        self._dwell_ms = 0.0
        self._afc_pegged = False
        self._afc_nudge = False
        self._afc_peg_n = 0
        self._rf_snap_at = 0.0
        self._hunt_span = 0.25e6
        self._dense_q: list[float] = []
        self._dense_seen: set[int] = set()
        self._mgc_last_mono = 0.0
        self._mgc_hold_until = 0.0
        self._mgc_state = auto_mgc.MgcState()
        self._mgc_auto_was = True
        self._empty_drops: set[int] = set()
        self._empty_drop_saved: dict[int, Detection] = {}
        self._operator_lock_at: float = 0.0
        self._operator_lock_hit_mhz: int | None = None
 
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

        Черга обмежена. Кадр і спектр завжди лишають останній знімок
        (drop-to-latest); інакше LOCK-відео витісняє FFT і смуга замирає.
        """
        enqueue_live_event(self.events, {"type": kind, "data": payload})
 
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
            want = float(kw["freq_hz"])
            force = bool(kw.get("force"))
            # Never snap want onto a nearby published center: that made
            # ±0.1 MHz (= MERGE_CHANNEL_HZ) look like the same bird.
            target, skip = scan_hits.lock_retune(
                want,
                self.state.lock_target,
                force=force,
                locked=self.state.mode == "LOCK",
            )
            if skip:
                self._arm_operator_lock(self.state.lock_target)
                return
            self._acc = None
            self._acc_parity = None
            self._afc = 0.0
            self._last_err = 0.0
            self._hunt_out = None
            self._hunt_note = None
            self._lock_score_peak = 0.0
            self._lock_tuned = None
            self._lock_n = 0
            self._lock_gen += 1
            self._reset_auto_mgc()
            self._afc_pegged = False
            self._afc_nudge = False
            self._afc_peg_n = 0
            self._rf_snap_at = 0.0
            self.state.lock_target = target
            self.state.tuned_hz = target
            self.state.mode = "LOCK"
            self.state.auto = False
            self._arm_operator_lock(target)
            self._last_spectrum = {
                "bins": None,
                "center_hz": target,
                "span_hz": float(self.cfg.get("video", {}).get("sample_rate") or 0),
                "floor_db": None,
                "peak_db": None,
                "nfft": 2048,
            }
            self._emit("spectrum", self._spectrum_view())
            self._emit("state", self.snapshot())
        elif name == "sweep":
            self.state.lock_target = None
            self.state.mode = "SWEEP"
            self.state.auto = False
            self._afc = 0.0
            self._lock_gen += 1
            self._lock_score_peak = 0.0
            self._operator_lock_hit_mhz = None
            self._operator_lock_at = 0.0
            self._stop_reader()
        elif name == "clear":
            self._peeked.clear()
            self._sweep_i = 0
            self.state.detections.clear()
            self._empty_drops.clear()
            self._empty_drop_saved.clear()
            self._operator_lock_hit_mhz = None
            self._operator_lock_at = 0.0
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
        self._last_frame_ref = f"out/video/{name}"
        self._emit("notice", {"level": "ok", "text": f"запис: {name}"})
 
    def _rec_stop(self):
        if self._rec is None:
            return
        st = self._rec.stop()
        self._rec = None
        path = st.get("path")
        if path:
            self._last_frame_ref = f"out/video/{Path(path).name}"
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

    def hold_auto_mgc(self, seconds: float | None = None) -> None:
        """Pause software MGC after the operator moved the gain slider."""
        hold = auto_mgc.HOLD_S if seconds is None else float(seconds)
        if seconds is None:
            try:
                hold = float(self.cfg.get("sdr", {}).get("auto_gain_hold_s", auto_mgc.HOLD_S))
            except (TypeError, ValueError):
                hold = auto_mgc.HOLD_S
        self._mgc_hold_until = time.monotonic() + max(0.0, hold)

    def disable_auto_mgc(self) -> None:
        """Operator gain write: stop software MGC until they click авто."""
        self.cfg.setdefault("sdr", {})["auto_gain"] = False
        self._mgc_auto_was = False
        self._mgc_state = auto_mgc.MgcState()

    def reset_auto_mgc(self) -> None:
        """Resume hill-climb from the current catalog gain (авто on)."""
        self._reset_auto_mgc()
        self._mgc_auto_was = True

    def _reset_auto_mgc(self) -> None:
        self._mgc_state = auto_mgc.MgcState()
        self._mgc_last_mono = time.monotonic()

    def _maybe_auto_mgc(self, frame, iq=None) -> None:
        """LOCK software MGC: step catalog gain_db. BladeRF AGC stays off."""
        sdr = self.cfg.get("sdr") or {}
        enabled = bool(sdr.get("auto_gain", True))
        if enabled and not self._mgc_auto_was:
            self._reset_auto_mgc()
        self._mgc_auto_was = enabled
        if not enabled:
            return
        now = time.monotonic()
        try:
            interval = float(sdr.get("auto_gain_interval_s", auto_mgc.INTERVAL_S))
        except (TypeError, ValueError):
            interval = auto_mgc.INTERVAL_S
        if not auto_mgc.due(now, self._mgc_last_mono, interval):
            return
        self._mgc_last_mono = now
        pic = cvbs.score_picture(frame)
        luma = None if frame is None else getattr(frame, "luma", None)
        src_rms = float(getattr(self.src, "adc_rms", 0.0) or 0.0)
        sample = auto_mgc.MgcSample(
            gain_db=float(sdr.get("gain_db", 30)),
            pic_locked=bool(pic.locked),
            pic_score=float(pic.value),
            pic_lines=int(pic.lines),
            row_corr=float(pic.row_corr),
            clip_frac=float(getattr(self.src, "clip_frac", 0.0) or 0.0),
            operator_hold=now < self._mgc_hold_until,
            now_s=now,
            adc_rms=max(src_rms, auto_mgc.iq_rms(iq)),
            sat_frac=auto_mgc.luma_sat_frac(luma),
            frame_sig=auto_mgc.luma_signature(luma),
            min_db=float(sdr.get("auto_gain_min_db", auto_mgc.MIN_DB)),
            max_db=float(sdr.get("auto_gain_max_db", auto_mgc.MAX_DB)),
            step_db=float(sdr.get("auto_gain_step_db", auto_mgc.STEP_DB)),
            clip_thresh=float(sdr.get("auto_gain_clip_frac", auto_mgc.CLIP_FRAC)),
        )
        nxt = int(round(auto_mgc.step(sample, self._mgc_state)))
        if nxt == int(round(sample.gain_db)):
            return
        block = self.cfg.setdefault("sdr", {})
        block["gain_db"] = nxt
        self._apply_bias_tee_gain(bool(sdr.get("bias_tee", False)))
        self._emit("state", self.snapshot())
 
    def _save_photo(self, frame):
        name = paths.stamped("shot", "webp", self.state.lock_target)
        dst = paths.ensure(paths.PHOTOS / name)
        dst.write_bytes(cvbs.encode(frame, "webp", 90, height=576))
        self._last_frame_ref = f"out/photos/{name}"
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
        if getattr(self.src, "fixed_freq", False):
            return [self.src.center_freq]
        scan = self.cfg["scan"]
        prio = PRIORITY_BANDS if scan.get("priority_bands", True) else None
        return scan_view.sweep_centers(scan, priority_bands=prio)

    def _publish_spectrum(self, center_hz: float, span_hz: float,
                          bins: list, floor_db: float, nfft: int) -> None:
        peak = max(bins) if bins else None
        now = time.perf_counter()
        if self._spec_ts is not None:
            dt = now - self._spec_ts
            if dt > 0:
                inst = 1.0 / dt
                self._spec_rate = inst if self._spec_rate == 0 else (
                    self._spec_rate * 0.7 + inst * 0.3)
                self._dwell_ms = dt * 1000
        self._spec_ts = now
        self._last_spectrum = {
            "bins": bins,
            "center_hz": float(center_hz),
            "span_hz": float(span_hz),
            "floor_db": float(floor_db),
            "peak_db": None if peak is None else round(float(peak), 1),
            "nfft": int(nfft),
            "rate_hz": round(self._spec_rate, 2) if self._spec_rate else None,
            "t_mono_ms": scan_view.mono_ms(),
        }
        spec = self._spectrum_view()
        spec["grid"] = self._grid_view()
        self._emit("spectrum", spec)

    def _spectrum_view(self, t_mono_ms: int | None = None) -> dict:
        view = scan_view.spectrum_snapshot(
            self.cfg,
            mode=self.state.mode,
            lock_target=self.state.lock_target,
            tuned_hz=self.state.tuned_hz,
            last=self._last_spectrum,
            afc_hz=self._afc,
            next_hz=self._next_hz,
            dwell_ms=self._dwell_ms or None,
            t_mono_ms=t_mono_ms,
        )
        view["afc_pegged"] = bool(self._afc_pegged)
        view["freq_err_hz"] = round(self._last_err, 0)
        view["afc_hz"] = round(self._afc, 0)
        view["hunt_span_hz"] = float(self._hunt_span)
        return view

    def _grid_view(self, t_mono_ms: int | None = None) -> dict:
        scan = self.cfg.get("scan") or {}
        prio = PRIORITY_BANDS if scan.get("priority_bands", True) else None
        if getattr(self.src, "fixed_freq", False):
            prio = None
        return scan_view.grid_snapshot(
            self.cfg,
            mode=self.state.mode,
            sweep_i=self._sweep_i,
            sweeps_done=self.state.sweeps_done,
            sweep_pos_hz=self.state.sweep_pos_hz,
            tuned_hz=self.state.tuned_hz,
            lock_target=self.state.lock_target,
            visiting_hz=list(self._recent_hz),
            priority_bands=prio,
            plan_len=1 if getattr(self.src, "fixed_freq", False) else None,
            afc_hz=self._afc,
            next_hz=self._next_hz,
            dwell_ms=self._dwell_ms or None,
            t_mono_ms=t_mono_ms,
        )
 
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
        if self._sweep_i >= len(plan) and not self._dense_q:
            self._sweep_i = 0
            self.state.sweeps_done += 1
            self._dense_seen.clear()

        while self._dense_q or self._sweep_i < len(plan):
            if self._stop.is_set():
                return
            self._drain_commands()
            if self.state.mode == "LOCK":
                return

            from_extra = bool(self._dense_q)
            if self._dense_q:
                f = float(self._dense_q.pop(0))
            else:
                f = plan[self._sweep_i]
                self._sweep_i += 1
            if self._dense_q:
                self._next_hz = float(self._dense_q[0])
            elif self._sweep_i < len(plan):
                self._next_hz = float(plan[self._sweep_i])
            else:
                self._next_hz = float(plan[0]) if plan else None
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
            self._recent_hz.append(float(f))
            if len(self._recent_hz) > 8:
                self._recent_hz = self._recent_hz[-8:]
 
            self._publish_spectrum(
                f, fs, spectrum.downsample_for_display(psd, 384),
                round(spectrum.noise_floor_db(psd), 1), nfft,
            )
 
            occ = spectrum.find_occupied(
                psd, f, fs,
                threshold_db=float(scan.get("threshold_db", 8)),
                min_bw_hz=scan_gate.fft_min_bw_hz(scan),
                dc_notch_hz=float(scan.get("dc_notch_hz", 200e3)))

            for o in occ:
                self._queue_cluster_dense(o.center_hz, f, scan)
                line_hint = (
                    self._sweep_line_hint(iq, fs, f, o, scan)
                    if from_extra else True
                )
                if not scan_gate.should_full_inspect(
                    dwell_hz=f, center_hz=o.center_hz,
                    bandwidth_hz=o.bandwidth_hz, snr_db=o.snr_db,
                    from_extra=from_extra, scan=scan, line_hint=line_hint,
                ):
                    continue
                self._inspect(iq, f, fs, o)
                self._lock_tuned = None

    def _queue_cluster_dense(self, peak_hz: float, dwell_hz: float, scan: dict) -> None:
        """Insert 4 MHz extras around a cluster hit. No second full-band pass."""
        if len(self._dense_q) >= 48:
            return
        bucket = int(round(float(peak_hz) / 4.0e6))
        if bucket in self._dense_seen:
            return
        extras = scan_view.extras_for_hit(dwell_hz, peak_hz, scan)
        if not extras:
            return
        self._dense_seen.add(bucket)
        have = {int(round(h / 1e6)) for h in self._dense_q}
        have.add(int(round(float(dwell_hz) / 1e6)))
        for hz in extras:
            key = int(round(hz / 1e6))
            if key in have:
                continue
            have.add(key)
            self._dense_q.append(hz)
            if len(self._dense_q) >= 48:
                break
 
 
 
    # ---------- INSPECT ----------
 
    def _inspect(self, _iq, center_hz, fs, occ):
        """Підтвердження кандидата за рядковою частотою.
 
        Свіповий буфер для цього закороткий: щоб побачити лінію
        15.7 кГц, потрібні десятки її періодів, тобто ~20+ мс ефіру.
        Тому тут робиться окреме, довше захоплення.
        """
        # Same bins, fresh stamp so the scan playhead can lerp through inspect.
        stamp = scan_view.mono_ms()
        self._emit("spectrum", {
            **self._spectrum_view(t_mono_ms=stamp),
            "grid": self._grid_view(t_mono_ms=stamp),
        })
        sc = self.cfg["scan"]
        insp_s = scan_gate.inspect_ms(sc, occ.center_hz) / 1000
        try:
            iq = self.src.retune_and_read(occ.center_hz, int(fs * insp_s))
            # Вікно класифікації = ширина зайнятості, з стелею inspect_bw.
            # Раніше додавали 2·MERGE_TOL (12 МГц) — шпора 12 МГц ставала
            # вікном 24 МГц і легше «бачила» рядкову лінію сусіда.
            insp_bw = scan_gate.inspect_bw_hz(sc, occ.center_hz)
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
        # Димова перевірка — м'який фільтр, не veto всього списку.
        # Кадр на будь-якому цифровому зсуві = беремо. Інакше PAL/NTSC
        # у смузі ~5–15 МГц, але не 12 МГц пляма лише з енергії (5018).
        if accepted and sc.get("inspect_decode", True):
            ok, pic, best_off = self._inspect_offsets(iq, fs, out_bw, occ, sc)
            if ok:
                pass
            elif self._inspect_soft(score, occ, pic, sc):
                best_off = 0.0
            else:
                accepted = False
                best_off = 0.0

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
            pic_lines=0 if pic is None else int(pic.lines),
        )
        self._merge(det)

    def _decode_confirm(self, base: np.ndarray, fs: float):
        """Коротке cvbs.decode + score_picture для INSPECT.

        Повертає (чи схоже на аналогове відео, оцінка). Той самий
        рахунок, що й _freq_hunt — критерії не розходяться.
        """
        sc = self.cfg["scan"]
        vcfg = self.cfg.get("video", {})
        # вужче за LOCK: INSPECT лише питає «чи є картинка», не OSD
        fr = cvbs.decode(demod.deemphasis(base, fs), fs,
                         width=min(320, int(vcfg.get("width", 640))),
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

    def _sweep_line_hint(self, iq, fs, dwell_hz, occ, scan) -> bool:
        """Cheap 15.7 kHz comb on extra-dwell IQ. No retune, no decode."""
        try:
            mix = float(occ.center_hz) - float(dwell_hz)
            insp_bw = scan_gate.inspect_bw_hz(scan, occ.center_hz)
            out_bw = min(max(float(occ.bandwidth_hz), 8e6), insp_bw, fs * 0.9)
            ch, fs2 = demod.channelize(
                iq, fs, mix, out_bw_hz=out_bw,
                fast=bool(scan.get("fast_channelizer", False)))
            base = demod.fm_demod(
                ch, fs2, deviation_hz=max(float(occ.bandwidth_hz), 8e6) / 5)
            return demod.line_comb_hint(base, fs2)
        except Exception:
            return False

    def _inspect_soft(self, score, occ, pic, sc) -> bool:
        """Спектральний обхід, коли decode не зібрав кадр."""
        corr = 0.0 if pic is None else float(pic.row_corr)
        lines = 0 if pic is None else int(pic.lines)
        return scan_gate.inspect_soft_ok(
            standard=score.standard,
            bandwidth_hz=occ.bandwidth_hz,
            prominence_db=score.prominence_db,
            confidence=score.confidence,
            harmonics=score.harmonics,
            row_corr=corr,
            pic_lines=lines,
            scan=sc,
            center_hz=occ.center_hz,
        )

    def _inspect_offsets(self, iq, fs, out_bw, occ, sc):
        """Шукає відео на кількох цифрових зсувах у вже знятому IQ.

        Беремо ПЕРШИЙ зсув із живим кадром; сніг (високий score через
        кадрову без corr) не перебиває. Без повторної перебудови RF.
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
            if ok:
                if not best_ok or pic.value > best_pic.value:
                    best_ok, best_pic, best_off = True, pic, mix
                    if pic.value >= 0.40 or pic.row_corr >= 0.18:
                        break
                continue
            if not best_ok and (best_pic is None or pic.value > best_pic.value):
                best_pic, best_off = pic, mix
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
        if self._is_empty_dropped(det.freq_hz):
            if scan_hits.video_confirmed(asdict(det)):
                self._clear_empty_drop_near(det.freq_hz)
            else:
                return
        lock_hz = self.state.lock_target if self.state.mode == "LOCK" else None
        for k, old in list(self.state.detections.items()):
            if lock_hz is not None:
                old_lock = abs(old.freq_hz - lock_hz) <= scan_hits.MERGE_LOCK_HZ
                new_lock = abs(det.freq_hz - lock_hz) <= scan_hits.MERGE_LOCK_HZ
                if old_lock != new_lock:
                    continue
            lock_pair = (
                lock_hz is not None
                and abs(old.freq_hz - lock_hz) <= scan_hits.MERGE_LOCK_HZ
                and abs(det.freq_hz - lock_hz) <= scan_hits.MERGE_LOCK_HZ
            )
            if not lock_pair and not self._same_tx(old, det):
                continue
            det.first_seen = old.first_seen
            det.hits = old.hits + 1
            if lock_pair or abs(old.freq_hz - det.freq_hz) <= scan_hits.MERGE_CHANNEL_HZ:
                # Same bird: sticky published center. Update SNR/pic in place.
                det.freq_hz = scan_hits.sticky_published_hz(
                    old.freq_hz, det.freq_hz,
                    lock_target=self.state.lock_target,
                    locked=self.state.mode == "LOCK",
                )
                det.bandwidth_hz = old.bandwidth_hz
                det.snr_db = max(old.snr_db, det.snr_db)
                det.pic_score = max(old.pic_score, det.pic_score)
                det.confidence = max(old.confidence, det.confidence)
                det.line_rate = old.line_rate or det.line_rate
                det.row_corr = max(old.row_corr, det.row_corr)
                det.pic_locked = old.pic_locked or det.pic_locked
                det.pic_lines = max(old.pic_lines, det.pic_lines)
                det.last_picture_at = max(old.last_picture_at, det.last_picture_at)
                det.standard = old.standard if old.pic_score >= det.pic_score else det.standard
                det.channel = old.channel or det.channel
                det.band = old.band or det.band
            else:
                keep_old = False
                if old.pic_score > det.pic_score + 0.04:
                    keep_old = True
                elif abs(old.pic_score - det.pic_score) <= 0.04:
                    if old.confidence > det.confidence + 0.05:
                        keep_old = True
                    elif abs(old.confidence - det.confidence) <= 0.05 and old.snr_db > det.snr_db:
                        keep_old = True
                if keep_old:
                    det.freq_hz = scan_hits.sticky_published_hz(
                        old.freq_hz, det.freq_hz,
                        lock_target=self.state.lock_target,
                        locked=self.state.mode == "LOCK",
                    )
                    det.bandwidth_hz = old.bandwidth_hz
                    det.snr_db = max(old.snr_db, det.snr_db)
                    det.pic_score = max(old.pic_score, det.pic_score)
                    det.confidence = max(old.confidence, det.confidence)
                    det.line_rate = old.line_rate or det.line_rate
                    det.row_corr = max(old.row_corr, det.row_corr)
                    det.pic_locked = old.pic_locked or det.pic_locked
                    det.pic_lines = max(old.pic_lines, det.pic_lines)
                    det.last_picture_at = max(old.last_picture_at, det.last_picture_at)
                    det.standard = old.standard if old.pic_score >= det.pic_score else det.standard
                    det.channel = old.channel or det.channel
                    det.band = old.band or det.band
                else:
                    det.snr_db = max(old.snr_db, det.snr_db)
            del self.state.detections[k]
            key = k
            break
        else:
            if lock_hz is not None and abs(det.freq_hz - lock_hz) <= scan_hits.MERGE_LOCK_HZ:
                det.freq_hz = float(lock_hz)
            key = scan_hits.hit_key(det.freq_hz)
        self.state.detections[key] = det
        # Одноразовий спалах у шумі не показуємо: справжній передавач
        # нікуди не подінеться і підтвердиться наступним проходом.
        if det.hits >= int(self.cfg["scan"].get("confirm_hits", 2)):
            if any(scan_hits.same_channel(float(d["freq_hz"]), det.freq_hz)
                   for d in self._published_detections()):
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
        if not scan_gate.auto_peek_allowed(det, sc) or self.state.mode == "LOCK":
            return
        key = scan_hits.hit_key(det.freq_hz)
        now = time.time()
        # Не повертатись на той самий канал щопроходу.
        if now - self._peeked.get(key, 0) < float(sc.get("auto_peek_cooldown_s", 60)):
            return
        self._peeked[key] = now
        self._lock_tuned = None
        self._acc = None
        self._acc_parity = None
        self._afc = 0.0
        self._lock_n = 0
        self._lock_gen += 1
        self._reset_auto_mgc()
        self._lock_score_peak = 0.0
        self._operator_lock_hit_mhz = None
        self._operator_lock_at = 0.0
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
        cap = float(vcfg.get("afc_digital_max_hz", 1.5e6))
        nyq = max(0.0, fs * 0.45 - abs(off) - ch_bw / 2)
        return min(lim, nyq, cap)

    def _video_ok_for_afc(self, frame: cvbs.Frame | None) -> bool:
        """AFC лише коли вже видно відео — на снігу freq_error бреше."""
        if frame is None or not frame.locked:
            return False
        return 14_000.0 < float(frame.line_rate) < 17_500.0

    def _apply_afc(self, base, fs: float, off: float, ch_bw: float,
                   deviation: float, vcfg: dict, *,
                   pic_locked: bool = False):
        """Щокадрова дешева AFC: лише цифровий зсув каналайзера.

        lock_target після «Стати» — якір оператора. freq_error (середина
        перцентилів ЧМ-відео) зміщена в бік синхри/спідниці і з'їжджала
        з картинки (3080→3075). На межі Найквіста — стоп, не ±2 МГц hunt.
        Wide hunt / RF ±2 MHz freeze while `pic_locked` — they tear.
        Pegged + locked raster: `_maybe_rf_snap` once moves RF by digital
        `_afc` and zeros the mixer (hits/pin follow the new lock_target).
        """
        err = demod.freq_error_from_demod(base, deviation)
        self._last_err = err
        dead = float(vcfg.get("afc_deadband_hz", 80e3))
        dig_lim = self._digital_afc_lim(fs, off, ch_bw, vcfg)
        if abs(err) <= dead or dig_lim < 50e3:
            self._update_afc_peg(vcfg, dig_lim, pic_locked=pic_locked)
            return
        max_step = float(vcfg.get("afc_max_step_hz", 0.25e6))
        gain = float(vcfg.get("afc_gain", 0.5))
        step = float(np.clip(err * gain, -max_step, max_step))
        self._afc = max(-dig_lim, min(dig_lim, self._afc + step))
        self._update_afc_peg(vcfg, dig_lim, pic_locked=pic_locked)

    def _update_afc_peg(self, vcfg: dict, dig_lim: float, *,
                        pic_locked: bool = False) -> None:
        self._afc_pegged = scan_view.afc_is_pegged(self._afc, dig_lim)
        nudge = scan_view.afc_should_nudge(
            self._afc, self._last_err, dig_lim, pic_locked=pic_locked)
        self._afc_nudge = nudge
        offs = vcfg.get("hunt_offsets_mhz") or [0.25]
        self._hunt_span = scan_view.hunt_span_hz(offs, pegged=nudge)
        if nudge:
            self._afc_peg_n += 1
        else:
            self._afc_peg_n = 0
        need = max(4, int(vcfg.get("afc_peg_frames", 8)))
        if nudge and self._afc_peg_n >= need and self.state.lock_target:
            self._nudge_lock_for_peg()

    def _nudge_lock_for_peg(self) -> None:
        """RF step toward residual error when digital AFC cannot follow."""
        span = float(self._hunt_span or 2e6)
        delta = float(np.clip(self._last_err, -span, span))
        if abs(delta) < 80e3:
            return
        self.state.lock_target = float(self.state.lock_target) + delta
        self.state.tuned_hz = self.state.lock_target
        self._afc = 0.0
        self._lock_tuned = None
        self._lock_gen += 1
        self._afc_peg_n = 0
        self._afc_pegged = False
        self._afc_nudge = False
        self._emit("notice", {"level": "ok", "text":
            f"AFC у упорі — якір {delta/1e6:+.2f} МГц → "
            f"{self.state.lock_target/1e6:.2f}"})

    def _maybe_rf_snap(self, *, pic_locked: bool, pic_score: float,
                       digital_max_hz: float) -> bool:
        """Retune RF to lock_hz + digital AFC when the mixer is pegged.

        Only with a usable locked raster. Does not hunt ±2 MHz. Cooldown
        blocks every-frame repeats. Hits list / spectrum pin follow
        lock_target (MERGE_LOCK_HZ still 2 MHz).
        """
        if self.state.mode != "LOCK" or not self.state.lock_target:
            return False
        now = time.monotonic()
        vcfg = self.cfg.get("video") or {}
        cooldown = float(vcfg.get("rf_snap_cooldown_s", scan_view.RF_SNAP_COOLDOWN_S))
        if not scan_view.rf_snap_due(
                pic_locked=pic_locked, pic_score=pic_score,
                afc_hz=self._afc, digital_max_hz=digital_max_hz,
                last_snap_mono=self._rf_snap_at, now_mono=now,
                cooldown_s=cooldown):
            return False
        delta = float(self._afc)
        new_hz = float(self.state.lock_target) + delta
        self.state.lock_target = new_hz
        self.state.tuned_hz = new_hz
        self._afc = 0.0
        self._lock_tuned = None
        self._lock_gen += 1
        self._rf_snap_at = now
        self._afc_peg_n = 0
        self._afc_pegged = False
        self._afc_nudge = False
        self._emit("notice", {"level": "ok", "text":
            f"RF snap {delta/1e6:+.2f} МГц → {new_hz/1e6:.2f}"})
        self._emit("state", self.snapshot())
        return True

    def _take_hunt_result(self, fs: float, off: float, ch_bw: float, *,
                          freeze: bool = False):
        """Цифровий зсув з фонової нитки. lock_target — якір, не чіпаємо."""
        delta = self._hunt_out
        note = self._hunt_note
        self._hunt_out = None
        self._hunt_note = None
        if freeze or not delta:
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
        """Фоновий ±0.25 МГц, не на нитці decode.

        Лише коли картинка вже була і оцінка впала. Перші кадри LOCK
        не чіпаємо — інакше hunt краде BLAS і трекінг не засідається.
        """
        if not vcfg.get("hunt", True):
            return
        # сніг / немає кадру — не смикаємо частоту і не крадемо BLAS
        if frame is None or not frame.locked:
            return
        if self._hunt_th is not None and self._hunt_th.is_alive():
            return
        # перші кадри LOCK — лише трекінг; hunt_after_lock раніше
        # НАВПАКИ частішав пошук і садив fps до ~3
        wait = int(vcfg.get("hunt_after_lock", 24))
        if self._lock_n < max(8, wait):
            return
        pic = cvbs.score_picture(frame)
        skip = float(vcfg.get("hunt_skip_if_score", 0.70))
        hold = float(vcfg.get("hunt_hold_score", scan_view.HUNT_HOLD_SCORE))
        if scan_view.freeze_afc_hunt(
                pic_locked=bool(frame.locked), pic_score=float(pic.value),
                min_score=hold):
            return
        peak = self._lock_score_peak
        if pic.value > peak:
            self._lock_score_peak = pic.value
            peak = pic.value
        if pic.value >= skip:
            return
        drop = float(vcfg.get("hunt_drop", 0.12))
        if not self._afc_nudge and (peak < 0.35 or pic.value >= peak - drop):
            return
        hunt_every = max(12, int(vcfg.get("hunt_every", 20)))
        if self._afc_nudge:
            hunt_every = min(hunt_every, 8)
        if self._lock_n % hunt_every != 0:
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
        offsets_mhz = list(vcfg.get("hunt_offsets_mhz") or [0.25])
        lim = self._digital_afc_lim(fs, off, ch_bw, vcfg)
        pegged = scan_view.afc_should_nudge(afc0, err, lim)
        if pegged:
            for extra in (0.5, 1.0, 2.0):
                if extra not in offsets_mhz:
                    offsets_mhz.append(extra)
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
        if pegged:
            nyq = max(0.0, fs * 0.45 - abs(off) - ch_bw / 2)
            safe = max(safe, min(nyq, 2.5e6))

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
        if best_off != 0.0 and best.value >= cur.value + need and best.row_corr >= 0.10:
            return best_off, cur.value, best.value
        return 0.0, cur.value, best.value

    def _do_lock(self):
        vcfg = self.cfg["video"]
        fs = float(vcfg.get("sample_rate", 20e6))
        f = self.state.lock_target
 
        if abs(self.src.sample_rate - fs) > 1.0:
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
 
        # Live catalog: spectrum_every_4 → 4, else YAML spectrum_every (1).
        # Sweep occupancy FFT does not read this.
        every = scan_view.lock_spectrum_every(vcfg)
        if self._rec is not None:
            every = max(every, int(vcfg.get("rec_spectrum_every", 6)))
        did_spec = False
        if scan_view.lock_spectrum_due(self._lock_n, every):
            # LOCK-спектр лише індикація: короткий зріз, не повний знімок.
            nfft = 2048
            sl = iq[:nfft * 2] if len(iq) >= nfft * 2 else iq
            psd = spectrum.psd_db(sl, nfft, 2)
            self._publish_spectrum(
                f + off, fs, spectrum.downsample_for_display(psd, 384),
                round(spectrum.noise_floor_db(psd), 1), nfft,
            )
            did_spec = True
 
        iq = iq - np.mean(iq)
 
        # На ~20 Мвідл/с децимація 2× (вікно 8–10 МГц) ламає PAL-синхру
        # у decode(), хоча рядкова лінія в спектрі ще є. dec=2 заборонено:
        # піднімаємо channel_bw до fs, щоб dec лишався 1 (30 Мвідл/с +
        # 12 МГц IF інакше дає dec=2).
        headroom = float(vcfg.get("afc_digital_headroom_hz", 1.5e6))
        ch_bw = scan_view.lock_channel_bw(
            fs, self._lock_bw(f, bw),
            off_hz=off, headroom_hz=headroom)
        # Цифрова AFC-корекція: зсуваємо вікно каналайзера на self._afc
        # замість перебудови приймача (див. коментар вище про want).
        base_iq, fs_ch = demod.channelize(iq, fs, -off + self._afc, ch_bw)
        t = self._mark("channelize", t)
 
        dec = scan_view.lock_decimation(fs, ch_bw)
        # не обнуляти DecodeState щокадру: сліпий decode 56 мс ≈ 3 к/с
        if self._lock_dec is not None and self._lock_dec != dec:
            self._lock_state = cvbs.DecodeState()
        self._lock_dec = dec
        if self._lock_state is None:
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
                            sharpen=float(vcfg.get("sharpen", 0.0)),
                            h_phase_frac=float(vcfg.get("h_phase_frac", 0.0)),
                            crop_left_frac=float(vcfg.get(
                                "crop_left_frac", cvbs.CROP_LEFT_FRAC)),
                            crop_bottom_lines=int(vcfg.get(
                                "crop_bottom_lines", cvbs.CROP_BOTTOM_LINES)),
                            h_pll=bool(vcfg.get("h_pll", False)))
        t = self._mark("decode", t)
 
        if frame is not None:
            # Поле рендериться до наступної кадрової синхри (один польовий
            # прохід), тож для NTSC це ~230-240 активних рядків, для
            # PAL — ~250-288. Поріг лишаємо низьким, щоб відсіювати лише
            # явний брак, а не коректні поля коротшого стандарту.
            need = {"PAL": 240, "NTSC": 210, "?": 200}.get(frame.standard, 200)
            min_lines = int(vcfg.get("min_lines", need))
            if frame.lines < min_lines:
                frame = None

        pic = cvbs.score_picture(frame)
        hold = float(vcfg.get("hunt_hold_score", scan_view.HUNT_HOLD_SCORE))
        freeze_hunt = scan_view.freeze_afc_hunt(
            pic_locked=bool(pic.locked), pic_score=float(pic.value),
            min_score=hold)

        self._maybe_auto_mgc(frame, iq)
        self._take_hunt_result(fs, off, ch_bw, freeze=freeze_hunt)
        if vcfg.get("afc", True) and self._video_ok_for_afc(frame):
            self._apply_afc(fm_base, fs, off, ch_bw, deviation, vcfg,
                            pic_locked=freeze_hunt)
        self._maybe_rf_snap(
            pic_locked=bool(pic.locked), pic_score=float(pic.value),
            digital_max_hz=self._digital_afc_lim(fs, off, ch_bw, vcfg),
        )
        t = self._mark("afc", t)

        self._kick_hunt(iq, fs, off, ch_bw, vcfg, frame)

        if frame is not None:
            k = float(vcfg.get("average", 0.0))
            if k > 0:
                cur = frame.luma.astype(np.float32)
                blended, self._acc_parity = blend_same_field(
                    self._acc, self._acc_parity, cur, frame.field_parity,
                    k, float(vcfg.get("motion_thresh", 24.0)),
                )
                self._acc = blended
                frame.luma = np.clip(blended, 0, 255).astype(np.uint8)
 
            if self._rec is not None:
                self._rec.push(frame.luma)
 
            if self._snap:
                self._snap = False
                self._save_photo(frame)
 
            self._last_video = {
                "freq_hz": f,
                "standard": frame.standard,
                "line_rate": round(frame.line_rate, 1),
                "lines": frame.lines,
                "locked": frame.locked,
                "pic_score": round(pic.value, 3),
                "row_corr": round(pic.row_corr, 3),
                "afc_hz": round(self._afc, 0),
                "freq_err_hz": round(self._last_err, 0),
            }
            # Full WS queue already drop-to-latest; skip an encode that would be dumped.
            # JPEG is ~10× faster than WebP on noisy luma; console sniffs MIME.
            if not (getattr(self.events, "maxsize", 0) and self.events.full()):
                img = cvbs.encode(frame, str(vcfg.get("stream_fmt", "jpeg")),
                                  int(vcfg.get("stream_quality", 75)),
                                  height=None,
                                  method=int(vcfg.get("stream_method", 0)))
                self._emit("frame", {**self._last_video, "img": img})
                self._touch_lock_picture(f, frame, pic)
            t = self._mark("encode", t)

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

        self._maybe_prune_empty_lock(frame, pic)

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
            self._acc_parity = None
            self._afc = 0.0
            self._lock_gen += 1
            self._lock_score_peak = 0.0
            self._stop_reader()        # звільняємо src перед _do_sweep()
 
    def refresh_lock(self) -> None:
        """Re-arm the LOCK reader/decoder on the same target.

        Used after a live cfg write so sample_rate / LO / capture / decode
        knobs take effect on the next IQ. Does not stop a recording and
        does not bounce through sweep.
        """
        if self.state.mode != "LOCK" or not self.state.lock_target:
            return
        self._acc = None
        self._acc_parity = None
        self._lock_state = None
        self._lock_dec = None
        self._lock_tuned = None
        self._lock_n = 0
        self._lock_gen += 1
        self._reset_auto_mgc()
        self._afc = 0.0
        self._last_err = 0.0
        self._afc_pegged = False
        self._afc_nudge = False
        self._afc_peg_n = 0
        self._rf_snap_at = 0.0
        self._hunt_out = None
        self._hunt_note = None
        if abs(self.src.sample_rate - float(self.cfg.get("video", {}).get("sample_rate") or 0)) > 1.0:
            self._stop_reader()

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
            "gain_db": float((self.cfg.get("sdr") or {}).get("gain_db", 0)),
            "auto_gain": bool((self.cfg.get("sdr") or {}).get("auto_gain", True)),
            "rec_seconds": (round(time.time() - self._rec.started_at, 1)
            if self._rec else 0),
            "fps": round(self._fps_ema, 2),
            "timings_ms": {k: round(v, 1) for k, v in self._timings.items()},
            "overflows": int(getattr(self.src, "overflows", 0)),
            "clip_frac": round(float(getattr(self.src, "clip_frac", 0.0)), 4),
            "adc_rms": round(float(getattr(self.src, "adc_rms", 0.0) or 0.0), 4),
            "afc_hz": round(self._afc, 0),
            "freq_err_hz": round(self._last_err, 0),
            "afc_pegged": bool(self._afc_pegged),
            "hunt_span_hz": float(self._hunt_span),
            "lock_tuned_hz": self._lock_tuned,
            "video": self._last_video,
            "last_frame_ref": self._last_frame_ref,
            "sweep_pos_hz": self.state.sweep_pos_hz,
            "spectrum": self._spectrum_view(),
            "grid": self._grid_view(),
            "inspect_dbg": list(self._insp_dbg),
            "detections": self._published_detections(),
        }

    def reset_sweep_plan(self) -> None:
        """Drop queued cluster extras so the next dwell uses the new step."""
        self._dense_q.clear()
        self._dense_seen.clear()

    def _published_detections(self) -> list[dict]:
        need = int(self.cfg["scan"].get("confirm_hits", 2))
        raw = [
            asdict(d) for d in sorted(
                self.state.detections.values(), key=lambda x: -x.snr_db)
            if d.hits >= need and not self._is_empty_dropped(d.freq_hz)
        ]
        items = scan_hits.filter_published(
            raw, self.cfg["scan"].get("hit_filter", scan_hits.HIT_FILTER_DEFAULT),
        )
        lock_hz = self.state.lock_target
        if self.state.mode == "LOCK" and lock_hz is not None:
            for d in items:
                if scan_hits.same_channel(d["freq_hz"], lock_hz, scan_hits.MERGE_LOCK_HZ):
                    d["freq_hz"] = float(lock_hz)
        return items

    def _arm_operator_lock(self, freq_hz: float) -> None:
        """Remember which published hit the operator selected, if any."""
        mhz = self._published_mhz_near(freq_hz)
        if (self._operator_lock_hit_mhz is not None
                and mhz == self._operator_lock_hit_mhz):
            return
        self._operator_lock_at = time.monotonic()
        self._operator_lock_hit_mhz = mhz

    def _published_mhz_near(self, freq_hz: float) -> int | None:
        best = None
        best_d = scan_hits.HIT_SELECT_HZ
        for d in self._published_detections():
            dist = abs(float(d["freq_hz"]) - float(freq_hz))
            if dist <= best_d:
                best_d = dist
                best = int(round(float(d["freq_hz"]) / 1e6))
        return best

    def _is_empty_dropped(self, freq_hz: float) -> bool:
        key = scan_hits.hit_key(freq_hz)
        span = int(scan_hits.HIT_SELECT_HZ / scan_hits.HIT_KEY_HZ)
        return any(abs(int(r) - key) <= span for r in self._empty_drops)

    def _clear_empty_drop_near(self, freq_hz: float) -> None:
        key = scan_hits.hit_key(freq_hz)
        span = int(scan_hits.HIT_SELECT_HZ / scan_hits.HIT_KEY_HZ)
        gone = [r for r in self._empty_drops if abs(int(r) - key) <= span]
        for r in gone:
            self._empty_drops.discard(r)
            self._empty_drop_saved.pop(r, None)

    def _detection_near(self, freq_hz: float, tol_hz: float = 2e6):
        best = None
        best_d = float(tol_hz)
        for d in self.state.detections.values():
            dist = abs(d.freq_hz - float(freq_hz))
            if dist <= best_d:
                best_d = dist
                best = d
        return best

    def _touch_lock_picture(self, freq_hz: float, frame, pic) -> None:
        det = self._detection_near(freq_hz)
        if det is None:
            return
        det.last_picture_at = time.time()
        det.last_seen = det.last_picture_at
        det.pic_score = round(float(pic.value), 3)
        det.row_corr = round(float(pic.row_corr), 3)
        det.pic_locked = bool(pic.locked)
        det.pic_lines = int(pic.lines)
        if frame is not None and getattr(frame, "standard", None):
            det.standard = frame.standard

    def _lock_pic_sample(self, frame, pic) -> dict:
        std = getattr(frame, "standard", "") if frame is not None else ""
        return {
            "pic_locked": bool(pic.locked),
            "pic_score": float(pic.value),
            "row_corr": float(pic.row_corr),
            "pic_lines": int(pic.lines),
            "standard": std or "",
            "lines": int(getattr(frame, "lines", 0) or pic.lines),
        }

    def _maybe_prune_empty_lock(self, frame, pic) -> None:
        if self.state.mode != "LOCK" or self.state.auto:
            return
        mhz = self._operator_lock_hit_mhz
        if mhz is None:
            return
        sample = self._lock_pic_sample(frame, pic)
        flowing = frame is not None and self._frame_ts is not None
        if scan_hits.is_video_green(sample, streaming=flowing):
            self._revive_empty_drop(mhz, sample)
            return
        elapsed = time.monotonic() - float(self._operator_lock_at or 0.0)
        if not scan_hits.should_prune_empty_lock(
            operator_selected=True,
            auto_lock=False,
            elapsed_s=elapsed,
            sample=sample,
        ):
            return
        self._drop_empty_lock(mhz)

    def _drop_empty_lock(self, mhz: int) -> None:
        target_hz = float(mhz) * 1e6
        key = None
        det = None
        for k, d in list(self.state.detections.items()):
            if abs(d.freq_hz - target_hz) <= scan_hits.HIT_SELECT_HZ:
                key, det = k, d
                break
        if det is None:
            return
        del self.state.detections[key]
        drop_key = int(key)
        self._empty_drops.add(drop_key)
        self._empty_drop_saved[drop_key] = det
        self._emit("state", self.snapshot())

    def _revive_empty_drop(self, mhz: int, sample: dict) -> None:
        target_key = scan_hits.hit_key(float(mhz) * 1e6)
        span = int(scan_hits.HIT_SELECT_HZ / scan_hits.HIT_KEY_HZ)
        key = None
        for r in list(self._empty_drops):
            if abs(int(r) - target_key) <= span:
                key = int(r)
                break
        if key is None:
            return
        det = self._empty_drop_saved.pop(key, None)
        self._empty_drops.discard(key)
        if det is None:
            return
        det.pic_locked = bool(sample.get("pic_locked"))
        det.pic_score = round(float(sample.get("pic_score") or 0.0), 3)
        det.row_corr = round(float(sample.get("row_corr") or 0.0), 3)
        det.pic_lines = int(sample.get("pic_lines") or 0)
        if sample.get("standard"):
            det.standard = str(sample["standard"])
        det.last_picture_at = time.time()
        det.last_seen = det.last_picture_at
        self.state.detections[key] = det
        self._emit("detection", asdict(det))
        self._emit("state", self.snapshot())
 


