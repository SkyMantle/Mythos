
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
import errno
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from queue import Queue, Empty
from .iqbuffer import IQRingBuffer
 
import numpy as np
 
from .bands import PRIORITY_BANDS, band_of, nearest_channel
from .dsp import spectrum, demod, cvbs, adaptive_if, streaming
from .dsp.field_blend import blend_same_field
from . import (
    auto_mgc, hardware_rate, paths, scan_gate, scan_hits, scan_view,
    sweep_inspect,
)
from .rotator import AntennaRotator
from .sdr.bladerf import BladeRFError
from .sdr.file import write_capture
try:
    from .sdr.file import is_disk_full_error
except ImportError:  # partial Pi deploy: new engine, older sdr/file.py
    def is_disk_full_error(exc: BaseException) -> bool:
        err = getattr(exc, "errno", None)
        if err in (errno.ENOSPC, errno.EDQUOT):
            return True
        msg = str(exc).lower()
        return "no space left" in msg or (
            "requested and" in msg and "written" in msg
        )
from .web.coalesce import enqueue_live_event
from .recorder import VideoRecorder, FfmpegMissing
from .video_publisher import SnapshotRequest, VideoPublisher
from .app.core.logging import sanitize_log_text
 
 
log = logging.getLogger(__name__)
IQ_CAPTURE_DEFAULT_SECONDS = 0.05
IQ_CAPTURE_MAX_BYTES = 32 * 1024 * 1024
IQ_CAPTURE_STALE_SECONDS = 1.0
IQ_BYTES_PER_SAMPLE = np.dtype(np.complex64).itemsize
SDR_OPEN_BACKOFF_S = (1.0, 2.0, 5.0, 10.0, 15.0, 30.0)
SDR_AVAILABILITY_CODES = frozenset((-1, -5, -6, -7, -17, -18))
SDR_DEVICE_LOST_CODES = frozenset((-1, -5, -7, -17, -18))
MOTOR_GUARD_SETTLE_S = 0.75
MOTOR_GUARD_STABLE_ITERATIONS = 4
# One PAL/NTSC field at the unified hardware rate.  Larger LOCK snapshots
# hold the GIL long enough that sync_rx times out during motor PWM, then
# the timeout path rebuilds USB RX and the BladeRF drops.
STREAM_LOCK_CHUNK_S = 0.040
# One PAL/NTSC field: enough for classify_video after 400 kHz decimation.
SWEEP_COMB_S = 0.022
SWEEP_AVERAGES_MAX = 8


class IQCaptureBusy(RuntimeError):
    """Another raw-IQ capture is being written."""


class IQCaptureUnavailable(RuntimeError):
    """The live LOCK ring has no samples to save."""


class IQCaptureTooLarge(ValueError):
    """The requested raw-IQ copy exceeds the process memory budget."""


class IQCaptureNoSpace(OSError):
    """The capture filesystem cannot hold the CF32 + JSON sidecar."""


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
    stage: str = "rf_candidate"
    analog_evidence: bool = False
    video_confirmed: bool = False
    prominence_db: float = 0.0
    inspect_votes: int = 0
    inspect_windows: int = 0
    inspect_retention: float = 0.0
    inspect_row_corr: float = 0.0
    inspect_elapsed_ms: float = 0.0
    adaptive_decimation: int = 0
    rejected_reason: str = ""
    sweep_generation: int = 0
 
 
 
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
    def __init__(
        self,
        source,
        cfg: dict,
        events: Queue,
        *,
        recovery_clock=None,
        recovery_wait=None,
    ):
        self.src = source
        self.cfg = cfg
        self._rate_plan = hardware_rate.derive_hardware_rate(cfg)
        hardware_rate.normalize_runtime_config(cfg, self._rate_plan)
        self.events = events            # чим годуємо веб-сокети
        self.state = EngineState()
        self._stop = threading.Event()
        self._cmd: Queue = Queue()
        self._thread: threading.Thread | None = None
        self._worker_started = False
        self._worker_failure: str | None = None
        self._io_cancel = threading.Event()
        self._io_epoch = 0
        self._io_epoch_lock = threading.Lock()
        self._sdr_state = "INIT"
        self._sdr_state_since = time.monotonic()
        self._sdr_consecutive_errors = 0
        self._sdr_recovery_attempts = 0
        self._sdr_open_attempts = 0
        self._sdr_open_failures = 0
        self._sdr_open_failure_streak = 0
        self._sdr_next_retry_ms = 0
        self._sdr_last_error: str | None = None
        self._sdr_last_error_code: int | None = None
        self._sdr_last_error_time: str | None = None
        self._sdr_last_recovery_action: str | None = None
        self._sdr_operation = "idle"
        self._sdr_last_operation = "startup"
        self._sdr_last_error_operation: str | None = None
        self._sdr_root_error: str | None = None
        self._sdr_root_error_code: int | None = None
        self._sdr_root_error_operation: str | None = None
        self._recovery_clock = recovery_clock or time.monotonic
        self._recovery_wait = recovery_wait or (
            lambda event, timeout: event.wait(timeout)
        )
        self._rec: VideoRecorder | None = None
        self._snap = False
        self._peeked: dict[int, float] = {}   # частоти, які вже бачилися в цьому проході
        self._lock_tuned: float | None = None  # на що вже перебудовані
        self._ring: IQRingBuffer | None = None       # кільцевий буфер IQ для LOCK
        self._iq_capture_lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()
        self._reader_pause = threading.Event()
        self._reader_idle = threading.Event()
        self._reader_idle.set()
        self._reader_err: Exception | None = None     # помилка з нитки читання
        self._lock_iq_buf: np.ndarray | None = None   # reused LOCK snapshot storage
        self._lock_cursor: int | None = None           # next unseen source sample
        self._stream_demod: streaming.StreamingDemodulator | None = None
        self._field_assembler: streaming.StreamingFieldAssembler | None = None
        self._stream_new_samples = 0
        self._stream_gap_count = 0
        self._stream_fields_dropped = 0
        self._stream_acquisition_ms = 0.0
        self._snow_frame_at = 0.0
        self._lock_state = None                        # cvbs.DecodeState | None
        self._lock_good: cvbs.Frame | None = None      # last visible analog raster
        self._lock_good_at = 0.0
        self._lock_dec: int | None = None              # коефіцієнт децимації минулого виклику
        self._adaptive_if = adaptive_if.Lifecycle()
        self._lock_n = 0
        self._lock_gen = 0                         # покоління LOCK (скидає hunt)
        self._acc: np.ndarray | None = None
        self._afc = 0.0                            # лише цифровий зсув каналайзера
        self._last_err = 0.0
        self._hunt_th: threading.Thread | None = None
        self._hunt_out: float | None = None
        self._hunt_note: tuple[float, float, float] | None = None
        self._hunt_hold = False                    # analog picture: abort in-flight hunt
        self._lock_score_peak = 0.0                # hunt лише коли оцінка впала
        self._insp_dbg: list[dict] = []
        self._sweep_dbg: list[dict] = []
        self._sweep_handoffs: dict[
            int, tuple[adaptive_if.Selection, float, int]
        ] = {}
        self._sweep_i = 0
        self._sweep_generation = 0
        self._sweep_auto_lock = bool(
            (cfg.get("scan") or {}).get("auto_peek", True)
        )
        self._timings: dict[str, float] = {}   # ковзне середнє по етапах, мс
        self._frame_ts: float | None = None    # час минулого відданого кадру
        self._fps_ema = 0.0
        self._iteration_ts: float | None = None
        self._iteration_fps_ema = 0.0
        self._ws_frame_ts: float | None = None
        self._ws_fps_ema = 0.0
        self._frame_seq = 0
        self._frame_mono = 0.0
        self._ws_frame_seq = 0
        self._ws_frame_mono = 0.0
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
        self._mgc_hold_reason = ""
        self._mgc_hold_lock = threading.Lock()
        self._mgc_state = auto_mgc.MgcState()
        self._mgc_auto_was = True
        self._sdr_guard_lock = threading.Lock()
        self._motor_guard_active = False
        self._motor_guard_moving = False
        self._motor_guard_settle_until = 0.0
        self._motor_guard_stable_count = 0
        self._deferred_gain_target: float | None = None
        self._deferred_gain_automatic = False
        self._deferred_bias_tee: bool | None = None
        self._empty_drops: set[int] = set()
        self._empty_drop_saved: dict[int, Detection] = {}
        self._operator_lock_at: float = 0.0
        self._operator_lock_hit_mhz: int | None = None
        self._lock_warn_at = 0.0
        self.rotator = AntennaRotator(cfg.get("rotator") or {})
        self._pub_lock = threading.Lock()
        self._pub_snap: dict | None = None
        self._pub_json = "{}"
        self._ws_json = '{"type":"state","data":{}}'
        self.video_publisher = VideoPublisher(
            snapshot_done=self._photo_done,
            snapshot_error=self._photo_error,
        )
        self.publish_snapshot()
 
    # ---------- зовнішнє API ----------
 
    def start(self):
        self.video_publisher.start()
        self._worker_started = True
        self._worker_failure = None
        self._set_sdr_state("STARTING")
        self._thread = threading.Thread(
            target=self._worker_entry, daemon=True, name="fpvscan-engine")
        self._thread.start()
 
    def stop(self):
        self._stop.set()
        self._io_cancel.set()
        if self._thread:
            # A synchronous libbladeRF receive cannot be interrupted inside C;
            # allow its one timeout to return, then cancellation skips retry.
            self._thread.join(timeout=self._sdr_reader_join_s() + 1.0)
        self.video_publisher.stop()
        rot = getattr(self, "rotator", None)
        if rot is not None:
            rot.close()
 
    def command(self, name: str, **kw):
        if name in ("lock", "sweep", "hardware_sample_rate"):
            # Thread-safe wakeup only. Engine state remains owned by _run.
            with self._io_epoch_lock:
                self._io_epoch += 1
                self._io_cancel.set()
                self._cmd.put((name, kw))
            return
        self._cmd.put((name, kw))

    def note_rate_parameter_values(self, values: dict) -> None:
        """Record legacy rate requests; only ``sdr.sample_rate`` changes RX.

        The adapter may write mode-specific values into the mutable runtime
        config first.  Restore both legacy keys to the selected shared source
        rate immediately so no SWEEP/LOCK path can act on those writes.
        """
        self._rate_plan = hardware_rate.with_runtime_requests(
            self._rate_plan, self.cfg, values,
        )
        if "sdr.sample_rate" not in values:
            hardware_rate.normalize_runtime_config(self.cfg, self._rate_plan)
            return

        requested_scan = self._rate_plan.requested_scan_rate_hz
        requested_video = self._rate_plan.requested_video_rate_hz
        target = hardware_rate.derive_hardware_rate(self.cfg)
        target = hardware_rate.with_runtime_requests(
            target,
            self.cfg,
            {
                "scan.sample_rate": requested_scan,
                "video.sample_rate": requested_video,
            },
        )
        hardware_rate.normalize_runtime_config(self.cfg, self._rate_plan)
        if self.worker_alive():
            self.command("hardware_sample_rate", plan=target)
        else:
            # The source is configured once when the service starts.
            self._rate_plan = target
            hardware_rate.normalize_runtime_config(self.cfg, target)

    def _reconfigure_hardware_rate_now(
        self, plan: hardware_rate.HardwareRatePlan,
    ) -> None:
        """One serialized full RX reconfigure for an explicit operator change."""
        target = float(plan.hardware_sample_rate_hz)
        current = float(getattr(self.src, "sample_rate", 0.0) or 0.0)
        if abs(current - target) <= 1.0:
            self._rate_plan = hardware_rate.with_actual_rate(plan, current)
            hardware_rate.normalize_runtime_config(self.cfg, self._rate_plan)
            return
        self._stop_reader()
        self._set_sdr_state("RECONFIGURING")
        self._sdr_last_recovery_action = (
            "explicit shared hardware sample-rate reconfiguration"
        )
        self._set_sdr_operation("sample_rate")
        self.src.set_sample_rate(target)
        actual = float(getattr(self.src, "sample_rate", target) or target)
        self._finish_sdr_operation()
        self._rate_plan = hardware_rate.with_actual_rate(plan, actual)
        hardware_rate.normalize_runtime_config(self.cfg, self._rate_plan)
        self._lock_tuned = None
        self._adaptive_if.reset("explicit hardware sample-rate change")
        self._sweep_handoffs.clear()
        self._dense_q.clear()
        self._dense_seen.clear()
        self._sweep_i = 0
        self._complete_sdr_operation(
            "explicit shared hardware sample-rate reconfiguration completed"
        )

    def worker_alive(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def _worker_entry(self) -> None:
        try:
            self._run()
        except BaseException as exc:
            self._worker_failure = sanitize_log_text(exc)
            self._set_sdr_state("ERROR")
            self._sdr_last_error = self._worker_failure
            self._sdr_last_error_time = datetime.now(timezone.utc).isoformat()
            log.exception("Engine worker terminated: %s", self._worker_failure)
            raise

    def _prepare_source_io(self) -> int:
        """Arm cancellation and return the transition epoch for this I/O."""
        with self._io_epoch_lock:
            epoch = self._io_epoch
            self._io_cancel.clear()
        set_cancel = getattr(self.src, "set_read_cancel_event", None)
        if callable(set_cancel):
            set_cancel(self._io_cancel)
        return epoch

    def _source_io_stale(self, epoch: int) -> bool:
        with self._io_epoch_lock:
            return epoch != self._io_epoch

    def _note_sdr_io_success(self) -> None:
        # A single low-level read is not enough to declare recovery: the LOCK
        # reader may succeed once and fail before the engine consumes a frame.
        # The owner loop marks OK only after a complete SWEEP/LOCK operation.
        return

    def _set_sdr_state(self, state: str) -> None:
        if state != self._sdr_state:
            clock = getattr(self, "_recovery_clock", time.monotonic)
            self._sdr_state_since = float(clock())
        self._sdr_state = state
        if state != "OK" and hasattr(self, "_sdr_guard_lock"):
            with self._sdr_guard_lock:
                if self._motor_guard_active:
                    self._motor_guard_stable_count = 0

    def _set_sdr_operation(self, operation: str) -> None:
        self._sdr_operation = str(operation)

    def _finish_sdr_operation(self) -> None:
        operation = self._sdr_operation
        if operation != "idle":
            self._sdr_last_operation = operation
        self._sdr_operation = "idle"

    def _sdr_state_age_ms(self) -> int:
        clock = getattr(self, "_recovery_clock", time.monotonic)
        since = float(getattr(self, "_sdr_state_since", clock()))
        return max(0, int((float(clock()) - since) * 1000))

    def _complete_sdr_operation(self, action: str) -> None:
        """Clear transient reconfigure state after one complete source operation."""
        self._sdr_consecutive_errors = 0
        self._sdr_next_retry_ms = 0
        self._set_sdr_state("OK")
        self._sdr_last_recovery_action = action
        self._finish_sdr_operation()

    @staticmethod
    def _is_transient_sdr_error(exc: Exception) -> bool:
        return Engine._is_expected_hardware_error(exc)

    @staticmethod
    def _is_expected_hardware_error(
        exc: Exception, *, during_open: bool = False
    ) -> bool:
        if not isinstance(exc, BladeRFError):
            return False
        code = getattr(exc, "code", None)
        if code in SDR_AVAILABILITY_CODES:
            return True
        if during_open and isinstance(code, int) and code < 0:
            # Every libbladeRF return during open/configuration is a hardware
            # availability failure. Programming exceptions remain separate.
            return True
        text = sanitize_log_text(exc).lower()
        hardware_words = (
            "no devices",
            "no device",
            "not open",
            "не відкрит",
            "зникла з шини",
            "usb",
            "permission",
            "access denied",
            "busy",
            "would block",
            "timeout",
            "timed out",
            "fpga не завантажена",
        )
        return any(word in text for word in hardware_words) and (
            during_open or "sync_rx" in text or "device" in text or "usb" in text
        )

    def _record_sdr_error(
        self,
        exc: Exception,
        action: str,
        *,
        preserve_recovery_root: bool = False,
    ) -> str:
        clean = sanitize_log_text(exc)
        code = getattr(exc, "code", None)
        if (
            preserve_recovery_root
            and self._sdr_root_error is not None
        ):
            self._sdr_last_error = self._sdr_root_error
            self._sdr_last_error_code = self._sdr_root_error_code
            self._sdr_last_error_operation = self._sdr_root_error_operation
            self._sdr_last_recovery_action = action
            self._finish_sdr_operation()
            return clean
        self._sdr_last_error = clean
        self._sdr_last_error_code = code
        self._sdr_last_error_time = datetime.now(timezone.utc).isoformat()
        self._sdr_last_recovery_action = action
        self._sdr_last_error_operation = self._sdr_operation
        if code in SDR_DEVICE_LOST_CODES and self._sdr_root_error is None:
            self._sdr_root_error = clean
            self._sdr_root_error_code = code
            self._sdr_root_error_operation = self._sdr_operation
        self._finish_sdr_operation()
        return clean

    @staticmethod
    def _requires_device_reopen(exc: Exception) -> bool:
        """Only invalid-handle/transport failures justify a USB reopen."""
        if not isinstance(exc, BladeRFError):
            return False
        code = getattr(exc, "code", None)
        if code in SDR_DEVICE_LOST_CODES:
            return True
        text = sanitize_log_text(exc).lower()
        return any(word in text for word in (
            "no devices", "no device", "device lost", "зникла з шини",
            "usb transport",
        ))

    def _close_source_for_reopen(self) -> None:
        self._set_sdr_state("WAITING_DEVICE")
        self._stop_reader()
        try:
            self.src.close()
        except Exception as exc:
            log.warning("SDR close during recovery failed: %s",
                        sanitize_log_text(exc))
        self._lock_tuned = None
        self._reader_err = None
        self._ring = None
        with self._io_epoch_lock:
            self._io_cancel.clear()
        self._set_sdr_state("WAITING_DEVICE")

    def _configure_open_source(self) -> None:
        self._set_sdr_operation("open")
        self.src.open()
        self._finish_sdr_operation()
        if getattr(self.src, "fixed_freq", False):
            self._rate_plan = hardware_rate.with_actual_rate(
                self._rate_plan,
                self.src.sample_rate,
                reason_suffix="fixed IQ fixture source rate",
            )
        else:
            self._set_sdr_operation("sample_rate")
            self.src.set_sample_rate(self._rate_plan.hardware_sample_rate_hz)
            self._finish_sdr_operation()
            actual = float(
                getattr(self.src, "sample_rate",
                        self._rate_plan.hardware_sample_rate_hz)
                or self._rate_plan.hardware_sample_rate_hz
            )
            self._rate_plan = hardware_rate.with_actual_rate(
                self._rate_plan, actual,
            )
        hardware_rate.normalize_runtime_config(self.cfg, self._rate_plan)
        self._set_sdr_operation("gain")
        self.src.set_gain(float(self.cfg["sdr"].get("gain_db", 30)))
        self._finish_sdr_operation()
        if hasattr(self.src, "set_bias_tee"):
            try:
                bt_on = bool(self.cfg["sdr"].get("bias_tee", False))
                self._set_sdr_operation("bias_tee")
                self.src.set_bias_tee(bt_on)
                self._finish_sdr_operation()
                self._apply_bias_tee_gain(bt_on, essential=True)
            except Exception as exc:
                if self._is_expected_hardware_error(exc, during_open=True):
                    raise
                self._emit("notice", {
                    "level": "error", "text": f"bias-tee: {exc}"})

        if (
            self.cfg["sdr"].get("quick_tune")
            and hasattr(self.src, "prime_quick_tune")
        ):
            self._set_sdr_operation("frequency")
            pts = self._sweep_plan()
            unique = sorted(set(int(point) for point in pts))
            ok = self.src.prime_quick_tune(unique)
            self._finish_sdr_operation()
            print(
                f"[quick tune] знято профілів: {ok} з {len(unique)}"
                + (
                    ""
                    if ok
                    else "  — плата/бібліотека не підтримує, "
                    "працюємо звичайно"
                ),
                flush=True,
            )

    def _try_open_source(self) -> bool:
        self._sdr_open_attempts += 1
        attempt = self._sdr_open_attempts
        self._set_sdr_state("RECOVERING")
        self._sdr_next_retry_ms = 0
        self._sdr_last_recovery_action = f"open attempt {attempt}"
        log.warning("SDR RECOVERING: open attempt %d", attempt)
        try:
            # Also clears a handle left by an open that failed halfway through.
            try:
                self.src.close()
            except Exception as exc:
                log.warning("SDR pre-open close failed: %s",
                            sanitize_log_text(exc))
            self._configure_open_source()
        except Exception as exc:
            try:
                self.src.close()
            except Exception:
                pass
            if not self._is_expected_hardware_error(exc, during_open=True):
                raise
            self._sdr_open_failures += 1
            self._sdr_open_failure_streak += 1
            self._sdr_recovery_attempts += 1
            self._set_sdr_state("WAITING_DEVICE")
            clean = self._record_sdr_error(
                exc,
                "waiting for device before next open attempt",
                preserve_recovery_root=True,
            )
            log.warning(
                "SDR WAITING_DEVICE: open attempt %d failed code=%s: %s",
                attempt,
                self._sdr_last_error_code,
                clean,
            )
            return False

        self._sdr_open_failure_streak = 0
        self._sdr_consecutive_errors = 0
        self._sdr_next_retry_ms = 0
        self._sdr_root_error = None
        self._sdr_root_error_code = None
        self._sdr_root_error_operation = None
        self._set_sdr_state("OK")
        self._finish_sdr_operation()
        self._sdr_last_recovery_action = "device open; requested mode resumed"
        self._lock_tuned = None
        self._reader_err = None
        self._ring = None
        log.info("SDR OK: device opened on attempt %d", attempt)
        return True

    def _open_retry_delay(self) -> float:
        index = max(0, self._sdr_open_failure_streak - 1)
        return SDR_OPEN_BACKOFF_S[min(index, len(SDR_OPEN_BACKOFF_S) - 1)]

    def _wait_for_sdr_retry(self, delay_s: float) -> bool:
        """Wait without hiding mode commands; shutdown always wins promptly."""
        deadline = self._recovery_clock() + max(0.0, delay_s)
        while not self._stop.is_set():
            remaining = max(0.0, deadline - self._recovery_clock())
            self._sdr_next_retry_ms = int(remaining * 1000)
            if remaining <= 0:
                break
            interrupted = bool(self._recovery_wait(self._io_cancel, remaining))
            if not interrupted:
                # A wait implementation returning false means its timeout
                # elapsed.  This also makes injected deterministic waits easy.
                break
            if self._stop.is_set():
                self._sdr_next_retry_ms = 0
                return False
            with self._io_epoch_lock:
                self._io_cancel.clear()
            self._drain_commands()
        self._sdr_next_retry_ms = 0
        return not self._stop.is_set()

    def capture_iq(self, seconds: float | None = None, note: str = "") -> dict:
        """Copy a bounded live window, then persist it outside the ring lock."""
        requested_s = (
            IQ_CAPTURE_DEFAULT_SECONDS if seconds is None else float(seconds)
        )
        if not 0 < requested_s <= 2.0:
            raise ValueError("seconds має бути в межах 0 < seconds <= 2")
        if not self._iq_capture_lock.acquire(blocking=False):
            raise IQCaptureBusy("захоплення IQ вже виконується")
        try:
            ring = self._ring
            if ring is None:
                raise IQCaptureUnavailable(
                    "IQ-кільце недоступне; спершу станьте на канал")

            context = dict(getattr(ring, "capture_context", {}) or {})
            fs = float(context.get("sample_rate")
                       or getattr(self.src, "sample_rate", 0)
                       or (self.cfg.get("video") or {}).get("sample_rate", 0))
            if fs <= 0:
                raise IQCaptureUnavailable("частота дискретизації IQ невідома")
            requested_samples = max(1, int(fs * requested_s))
            requested_bytes = requested_samples * IQ_BYTES_PER_SAMPLE
            if requested_bytes > IQ_CAPTURE_MAX_BYTES:
                max_s = IQ_CAPTURE_MAX_BYTES / (fs * IQ_BYTES_PER_SAMPLE)
                raise IQCaptureTooLarge(
                    f"IQ capture {requested_s:.3f} с потребує "
                    f"{requested_bytes / 1e6:.1f} МБ; максимум "
                    f"{IQ_CAPTURE_MAX_BYTES / 1e6:.1f} МБ "
                    f"({max_s:.3f} с при {fs / 1e6:.1f} Мвідл/с)"
                )
            filled = int(getattr(ring, "filled", ring.capacity))
            if filled <= 0:
                raise IQCaptureUnavailable(
                    "IQ-кільце ще не містить відліків")
            age_s = float(getattr(ring, "last_write_age_s", 0.0))
            if age_s > IQ_CAPTURE_STALE_SECONDS:
                raise IQCaptureUnavailable(
                    f"IQ-кільце не оновлювалось {age_s:.2f} с")

            wanted = min(int(ring.capacity), filled, requested_samples)
            snapshot_buf = np.empty(wanted, dtype=np.complex64)
            snap_t0 = time.perf_counter()
            snapshot_into = getattr(ring, "snapshot_into", None)
            if callable(snapshot_into):
                iq, abs_start = snapshot_into(snapshot_buf)
            else:
                iq, abs_start = ring.snapshot(wanted)
            snapshot_ms = (time.perf_counter() - snap_t0) * 1000
            if iq.size == 0:
                raise IQCaptureUnavailable("IQ-кільце ще не містить відліків")

            center_hz = float(context.get("center_hz")
                              or self._lock_tuned
                              or self.state.tuned_hz)
            tuned_hz = context.get("tuned_hz", self.state.lock_target)
            sdr = self.cfg.get("sdr") or {}
            video = self.cfg.get("video") or {}
            gain_db = float(context.get(
                "gain_db", sdr.get("gain_db", getattr(self.src, "gain_db", 0))))
            now = datetime.now(timezone.utc)
            stamp = now.strftime("%Y%m%dT%H%M%S.%fZ")
            name = f"iq_{stamp}_{center_hz / 1e6:.3f}M.cf32"
            caps = paths.CAPS.resolve()
            dst = paths.ensure(caps / name)
            if dst.parent.resolve() != caps:
                raise RuntimeError("некоректний каталог IQ capture")

            app_version = None
            version_file = paths.ROOT / "deploy" / "VERSION"
            try:
                app_version = version_file.read_text(encoding="utf-8").strip() or None
            except OSError:
                pass
            serial = getattr(self.src, "serial", None)
            metadata = {
                "timestamp_utc": now.isoformat().replace("+00:00", "Z"),
                "center_hz": center_hz,
                "tuned_hz": None if tuned_hz is None else float(tuned_hz),
                "sample_rate": fs,
                "bandwidth_hz": context.get(
                    "bandwidth_hz", video.get("channel_bw_hz")),
                "channel_bw_hz": context.get(
                    "bandwidth_hz", video.get("channel_bw_hz")),
                "gain_db": gain_db,
                "mgc_db": gain_db,
                "auto_gain": bool(sdr.get("auto_gain", False)),
                "agc": context.get("agc", getattr(self.src, "agc", sdr.get("agc"))),
                "source": getattr(self.src, "name", None),
                "device": sdr.get("device") or getattr(self.src, "device", None),
                "serial": serial,
                "rx_channel": sdr.get(
                    "rx_channel", getattr(self.src, "ch", None)),
                "bias_tee": context.get(
                    "bias_tee", getattr(self.src, "bias_tee", sdr.get("bias_tee"))),
                "requested_seconds": requested_s,
                "default_seconds_used": seconds is None,
                "actual_samples": int(iq.size),
                "actual_duration_s": float(iq.size / fs),
                "ring_abs_start": int(abs_start),
                "ring_capacity_samples": int(ring.capacity),
                "config_version": self.cfg.get("version"),
                "app_version": app_version,
                "format": "complex64",
                "bytes_per_sample": IQ_BYTES_PER_SAMPLE,
                "snapshot_ms": snapshot_ms,
                "note": str(note or ""),
            }
            write_t0 = time.perf_counter()
            saved = write_capture(
                dst, iq, center_hz, fs, gain_db, str(note or ""),
                metadata=metadata,
            )
            write_ms = (time.perf_counter() - write_t0) * 1000
            byte_count = saved.stat().st_size
            log.warning(
                "IQ capture saved samples=%d bytes=%d snapshot_ms=%.1f "
                "write_ms=%.1f path=%s",
                iq.size, byte_count, snapshot_ms, write_ms, saved,
            )
            return {
                "success": True,
                "path": str(saved),
                "metadata_path": str(saved.with_suffix(".json")),
                "sample_count": int(iq.size),
                "samples": int(iq.size),
                "duration_s": float(iq.size / fs),
                "center_hz": center_hz,
                "sample_rate": fs,
                "byte_count": int(byte_count),
                "requested_duration_s": requested_s,
                "snapshot_ms": round(snapshot_ms, 3),
                "write_ms": round(write_ms, 3),
            }
        except (IQCaptureUnavailable, IQCaptureTooLarge, IQCaptureNoSpace, ValueError):
            raise
        except OSError as exc:
            if is_disk_full_error(exc):
                raise IQCaptureNoSpace(
                    "немає місця на диску для IQ capture"
                ) from exc
            log.exception("IQ capture failed requested_seconds=%.3f", requested_s)
            raise
        except Exception:
            log.exception("IQ capture failed requested_seconds=%.3f", requested_s)
            raise
        finally:
            self._iq_capture_lock.release()
 
    # ---------- внутрішнє ----------
 
    def _mark(self, stage: str, t0: float) -> float:
        """Ковзне середнє часу етапу (мс). Легке — тільки арифметика,
        жодних додаткових захоплень чи алокацій на гарячому шляху."""
        t1 = time.perf_counter()
        dt_ms = (t1 - t0) * 1000
        prev = self._timings.get(stage)
        self._timings[stage] = dt_ms if prev is None else prev * 0.8 + dt_ms * 0.2
        return t1

    def _observe_timing(self, stage: str, dt_ms: float) -> None:
        value = max(0.0, float(dt_ms))
        prev = self._timings.get(stage)
        self._timings[stage] = value if prev is None else prev * 0.8 + value * 0.2

    @staticmethod
    def _tick_fps(now: float, previous: float | None, ema: float
                  ) -> tuple[float, float]:
        if previous is None or now <= previous:
            return now, ema
        inst = 1.0 / (now - previous)
        return now, inst if ema == 0.0 else ema * 0.8 + inst * 0.2

    def _reset_lock_metrics(self) -> None:
        self._frame_ts = None
        self._fps_ema = 0.0
        self._iteration_ts = None
        self._iteration_fps_ema = 0.0
        self._ws_frame_ts = None
        self._ws_fps_ema = 0.0
        self._frame_seq = 0
        self._frame_mono = 0.0
        self._ws_frame_seq = 0
        self._ws_frame_mono = 0.0
        self._stream_new_samples = 0
        self._stream_gap_count = 0
        self._stream_fields_dropped = 0
        self._stream_acquisition_ms = 0.0
        self.video_publisher.reset()
        for key in (
            "ring_snapshot", "channelize", "demod", "demod_fm",
            "deemphasis", "adaptive_if", "decode", "line_hunt",
            "fallback", "blend", "submit", "encode", "lock_total",
            "stream_nco", "stream_filter_decimate", "stream_fm",
            "stream_deemphasis", "field_assembler",
        ):
            self._timings.pop(key, None)
 
    def _emit(self, kind: str, payload):
        """Кладе подію в чергу до веб-шару.

        Черга обмежена. Кадр і спектр завжди лишають останній знімок
        (drop-to-latest); інакше LOCK-відео витісняє FFT і смуга замирає.
        """
        enqueue_live_event(self.events, {"type": kind, "data": payload})

    def note_ws_frame_emitted(self, frame_seq: int) -> None:
        """Compatibility hook; publisher normally records emission itself."""
        now = time.perf_counter()
        self._ws_frame_ts, self._ws_fps_ema = self._tick_fps(
            now, self._ws_frame_ts, self._ws_fps_ema)
        self._ws_frame_seq = max(self._ws_frame_seq, int(frame_seq))
        self._ws_frame_mono = time.monotonic()
        self.video_publisher.note_emitted(frame_seq)

    def set_video_clients(self, count: int) -> None:
        self.video_publisher.set_clients(count)

    def take_video_frame(self) -> dict | None:
        return self.video_publisher.take_latest()
 
    def _drain_commands(self):
        while True:
            try:
                name, kw = self._cmd.get_nowait()
            except Empty:
                return
            try:
                self._handle_command(name, kw)
            except Exception as e:
                clean = sanitize_log_text(e)
                self._emit("notice", {"level": "error",
                                "text": f"команда «{name}»: {clean}"})
                print(f"[рушій] команда «{name}» впала: {clean}", flush=True)

    def _yield_to_commands(self) -> bool:
        """Віддати GIL і злити чергу. True — SWEEP inspect має зупинитись."""
        time.sleep(0.001)
        self._drain_commands()
        return self.state.mode == "LOCK" or self._stop.is_set()
 
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
            # A LOCK command retires every in-flight SWEEP result. Stop and
            # join any previous LOCK reader before the next direct retune.
            self._sweep_generation += 1
            self._stop_reader()
            self._acc = None
            self._acc_parity = None
            self._afc = 0.0
            self._last_err = 0.0
            self._hunt_out = None
            self._hunt_note = None
            self._hunt_hold = False
            self._lock_score_peak = 0.0
            self._lock_tuned = None
            self._lock_good = None
            self._lock_good_at = 0.0
            self._lock_n = 0
            self._lock_gen += 1
            self._adaptive_if.reset("new lock")
            self._adopt_sweep_handoff(target)
            self._reset_lock_metrics()
            self._reset_auto_mgc()
            self._afc_pegged = False
            self._afc_nudge = False
            self._afc_peg_n = 0
            self._rf_snap_at = 0.0
            self.state.lock_target = target
            self.state.tuned_hz = target
            self.state.mode = "LOCK"
            self.state.auto = False
            self.state.auto_until = 0.0
            # Drop the previous frequency's blob so HTTP /api/state cannot
            # keep serving 3250 metrics after a 3409 lock (snapshot() overlays
            # _last_video; publish_snapshot is rare during LOCK).
            self._last_video = None
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
            self._emit_state()
        elif name == "sweep":
            # An explicit SWEEP is a new operator-visible generation.  Retire
            # every candidate/adaptive artifact that could hand the previous
            # LOCK straight back to _maybe_peek(), and make an in-flight
            # inspect fail its generation check below.
            self._sweep_generation += 1
            self._sweep_auto_lock = bool(kw.get("auto_lock", False))
            self.state.lock_target = None
            self.state.mode = "SWEEP"
            self.state.auto = False
            self.state.auto_until = 0.0
            self._afc = 0.0
            self._lock_gen += 1
            self._adaptive_if.reset("left lock")
            self._sweep_handoffs.clear()
            self._peeked.clear()
            self._dense_q.clear()
            self._dense_seen.clear()
            self._empty_drops.clear()
            self._empty_drop_saved.clear()
            self._sweep_i = 0
            self._recent_hz.clear()
            self._next_hz = None
            self._acc = None
            self._acc_parity = None
            self._lock_state = None
            self._lock_good = None
            self._lock_good_at = 0.0
            self._lock_dec = None
            self._lock_tuned = None
            self._last_video = None
            self._lock_n = 0
            self._hunt_out = None
            self._hunt_note = None
            self._hunt_hold = False
            self._last_err = 0.0
            self._afc_pegged = False
            self._afc_nudge = False
            self._afc_peg_n = 0
            self._rf_snap_at = 0.0
            self._lock_score_peak = 0.0
            self._operator_lock_hit_mhz = None
            self._operator_lock_at = 0.0
            self._reset_lock_metrics()
            self._reset_auto_mgc()
            self._stop_reader()
            self._emit_state()
        elif name == "clear":
            self._peeked.clear()
            self._sweep_i = 0
            self.state.detections.clear()
            self._sweep_handoffs.clear()
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
                if self._motor_sdr_guarded():
                    with self._sdr_guard_lock:
                        self._deferred_bias_tee = on
                    self._emit("notice", {
                        "level": "ok",
                        "text": "bias-tee відкладено до стабілізації SDR",
                    })
                else:
                    self._stop_reader()
                    self._set_sdr_operation("bias_tee")
                    ok = self.src.set_bias_tee(on)
                    self._finish_sdr_operation()
                    if ok:
                        self._apply_bias_tee_gain(on)
                    self._emit("notice", {"level": "ok" if ok else "error", "text":
                               f"bias-tee: {'увімкнено' if on and ok else 'вимкнено' if ok else 'не підтримується платою'}"})
            else:
                self._emit("notice", {"level": "error", "text": "джерело не підтримує bias-tee"})
        elif name == "gain":
            self._apply_bias_tee_gain(
                bool(self.cfg.get("sdr", {}).get("bias_tee", False)))
        elif name == "hardware_sample_rate":
            plan = kw.get("plan")
            if not isinstance(plan, hardware_rate.HardwareRatePlan):
                raise ValueError("hardware_sample_rate requires a rate plan")
            self._reconfigure_hardware_rate_now(plan)
            self._emit_state()
        elif name == "refresh_lock":
            self._refresh_lock_now()
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
 
    def _board_gain_db(self, catalog_db: float, bias_on: bool) -> float:
        """Slider ``gain_db`` → MGC on the board.

        Bias-T LNA offset is headroom in the middle of the range, not a
        hidden cap: slider max still reaches the board max so a distant
        analog bird can use the full 0…60. Overload is clip_frac / auto MGC.
        """
        base = float(catalog_db)
        if not bias_on:
            return base
        offset = float(self.cfg["sdr"].get("bias_tee_gain_offset_db", 15))
        limits = getattr(self.src, "_gain_limits", (-15, 60))
        try:
            hi = float(limits[1])
        except (TypeError, ValueError, IndexError):
            hi = 60.0
        headroom = max(0.0, hi - base)
        take = min(max(0.0, offset), headroom)
        return base - take

    def _motor_guard_settle_s(self) -> float:
        cfg = self.cfg.get("rotator") or {}
        try:
            return max(0.0, float(cfg.get(
                "sdr_guard_settle_s",
                cfg.get("auto_mgc_settle_s", MOTOR_GUARD_SETTLE_S),
            )))
        except (TypeError, ValueError):
            return MOTOR_GUARD_SETTLE_S

    def _motor_guard_required_iterations(self) -> int:
        try:
            value = int((self.cfg.get("rotator") or {}).get(
                "sdr_guard_stable_iterations",
                MOTOR_GUARD_STABLE_ITERATIONS,
            ))
        except (TypeError, ValueError):
            value = MOTOR_GUARD_STABLE_ITERATIONS
        return max(1, min(20, value))

    def begin_motor_guard(self) -> None:
        """Quiesce USB RX before a motor PWM command."""
        with self._sdr_guard_lock:
            self._motor_guard_active = True
            self._motor_guard_moving = True
            self._motor_guard_settle_until = 0.0
            self._motor_guard_stable_count = 0
        self._set_rx_fragile(True)
        self._pause_lock_reader(True)
        self._wait_rx_idle(0.85)

    def _pause_lock_reader(self, paused: bool) -> None:
        if paused:
            self._reader_pause.set()
        else:
            self._reader_pause.clear()

    def _wait_rx_idle(self, timeout_s: float) -> None:
        """Do not start PWM while BladeRF still has an in-flight USB transfer."""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            idle = self._reader_idle.is_set()
            thread = self._reader_thread
            if thread is None or not thread.is_alive():
                idle = True
            sync = bool(getattr(self.src, "_sync_rx_active", False))
            if idle and not sync:
                return
            time.sleep(0.005)

    def _set_rx_fragile(self, fragile: bool) -> None:
        setter = getattr(self.src, "set_rx_fragile", None)
        if callable(setter):
            setter(bool(fragile))
        elif hasattr(self.src, "_rx_fragile"):
            self.src._rx_fragile = bool(fragile)

    def _update_motor_guard_status(self, st: dict, now: float) -> None:
        moving = bool(st.get("moving") or st.get("busy"))
        with self._sdr_guard_lock:
            if not self._motor_guard_active:
                return
            if moving:
                self._motor_guard_moving = True
                self._motor_guard_settle_until = 0.0
                self._motor_guard_stable_count = 0
            elif self._motor_guard_moving or self._motor_guard_settle_until <= 0.0:
                self._motor_guard_moving = False
                self._motor_guard_settle_until = (
                    now + self._motor_guard_settle_s())
                self._motor_guard_stable_count = 0

    def _refresh_motor_guard_status(self) -> None:
        with self._sdr_guard_lock:
            active = self._motor_guard_active
        if not active:
            return
        rot = getattr(self, "rotator", None)
        if rot is None:
            return
        try:
            self._update_motor_guard_status(rot.status(), time.monotonic())
        except Exception:
            # A status failure must keep the conservative guard engaged.
            return

    def _motor_sdr_guarded(self) -> bool:
        self._refresh_motor_guard_status()
        with self._sdr_guard_lock:
            return self._motor_guard_active

    def _motor_guard_snapshot(self) -> dict:
        self._refresh_motor_guard_status()
        now = time.monotonic()
        with self._sdr_guard_lock:
            settle_ms = max(
                0, int((self._motor_guard_settle_until - now) * 1000))
            return {
                "active": self._motor_guard_active,
                "moving": self._motor_guard_moving,
                "settle_ms": settle_ms,
                "stable_iterations": self._motor_guard_stable_count,
                "required_iterations": self._motor_guard_required_iterations(),
                "gain_deferred": self._deferred_gain_target is not None,
                "bias_tee_deferred": self._deferred_bias_tee is not None,
            }

    def _note_motor_guard_rx_success(self) -> None:
        """Release deferred controls only after stable, fresh receive work."""
        self._refresh_motor_guard_status()
        now = time.monotonic()
        target: float | None = None
        automatic = False
        bias_tee: bool | None = None
        with self._sdr_guard_lock:
            if not self._motor_guard_active:
                return
            if (
                self._motor_guard_moving
                or now < self._motor_guard_settle_until
                or self._sdr_state != "OK"
            ):
                self._motor_guard_stable_count = 0
                return
            self._motor_guard_stable_count += 1
            if (
                self._motor_guard_stable_count
                < self._motor_guard_required_iterations()
            ):
                return
            self._motor_guard_active = False
            target = self._deferred_gain_target
            automatic = self._deferred_gain_automatic
            bias_tee = self._deferred_bias_tee
            self._deferred_gain_target = None
            self._deferred_gain_automatic = False
            self._deferred_bias_tee = None
        self._set_rx_fragile(False)
        if bias_tee is not None:
            self._stop_reader()
            self._set_sdr_operation("bias_tee")
            ok = self.src.set_bias_tee(bias_tee)
            self._finish_sdr_operation()
            if ok and target is None:
                target = self._board_gain_db(
                    float((self.cfg.get("sdr") or {}).get("gain_db", 30)),
                    bias_tee,
                )
        if target is not None and (
            not automatic
            or bool((self.cfg.get("sdr") or {}).get("auto_gain", True))
        ):
            self._write_sdr_gain(target)
        self._pause_lock_reader(False)

    def _write_sdr_gain(self, target: float) -> None:
        self._set_sdr_operation("gain")
        self.src.set_gain(float(target))
        self._finish_sdr_operation()

    def _apply_bias_tee_gain(
        self,
        on: bool,
        *,
        automatic: bool = False,
        essential: bool = False,
    ):
        """Write catalog gain to the board, with LNA headroom if Bias-T is on.

        Nearby + LNA still backs off while there is room below the board
        max. Cranking the slider to 60 no longer silently stops at 45.
        """
        base_gain = float(self.cfg["sdr"].get("gain_db", 30))
        target = self._board_gain_db(base_gain, on)
        if not essential and self._motor_sdr_guarded():
            with self._sdr_guard_lock:
                self._deferred_gain_target = float(target)
                self._deferred_gain_automatic = bool(automatic)
            return False
        try:
            self._write_sdr_gain(target)
            return True
        except Exception as e:
            if self._is_expected_hardware_error(e):
                raise
            self._finish_sdr_operation()
            self._emit("notice", {"level": "error", "text": f"gain при bias-tee: {e}"})
            return False

    def hold_auto_mgc(
        self,
        seconds: float | None = None,
        *,
        reason: str = "operator",
    ) -> None:
        """Pause software MGC after the operator moved the gain slider."""
        hold = auto_mgc.HOLD_S if seconds is None else float(seconds)
        if seconds is None:
            try:
                hold = float(self.cfg.get("sdr", {}).get("auto_gain_hold_s", auto_mgc.HOLD_S))
            except (TypeError, ValueError):
                hold = auto_mgc.HOLD_S
        deadline = time.monotonic() + max(0.0, hold)
        with self._mgc_hold_lock:
            if deadline >= self._mgc_hold_until:
                self._mgc_hold_until = deadline
                self._mgc_hold_reason = str(reason)

    def _auto_mgc_hold_snapshot(self) -> tuple[int, str | None]:
        now = time.monotonic()
        with self._mgc_hold_lock:
            remaining_ms = max(0, int((self._mgc_hold_until - now) * 1000))
            reason = self._mgc_hold_reason if remaining_ms else None
        if self._motor_sdr_guarded():
            return max(1, remaining_ms), "rotator motor guard"
        return remaining_ms, reason

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

    def _maybe_auto_mgc(self, frame, iq=None, pic=None) -> None:
        """LOCK software MGC: step catalog gain_db. BladeRF AGC stays off."""
        sdr = self.cfg.get("sdr") or {}
        enabled = bool(sdr.get("auto_gain", True))
        if enabled and not self._mgc_auto_was:
            self._reset_auto_mgc()
        self._mgc_auto_was = enabled
        if not enabled or self._sdr_state not in ("OK", "INIT"):
            return
        if self._motor_sdr_guarded():
            return
        now = time.monotonic()
        try:
            interval = float(sdr.get("auto_gain_interval_s", auto_mgc.INTERVAL_S))
        except (TypeError, ValueError):
            interval = auto_mgc.INTERVAL_S
        if not auto_mgc.due(now, self._mgc_last_mono, interval):
            return
        self._mgc_last_mono = now
        if pic is None:
            pic = cvbs.score_picture(frame)
        luma = None if frame is None else getattr(frame, "luma", None)
        src_rms = float(getattr(self.src, "adc_rms", 0.0) or 0.0)
        with self._mgc_hold_lock:
            hold_until = self._mgc_hold_until
        sample = auto_mgc.MgcSample(
            gain_db=float(sdr.get("gain_db", 30)),
            pic_locked=bool(pic.locked),
            pic_score=float(pic.value),
            pic_lines=int(pic.lines),
            row_corr=float(pic.row_corr),
            clip_frac=float(getattr(self.src, "clip_frac", 0.0) or 0.0),
            operator_hold=now < hold_until,
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
        self._apply_bias_tee_gain(
            bool(sdr.get("bias_tee", False)), automatic=True)
        self._emit_state()
 
    def _snapshot_request(self) -> SnapshotRequest:
        name = paths.stamped("shot", "webp", self.state.lock_target)
        dst = paths.ensure(paths.PHOTOS / name)
        return SnapshotRequest(dst, f"out/photos/{name}", name)

    def _photo_done(self, request: SnapshotRequest, byte_count: int) -> None:
        self._last_frame_ref = request.ref
        self._emit("notice", {"level": "ok", "text":
                   f"знімок: {request.name} ({byte_count/1024:.1f} КБ)"})

    def _photo_error(self, request: SnapshotRequest, exc: Exception) -> None:
        self._emit("notice", {"level": "error", "text":
                   f"знімок {request.name}: {exc}"})
 
    def _run(self):
        source_open = False
        try:
            while not self._stop.is_set():
                if not source_open:
                    self._drain_commands()
                    if self._try_open_source():
                        source_open = True
                        continue
                    if not self._wait_for_sdr_retry(self._open_retry_delay()):
                        break
                    continue
                try:
                    # Source controls queued by HTTP are part of the same
                    # hardware recovery boundary as SWEEP/LOCK I/O.
                    self._drain_commands()
                    if self.state.mode == "LOCK" and self.state.lock_target:
                        self._do_lock()
                    else:
                        self._do_sweep()
                    self._sdr_consecutive_errors = 0
                    self._set_sdr_state("OK")
                    self._finish_sdr_operation()
                    self._sdr_next_retry_ms = 0
                    self._sdr_last_recovery_action = "read recovered"
                except InterruptedError:
                    # A queued mode command cancelled a stale blocking read.
                    # Next iteration drains it inside the recovery boundary.
                    continue
                except Exception as exc:
                    if not self._is_expected_hardware_error(exc):
                        raise
                    fails = self._sdr_consecutive_errors + 1
                    self._sdr_consecutive_errors = fails
                    self._sdr_recovery_attempts += 1
                    code = getattr(exc, "code", None)
                    if code == -6 or not self._requires_device_reopen(exc):
                        self._set_sdr_state("STREAM_RECOVERING")
                        clean = self._record_sdr_error(
                            exc,
                            "bounded in-place stream retry; device remains open",
                        )
                        if fails <= 3 or fails % 10 == 0:
                            log.warning(
                                "SDR STREAM_RECOVERING: read failure %d "
                                "code=%s (device remains open): %s",
                                fails,
                                code,
                                clean,
                            )
                        self._emit("notice", {
                            "level": "error", "text": f"приймач: {clean}"})
                        if not self._wait_for_sdr_retry(
                            min(5.0, 0.5 * fails)
                        ):
                            break
                        continue

                    clean = self._record_sdr_error(
                        exc, "close device and enter open recovery loop")
                    self._set_sdr_state("WAITING_DEVICE")
                    log.warning(
                        "SDR WAITING_DEVICE: read failure %d code=%s, "
                        "dead handle abandoned before reopen: %s",
                        fails,
                        code,
                        clean,
                    )
                    self._emit("notice", {
                        "level": "error",
                        "text": f"приймач перевідкривається: {clean}",
                    })
                    self._close_source_for_reopen()
                    source_open = False
        finally:
            self._stop_reader()
            self._rec_stop()
            try:
                self.src.close()
            except Exception as exc:
                log.warning("SDR final close failed: %s",
                            sanitize_log_text(exc))
            if self._stop.is_set():
                self._set_sdr_state("STOPPED")
 
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
        self._patch_pub_scan(spec, spec["grid"])
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
        if self._motor_sdr_guarded():
            raise InterruptedError("SWEEP I/O deferred during rotator PWM")
        # Direct SWEEP I/O owns the source only after a LOCK reader has
        # stopped. The epoch prevents handing stale SWEEP data into LOCK.
        self._stop_reader()
        epoch = self._prepare_source_io()
        self._set_sdr_operation("frequency")
        fast = getattr(self.src, "retune_and_read_fast", None)
        iq = fast(f, n) if fast else self.src.retune_and_read(f, n)
        self._note_sdr_io_success()
        if self._source_io_stale(epoch):
            raise InterruptedError("stale SWEEP read after mode transition")
        self._complete_sdr_operation("SWEEP retune/read completed")
        self._note_motor_guard_rx_success()
        return iq

    def _energy_hit_ok(self, dwell_hz, occ, scan, offset_db) -> bool:
        """Старий scan_gate без offset_db не має валити свіп."""
        kw = dict(
            dwell_hz=dwell_hz, center_hz=occ.center_hz,
            bandwidth_hz=occ.bandwidth_hz, snr_db=occ.snr_db, scan=scan)
        try:
            return scan_gate.energy_hit_ok(**kw, offset_db=offset_db)
        except TypeError:
            return scan_gate.energy_hit_ok(**kw)

    def _do_sweep(self):
        if self._motor_sdr_guarded():
            time.sleep(0.02)
            self._note_motor_guard_rx_success()
            return
        scan = self.cfg["scan"]
        # Runtime source truth: SWEEP/LOCK share one configured rate, but the
        # device may quantize it.  Streaming positions must use that actual Hz.
        fs = float(
            getattr(self.src, "sample_rate",
                    self._rate_plan.hardware_sample_rate_hz)
            or self._rate_plan.hardware_sample_rate_hz
        )
        nfft = int(scan.get("fft_size", 8192))
        avg = max(1, min(int(scan.get("averages", 16)), SWEEP_AVERAGES_MAX))
        need = max(nfft * avg, int(fs * SWEEP_COMB_S))
 
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
            generation = self._sweep_generation
            iq = self._grab(f, need)
            self._drain_commands()
            if (
                self.state.mode == "LOCK"
                or generation != self._sweep_generation
                or self._stop.is_set()
            ):
                return
            psd = spectrum.psd_db(iq, nfft, avg)
            self.state.tuned_hz = f
            self.state.sweep_pos_hz = f
            self._recent_hz.append(float(f))
            if len(self._recent_hz) > 8:
                self._recent_hz = self._recent_hz[-8:]
 
            edge = float(scan.get("edge_guard", 0.95))
            notch = float(scan.get("dc_notch_hz", 200e3))
            pct = float(scan.get("noise_percentile", 25))
            view = spectrum.usable_view(
                psd, fs, edge_guard=edge, dc_notch_hz=notch,
                noise_percentile=pct)
            nf = spectrum.noise_floor_db(view, percentile=pct)
            offset = spectrum.occupancy_offset_db(view, scan)
            self._publish_spectrum(
                f, fs, spectrum.downsample_for_display(psd, 384),
                round(nf, 1), nfft,
            )
 
            occ = spectrum.find_occupied(
                psd, f, fs,
                threshold_db=offset,
                min_bw_hz=scan_gate.fft_min_bw_hz(scan),
                edge_guard=edge,
                dc_notch_hz=notch,
                noise_percentile=pct)
            seen = len(occ)
            occ = sweep_inspect.limit_proposals(
                occ, fs, metadata_center_hz=f, scan=scan,
            )

            for o in occ:
                rf_diag = {
                    "stage": "rf_proposed",
                    "freq_hz": round(o.center_hz, 1),
                    "peak_hz": round(o.peak_hz, 1),
                    "bandwidth_hz": round(o.bandwidth_hz, 1),
                    "snr_db": round(o.snr_db, 2),
                    "threshold_db": round(offset, 2),
                    "proposals_seen": seen,
                    "proposals_kept": len(occ),
                    "rejected_reason": "",
                }
                self._sweep_dbg.append(rf_diag)
                self._sweep_dbg = self._sweep_dbg[-16:]
                if scan.get("debug_candidates"):
                    self._emit("candidate", rf_diag)
                if self._energy_hit_ok(f, o, scan, offset):
                    self._merge_occupancy(o)
                score = self._sweep_classify(iq, fs, f, o, scan)
                line_hint = bool(
                    score is not None
                    and score.is_video
                    and score.standard in ("PAL", "NTSC")
                )
                if line_hint:
                    self._queue_cluster_dense(o.center_hz, f, scan)
                    self._merge_comb_detection(o, score, generation)

    def _merge_occupancy(self, occ) -> None:
        """List FFT energy like a typical FPV scanner — no PAL/NTSC required."""
        now = time.time()
        self._merge(Detection(
            freq_hz=occ.center_hz,
            bandwidth_hz=occ.bandwidth_hz,
            snr_db=round(occ.snr_db, 1),
            standard="?",
            confidence=0.0,
            channel=nearest_channel(occ.center_hz),
            band=band_of(occ.center_hz),
            first_seen=now, last_seen=now,
            sweep_generation=self._sweep_generation,
        ))

    def _queue_cluster_dense(self, peak_hz: float, dwell_hz: float, scan: dict) -> None:
        """Insert 4 MHz extras around a cluster hit. No second full-band pass."""
        if len(self._dense_q) >= 96:
            return
        bucket = int(round(float(peak_hz) / 2.0e6))
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
            if len(self._dense_q) >= 96:
                break
 
 
 
    # ---------- INSPECT ----------
 
    def _inspect(self, _iq, center_hz, fs, occ):
        """Підтвердження кандидата за рядковою частотою.
 
        Свіповий буфер для цього закороткий: щоб побачити лінію
        15.7 кГц, потрібні десятки її періодів, тобто ~20+ мс ефіру.
        Тому тут робиться окреме, довше захоплення.
        """
        generation = self._sweep_generation
        if self._yield_to_commands() or generation != self._sweep_generation:
            return
        # Same bins, fresh stamp so the scan playhead can lerp through inspect.
        stamp = scan_view.mono_ms()
        spec = {
            **self._spectrum_view(t_mono_ms=stamp),
            "grid": self._grid_view(t_mono_ms=stamp),
        }
        self._patch_pub_scan(spec, spec["grid"])
        self._emit("spectrum", spec)
        sc = self.cfg["scan"]
        insp_s = sweep_inspect.inspect_capture_seconds({
            **sc,
            "inspect_ms": scan_gate.inspect_ms(sc, occ.center_hz),
        })
        try:
            iq = self.src.retune_and_read(occ.center_hz, int(fs * insp_s))
            evidence = sweep_inspect.inspect_iq(
                iq,
                fs,
                bandwidth_hz=occ.bandwidth_hz,
                scan=sc,
                video=self.cfg.get("video") or {},
            )
        except Exception as e:
            failure = {
                "stage": "rejected",
                "freq_hz": occ.center_hz,
                "accepted": False,
                "rejected_reason": f"inspect failure: {e}",
            }
            self._sweep_dbg.append(failure)
            self._sweep_dbg = self._sweep_dbg[-16:]
            if sc.get("debug_candidates"):
                self._emit("candidate", failure)
            return

        if self._yield_to_commands() or generation != self._sweep_generation:
            return

        accepted = evidence.analog_evidence
        diag = {
            "stage": evidence.stage,
            "freq_hz": round(occ.center_hz, 1),
            "peak_hz": round(occ.peak_hz, 1),
            "bandwidth_hz": round(occ.bandwidth_hz, 1),
            "snr_db": round(occ.snr_db, 1),
            "line_rate": round(evidence.line_rate_hz, 1),
            "standard": evidence.standard,
            "confidence": round(evidence.line_confidence, 3),
            "prominence_db": round(evidence.prominence_db, 2),
            "votes": evidence.votes,
            "windows": evidence.windows,
            "retention": round(evidence.retention, 3),
            "row_corr": round(evidence.row_corr, 3),
            "video_confirmed": evidence.video_confirmed,
            "accepted": accepted,
            "rejected_reason": evidence.rejected_reason,
            "elapsed_ms": round(evidence.elapsed_ms, 1),
            "evaluations": evidence.evaluations,
        }
        self._sweep_dbg.append(diag)
        self._sweep_dbg = self._sweep_dbg[-16:]
        if sc.get("debug_candidates"):
            self._emit("candidate", diag)
        if not accepted:
            return
 
        now = time.time()
        published_pic = evidence.video_confirmed
        det = Detection(
            freq_hz=occ.center_hz + evidence.center_offset_hz,
            bandwidth_hz=occ.bandwidth_hz,
            snr_db=round(occ.snr_db, 1),
            standard=evidence.standard,
            confidence=evidence.line_confidence,
            channel=nearest_channel(occ.center_hz),
            band=band_of(occ.center_hz),
            first_seen=now, last_seen=now,
            pic_score=round(evidence.picture_score, 3) if published_pic else 0.0,
            line_rate=evidence.line_rate_hz,
            row_corr=round(evidence.row_corr, 3) if published_pic else 0.0,
            pic_locked=bool(evidence.picture_locked) if published_pic else False,
            pic_lines=int(evidence.picture_lines) if published_pic else 0,
            stage=evidence.stage,
            analog_evidence=True,
            video_confirmed=published_pic,
            prominence_db=round(evidence.prominence_db, 2),
            inspect_votes=evidence.votes,
            inspect_windows=evidence.windows,
            inspect_retention=round(evidence.retention, 3),
            inspect_row_corr=round(evidence.row_corr, 3),
            inspect_elapsed_ms=round(evidence.elapsed_ms, 1),
            adaptive_decimation=evidence.selection.decimation,
            rejected_reason=evidence.rejected_reason,
            sweep_generation=generation,
        )
        self._sweep_handoffs[scan_hits.hit_key(det.freq_hz)] = (
            evidence.selection, fs, generation,
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
            min_corr=float(sc.get("inspect_min_row_corr", 0.02)),
            require_lock=bool(sc.get("inspect_require_lock", False)),
            min_lines=int(sc.get("inspect_min_lines", 32)))
        return ok, pic

    def _note_insp(self, occ, score, pic, accepted: bool):
        """Кілька останніх рішень INSPECT — щоб бачити, чому список порожній."""
        floor = float(self.cfg["scan"].get("energy_min_snr_db", 1.5))
        if occ.snr_db < floor and not (score and score.is_video):
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
            return demod.line_comb_hint(
                base, fs2,
                min_prominence_db=float(scan.get("line_prominence_db", 2)),
                harm_db=float(scan.get("line_harm_db", 3)))
        except Exception:
            return False

    def _sweep_classify(self, iq, fs, dwell_hz, occ, scan):
        """PAL/NTSC comb on the dwell IQ. No extra retune, no decode."""
        try:
            mix = float(occ.center_hz) - float(dwell_hz)
            insp_bw = scan_gate.inspect_bw_hz(scan, occ.center_hz)
            out_bw = min(max(float(occ.bandwidth_hz), 8e6), insp_bw, fs * 0.9)
            ch, fs2 = demod.channelize(
                iq, fs, mix, out_bw_hz=out_bw,
                fast=bool(scan.get("fast_channelizer", False)))
            base = demod.fm_demod(
                ch, fs2, deviation_hz=max(float(occ.bandwidth_hz), 8e6) / 5)
            return demod.classify_video(
                base, fs2,
                tol_hz=float(scan.get("line_tol_hz", 200)),
                min_prominence_db=float(scan.get("line_prominence_db", 2)),
                min_conf=float(scan.get("min_confidence", 0.0)),
                min_harmonics=int(scan.get("min_harmonics", 0)),
                harm_db=float(scan.get("line_harm_db", 3)),
            )
        except Exception:
            return None

    def _merge_comb_detection(self, occ, score, generation) -> None:
        """List analog PAL/NTSC from the dwell comb without a 90 ms inspect."""
        if score is None or not score.is_video:
            return
        if score.standard not in ("PAL", "NTSC"):
            return
        now = time.time()
        self._merge(Detection(
            freq_hz=occ.center_hz,
            bandwidth_hz=occ.bandwidth_hz,
            snr_db=round(occ.snr_db, 1),
            standard=score.standard,
            confidence=float(score.confidence),
            channel=nearest_channel(occ.center_hz),
            band=band_of(occ.center_hz),
            first_seen=now, last_seen=now,
            line_rate=float(score.line_rate),
            stage="analog_evidence",
            analog_evidence=True,
            video_confirmed=False,
            prominence_db=round(float(score.prominence_db), 2),
            inspect_votes=max(1, int(score.harmonics)),
            inspect_windows=1,
            inspect_retention=1.0,
            rejected_reason="decoder preview not confirmed",
            sweep_generation=generation,
        ))

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

    def _inspect_offsets(self, iq, fs, out_bw, occ, sc, *,
                         deadline: float | None = None):
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
            if deadline is not None and time.monotonic() >= deadline:
                break
            if self._yield_to_commands():
                break
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
        matched = None
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
            if (
                self.state.mode == "SWEEP"
                and old.sweep_generation != det.sweep_generation
            ):
                # Keep old rows visible until revisited, but never inherit
                # their hit count, analogue votes, or confirmation into a new
                # SWEEP generation.
                del self.state.detections[k]
                key = k
                matched = None
                break
            matched = old
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
                det.analog_evidence = old.analog_evidence or det.analog_evidence
                det.video_confirmed = old.video_confirmed or det.video_confirmed
                det.prominence_db = max(old.prominence_db, det.prominence_db)
                det.inspect_votes = max(old.inspect_votes, det.inspect_votes)
                det.inspect_windows = max(old.inspect_windows, det.inspect_windows)
                det.inspect_retention = max(old.inspect_retention, det.inspect_retention)
                det.inspect_row_corr = max(old.inspect_row_corr, det.inspect_row_corr)
                det.inspect_elapsed_ms = max(
                    old.inspect_elapsed_ms, det.inspect_elapsed_ms,
                )
                if old.video_confirmed:
                    det.stage = old.stage
                elif det.analog_evidence and det.stage == "rf_candidate":
                    det.stage = "analog_evidence"
                det.standard = scan_hits.merge_standard(
                    old.standard, det.standard,
                    old_pic=old.pic_score, new_pic=det.pic_score,
                )
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
                    det.analog_evidence = old.analog_evidence or det.analog_evidence
                    det.video_confirmed = old.video_confirmed or det.video_confirmed
                    det.prominence_db = max(old.prominence_db, det.prominence_db)
                    det.inspect_votes = max(old.inspect_votes, det.inspect_votes)
                    det.inspect_windows = max(old.inspect_windows, det.inspect_windows)
                    det.inspect_retention = max(
                        old.inspect_retention, det.inspect_retention,
                    )
                    det.inspect_row_corr = max(
                        old.inspect_row_corr, det.inspect_row_corr,
                    )
                    det.standard = scan_hits.merge_standard(
                        old.standard, det.standard,
                        old_pic=old.pic_score, new_pic=det.pic_score,
                    )
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
        self._promote_analog_hit(det, matched)
        self.state.detections[key] = det
        # Одноразовий спалах у шумі не показуємо: справжній передавач
        # нікуди не подінеться і підтвердиться наступним проходом.
        if det.hits >= int(self.cfg["scan"].get("confirm_hits", 1)):
            if any(scan_hits.same_channel(float(d["freq_hz"]), det.freq_hz)
                   for d in self._published_detections()):
                self._emit("detection", asdict(det))
            self._maybe_peek(det)
 
    def _promote_analog_hit(self, det: Detection, old: Detection | None) -> None:
        """Keep analog PAL/NTSC in the list when energy later publishes '?'."""
        analog, confirmed, std, stage = scan_hits.promote_analog_identity(
            analog_evidence=bool(
                det.analog_evidence or (old is not None and old.analog_evidence)
            ),
            video_confirmed=bool(
                det.video_confirmed or (old is not None and old.video_confirmed)
            ),
            standard=det.standard,
            other_standard="" if old is None else old.standard,
            stage=det.stage,
        )
        det.analog_evidence = analog
        det.video_confirmed = confirmed
        det.standard = std
        det.stage = stage

    def _maybe_peek(self, det: Detection):
        """Automatically inspect a fresh candidate when auto-lock is enabled.
 
        Manual SWEEP hold never enters LOCK here.  Automated SWEEP can show a
        fresh current-generation candidate briefly and then resume scanning.
        """
        sc = self.cfg["scan"]
        if (
            not self._sweep_auto_lock
            or self.state.mode != "SWEEP"
            or det.sweep_generation != self._sweep_generation
            or not scan_gate.auto_peek_allowed(det, sc)
        ):
            return
        key = scan_hits.hit_key(det.freq_hz)
        now = time.time()
        # Не повертатись на той самий канал щопроходу.
        if now - self._peeked.get(key, 0) < float(sc.get("auto_peek_cooldown_s", 60)):
            return
        self._peeked[key] = now
        self._lock_tuned = None
        self._lock_good = None
        self._lock_good_at = 0.0
        self._acc = None
        self._acc_parity = None
        self._afc = 0.0
        self._lock_n = 0
        self._lock_gen += 1
        self._adaptive_if.reset("auto peek")
        self._adopt_sweep_handoff(det.freq_hz)
        self._reset_lock_metrics()
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

    def _adopt_sweep_handoff(self, freq_hz: float) -> bool:
        """Reuse INSPECT DSP selection only when LOCK has the same sample rate."""
        target_fs = self._rate_plan.hardware_sample_rate_hz
        if target_fs <= 0.0:
            return False
        best: tuple[adaptive_if.Selection, float] | None = None
        best_dist = float("inf")
        for key, item in self._sweep_handoffs.items():
            center = float(key) * scan_hits.HIT_KEY_HZ
            dist = abs(center - float(freq_hz))
            selection, source_fs, generation = item
            if (
                generation == self._sweep_generation
                and dist <= self.MERGE_TOL_HZ
                and dist < best_dist
            ):
                best = (selection, source_fs)
                best_dist = dist
        if best is None:
            return False
        selection, source_fs = best
        if abs(float(source_fs) - target_fs) > 1.0:
            return False
        if not (selection.confirmed and selection.stable and selection.analog):
            return False
        self._adaptive_if.adopt(selection, target_fs)
        return True
 
    # ---------- LOCK ----------
 
    def _lock_bw(self, _freq_hz: float, default_bw: float) -> float:
        """LOCK IF width. YAML is the ceiling and the floor (≥8 MHz).

        Sweep occupancy can be a 16+ MHz smear *or* a 2 MHz peak tip.
        Never raise above YAML. Never shrink to the FFT tip — that left
        3700 PAL inside a 8–9 MHz window while another box decoded it.
        """
        return max(8e6, float(default_bw))
 
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
        actual_fs = float(getattr(self.src, "sample_rate", fs) or fs)
        fs = actual_fs
        self._reader_stop = threading.Event()
        epoch = self._prepare_source_io()
        self._set_sdr_operation("frequency")
        first = self.src.retune_and_read(want, max(2048, int(fs * 0.01)))
        self._note_sdr_io_success()
        if self._source_io_stale(epoch):
            raise InterruptedError("stale LOCK startup after mode transition")
        self._complete_sdr_operation("LOCK reconfigure/retune/read completed")
        ring = IQRingBuffer(capacity=max(len(first), int(fs * ring_seconds)))
        ring.capture_context = {
            "center_hz": float(want),
            "tuned_hz": (
                None if self.state.lock_target is None
                else float(self.state.lock_target)
            ),
            "sample_rate": float(fs),
            "bandwidth_hz": float(
                (self.cfg.get("video") or {}).get("channel_bw_hz", 0)
            ) or None,
            "gain_db": float(
                (self.cfg.get("sdr") or {}).get(
                    "gain_db", getattr(self.src, "gain_db", 0)
                )
            ),
            "agc": getattr(self.src, "agc", (self.cfg.get("sdr") or {}).get("agc")),
            "bias_tee": getattr(
                self.src, "bias_tee", (self.cfg.get("sdr") or {}).get("bias_tee")
            ),
        }
        ring.write(first)
        self._ring = ring
        self._lock_cursor = 0
        self._stream_demod = None
        self._field_assembler = None
        self._snow_frame_at = 0.0
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
 
    def _sdr_reader_join_s(self) -> float:
        """Join outlasts one receive timeout; cancellation skips its retry."""
        ms = float(getattr(self.src, "timeout_ms", 3500) or 3500)
        return max(0.5, ms / 1000.0 + 0.5)

    def _stop_reader(self):
        thread = self._reader_thread
        if thread is None:
            self._ring = None
            self._lock_cursor = None
            self._stream_demod = None
            self._field_assembler = None
            return
        self._set_sdr_state("STREAM_STOPPING")
        self._sdr_last_recovery_action = "stop and join LOCK reader"
        self._reader_stop.set()
        self._io_cancel.set()
        join_s = self._sdr_reader_join_s()
        t0 = time.perf_counter()
        thread.join(timeout=join_s)
        elapsed = time.perf_counter() - t0
        if thread.is_alive():
            log.error(
                "SDR reader did not stop after %.3f s (timeout %.3f s)",
                elapsed, join_s,
            )
            raise RuntimeError(
                f"нитка SDR не зупинилась за {join_s:.1f} с")
        if elapsed >= 0.25:
            log.warning("SDR reader stop took %.3f s", elapsed)
        self._reader_thread = None
        self._ring = None
        self._lock_cursor = None
        self._stream_demod = None
        self._field_assembler = None

    def _restart_reader_keep_stream(self, fs: float) -> None:
        """Resume LOCK RX after a timeout without retune or USB rebuild."""
        thread = self._reader_thread
        if thread is not None and thread.is_alive():
            return
        if self._ring is None:
            return
        self._reader_err = None
        self._reader_stop = threading.Event()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, args=(fs,), daemon=True)
        self._reader_thread.start()

    def _lock_during_motor_pwm(self, fs: float) -> None:
        """No USB during PWM. Last JPEG stays; RX resumes after settle."""
        del fs
        time.sleep(0.02)
        self._note_motor_guard_rx_success()
 
    def _reader_loop(self, fs: float):
        chunk = max(1024, int(fs * 0.005))     # 5 мс за раз
        previous_stream_marks = (
            int(getattr(self.src, "overflows", 0)),
            int(getattr(self.src, "stream_restarts", 0)),
            int(getattr(self.src, "stream_discontinuities", 0)),
        )
        while not self._reader_stop.is_set():
            if self._reader_pause.is_set():
                self._reader_idle.set()
                time.sleep(0.01)
                continue
            self._reader_idle.clear()
            if self._reader_pause.is_set() or self._reader_stop.is_set():
                self._reader_idle.set()
                continue
            try:
                iq = self.src.read(chunk)
                self._note_sdr_io_success()
            except Exception as e:
                self._reader_idle.set()
                if self._reader_stop.is_set():
                    return
                self._reader_err = e
                return
            self._reader_idle.set()
            ring = self._ring
            if ring is None:
                return
            stream_marks = (
                int(getattr(self.src, "overflows", 0)),
                int(getattr(self.src, "stream_restarts", 0)),
                int(getattr(self.src, "stream_discontinuities", 0)),
            )
            discontinuity = stream_marks != previous_stream_marks
            previous_stream_marks = stream_marks
            ring.write(iq, discontinuity=discontinuity)
 
    def _digital_afc_lim(self, fs: float, off: float, ch_bw: float,
                         vcfg: dict) -> float:
        """Стеля |цифрового AFC|: запас Найквіста. Далі — стоп, не RF."""
        lim = float(vcfg.get("afc_limit_hz", 20e6))
        cap = float(vcfg.get("afc_digital_max_hz", 1.5e6))
        nyq = max(0.0, fs * 0.45 - abs(off) - ch_bw / 2)
        return min(lim, nyq, cap)

    def _video_ok_for_afc(self, frame: cvbs.Frame | None, pic=None) -> bool:
        """Digital mixer AFC when a PAL/NTSC comb is on screen, not snow.

        Must not require ``locked`` or ``analog_usable`` (0.28): free-run
        distant analog lives at corr 0.02–0.20.  Requiring both usable
        *and* ``not analog_ok`` made ``_apply_afc`` unreachable.
        """
        if frame is None or cvbs.raster_is_black(frame):
            return False
        return adaptive_if.analog_line_present(frame.line_rate)

    def _remember_good_frame(self, frame: cvbs.Frame | None, pic=None) -> None:
        if frame is None or cvbs.raster_is_black(frame):
            return
        if not cvbs.analog_usable(frame, pic=pic):
            return
        old_luma = (
            self._lock_good.luma if self._lock_good is not None else None
        )
        if (
            old_luma is None
            or old_luma.shape != frame.luma.shape
            or old_luma.dtype != np.uint8
        ):
            old_luma = np.empty(frame.luma.shape, dtype=np.uint8)
        np.copyto(old_luma, frame.luma, casting="unsafe")
        self._lock_good = cvbs.Frame(
            luma=old_luma,
            line_rate=float(frame.line_rate),
            lines=int(frame.lines),
            standard=str(frame.standard),
            locked=bool(frame.locked),
            field_parity=frame.field_parity,
            free_run=False,
        )
        self._lock_good_at = time.monotonic()

    def _fresh_lock_good(self, max_age_s: float = 0.25) -> cvbs.Frame | None:
        if self._lock_good is None:
            return None
        if time.monotonic() - self._lock_good_at > max(0.0, float(max_age_s)):
            self._lock_good = None
            self._lock_good_at = 0.0
            return None
        return self._lock_good

    def _analog_lock_present(self, assembler=None) -> bool:
        """True when H-line structure is still visible on the LOCK stream."""
        assembler = self._field_assembler if assembler is None else assembler
        if assembler is None:
            return False
        present = getattr(assembler, "line_comb_present", None)
        if callable(present):
            return bool(present())
        return assembler.period is not None

    def _held_lock_good(
        self,
        assembler=None,
        *,
        now: float | None = None,
        hold_s: float = 5.0,
        grace_s: float = 0.25,
    ) -> cvbs.Frame | None:
        """Keep the last analog_usable field while LOCK still has analog.

        Field cadence is ~20 ms; 1–2 s without a new complete field must
        still show the last picture if the H comb is present.  Snow/free_run
        only after analog is gone (no comb, beyond a short grace) or the
        long hold timeout.
        """
        if self._lock_good is None:
            return None
        now_m = time.monotonic() if now is None else float(now)
        age = now_m - float(self._lock_good_at or 0.0)
        present = self._analog_lock_present(assembler)
        limit = float(hold_s) if present else float(grace_s)
        if age > limit:
            self._lock_good = None
            self._lock_good_at = 0.0
            return None
        return self._lock_good

    def _copy_lock_frame(self, frame: cvbs.Frame) -> cvbs.Frame:
        luma = np.array(frame.luma, copy=True)
        return streaming.cvbs_frame(
            luma=luma,
            line_rate=float(frame.line_rate),
            lines=int(frame.lines),
            standard=str(frame.standard),
            locked=bool(frame.locked),
            field_parity=frame.field_parity,
            free_run=bool(getattr(frame, "free_run", False)),
            field_t0=getattr(frame, "field_t0", None),
            incomplete=bool(getattr(frame, "incomplete", False)),
        )

    def _choose_lock_display(
        self,
        frame: cvbs.Frame | None,
        assembler,
        vcfg: dict,
        *,
        now: float | None = None,
    ) -> cvbs.Frame | None:
        """Latest complete analog field, else last-good, else bounded snow."""
        now_m = time.monotonic() if now is None else float(now)
        if frame is not None and not bool(getattr(frame, "free_run", False)):
            return frame
        hold_s = float(vcfg.get("lock_good_hold_s", 5.0))
        held = self._held_lock_good(
            assembler, now=now_m, hold_s=hold_s,
        )
        if held is not None:
            return self._copy_lock_frame(held)
        snow_period = max(
            0.05, float(vcfg.get("snow_interval_ms", 120.0)) / 1000.0,
        )
        if now_m - self._snow_frame_at < snow_period:
            return None
        snow = (
            assembler.latest_free_run()
            if assembler is not None else None
        )
        if snow is not None:
            self._snow_frame_at = now_m
        return snow

    def _lock_holding(self) -> bool:
        """Period is known and either tracking or sticky field-start."""
        assembler = self._field_assembler
        if assembler is not None and assembler.period is not None:
            return True
        st = self._lock_state
        if st is None or st.period is None:
            return False
        return int(st.lost) == 0 or bool(getattr(st, "sticky", False))

    def _apply_afc(self, base, fs: float, off: float, ch_bw: float,
                   deviation: float, vcfg: dict, *,
                   pic_locked: bool = False, freeze_rf: bool = False):
        """Щокадрова дешева AFC: лише цифровий зсув каналайзера.

        lock_target після «Стати» — якір оператора. freq_error (середина
        перцентилів ЧМ-відео) зміщена в бік синхри/спідниці і з'їжджала
        з картинки (3080→3075). На межі Найквіста — стоп, не ±2 МГц hunt.
        Wide hunt / RF ±2 MHz freeze while `pic_locked` — they tear.
        Pegged + locked raster: `_maybe_rf_snap` once moves RF by digital
        `_afc` and zeros the mixer (hits/pin follow the new lock_target).
        ``freeze_rf`` (good analog_ok picture) still ticks the mixer in
        the deadband but never nudges the LO.
        """
        err = demod.freq_error_from_demod(base, deviation)
        self._last_err = err
        dead = float(vcfg.get("afc_deadband_hz", 80e3))
        dig_lim = self._digital_afc_lim(fs, off, ch_bw, vcfg)
        if abs(err) <= dead or dig_lim < 50e3:
            self._update_afc_peg(
                vcfg, dig_lim, pic_locked=pic_locked, freeze_rf=freeze_rf)
            return
        max_step = float(vcfg.get("afc_max_step_hz", 0.25e6))
        gain = float(vcfg.get("afc_gain", 0.5))
        step = float(np.clip(err * gain, -max_step, max_step))
        self._afc = max(-dig_lim, min(dig_lim, self._afc + step))
        self._update_afc_peg(
            vcfg, dig_lim, pic_locked=pic_locked, freeze_rf=freeze_rf)

    def _update_afc_peg(self, vcfg: dict, dig_lim: float, *,
                        pic_locked: bool = False,
                        freeze_rf: bool = False) -> None:
        self._afc_pegged = scan_view.afc_is_pegged(self._afc, dig_lim)
        if freeze_rf:
            self._afc_nudge = False
            self._afc_peg_n = 0
            offs = vcfg.get("hunt_offsets_mhz") or [0.25]
            self._hunt_span = scan_view.hunt_span_hz(offs, pegged=False)
            return
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
        if self._motor_sdr_guarded():
            return
        span = float(self._hunt_span or 2e6)
        delta = float(np.clip(self._last_err, -span, span))
        if abs(delta) < 80e3:
            return
        self.state.lock_target = float(self.state.lock_target) + delta
        self.state.tuned_hz = self.state.lock_target
        self._afc = 0.0
        self._lock_tuned = None
        self._lock_gen += 1
        self._adaptive_if.reset("RF anchor change")
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
        if self._motor_sdr_guarded():
            return False
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
        self._adaptive_if.reset("RF snap")
        self._rf_snap_at = now
        self._afc_peg_n = 0
        self._afc_pegged = False
        self._afc_nudge = False
        self._emit("notice", {"level": "ok", "text":
            f"RF snap {delta/1e6:+.2f} МГц → {new_hz/1e6:.2f}"})
        self._emit_state()
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
                   ch_bw: float, vcfg: dict, frame: cvbs.Frame | None,
                   hold_afc: bool = False, analog_ok: bool = False,
                   pic=None):
        """Фоновий ±0.25 МГц, не на нитці decode.

        Лише коли картинка вже була і оцінка впала. Перші кадри LOCK
        не чіпаємо — інакше hunt краде BLAS і трекінг не засідається.
        Visible analog (hold_afc / analog_ok) must not start a hunt —
        that walked 3430→3431 while sticky already had a picture.
        """
        if not vcfg.get("hunt", True):
            return
        if hold_afc or analog_ok or self._hunt_hold:
            return
        # сніг / немає кадру — не смикаємо частоту і не крадемо BLAS
        if frame is None or not frame.locked:
            return
        slow_ms = float(vcfg.get("hunt_skip_if_decode_ms", 120))
        dec_ms = self._timings.get("decode")
        if slow_ms > 0 and dec_ms is not None and dec_ms >= slow_ms:
            return
        if self._hunt_th is not None and self._hunt_th.is_alive():
            return
        # перші кадри LOCK — лише трекінг; hunt_after_lock раніше
        # НАВПАКИ частішав пошук і садив fps до ~3
        wait = int(vcfg.get("hunt_after_lock", 24))
        if self._lock_n < max(8, wait):
            return
        if pic is None:
            pic = cvbs.score_picture(frame)
        skip = float(vcfg.get("hunt_skip_if_score", 0.70))
        hold = float(vcfg.get("hunt_hold_score", scan_view.HUNT_HOLD_SCORE))
        analog_ok = cvbs.analog_usable(frame, pic=pic)
        if scan_view.freeze_lock_afc(
                analog_ok=analog_ok and not cvbs.raster_is_black(frame),
                pic_locked=bool(frame.locked) and not cvbs.raster_is_black(frame),
                pic_score=float(pic.value),
                row_corr=float(pic.row_corr),
                luma_mean=cvbs.luma_mean(frame),
                min_score=hold):
            return
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

        # Analog became usable after this thread started: drop remaining
        # mix trials so hunt does not keep BLAS while LOCK is decoding.
        if self._hunt_hold:
            return 0.0, 0.0, 0.0
        cur = score_mix(mix0)
        best_off = 0.0
        best = cur
        for trial in trials:
            if self._hunt_hold:
                return 0.0, cur.value, best.value
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

    @staticmethod
    def _adaptive_if_enabled(vcfg: dict) -> bool:
        """Code-default adaptive mode; explicit false keeps the legacy path."""
        raw = vcfg.get("adaptive_if", True)
        if isinstance(raw, str):
            return raw.strip().lower() not in ("0", "false", "off", "manual")
        return bool(raw)

    def _adaptive_if_snapshot(self) -> dict:
        if self._adaptive_if_enabled(self.cfg.get("video") or {}):
            return self._adaptive_if.diagnostics()
        return {
            "enabled": False,
            "selected": False,
            "score": 0.0,
            "reason": "explicitly disabled",
            "generation": self._adaptive_if.generation,
            "evaluations": self._adaptive_if.evaluations,
            "unusable_frames": 0,
        }

    def _adaptive_demod(
        self,
        iq: np.ndarray,
        fs: float,
        off: float,
        vcfg: dict,
        deviation: float,
    ) -> tuple[
        np.ndarray, np.ndarray, float, float, int,
        cvbs.LineRateHint | None, int,
    ]:
        """Select once after lock/loss, otherwise reuse cached DSP settings.

        This method never touches the SDR reader.  RF/sample-rate changes are
        handled by the existing command/reader path before DSP selection.
        """
        base_mix = -float(off) + float(self._afc)
        lifecycle = self._adaptive_if
        if lifecycle.needs_evaluation(fs):
            cap = adaptive_if.numeric_channel_cap(vcfg.get("channel_bw_hz"))
            # Blob centroid is only a proposal.  The selector scores zero/current
            # first and accepts this extra mix only with material hysteresis.
            proposal_bw = min(cap or fs * 0.9, fs * 0.9)
            nudge = scan_view.prelock_mix_hz(
                demod.blob_offset_hz(iq, fs),
                tracking=False, fs=fs, ch_bw=proposal_bw, off_hz=off,
                max_hz=float(vcfg.get("prelock_blob_max_hz", 2.5e6)),
            )
            result = adaptive_if.select(
                iq, fs,
                base_mix_hz=base_mix,
                channel_cap_hz=cap,
                deviation_hz=deviation,
                width=int(vcfg.get("width", 640)),
                proposed_nudge_hz=nudge,
                generation=lifecycle.generation + 1,
                offset_hysteresis=float(vcfg.get(
                    "adaptive_if_offset_hysteresis", adaptive_if.OFFSET_HYSTERESIS)),
            )
            selected = lifecycle.adopt(result.selection, fs)
            result.selection = selected
            return (
                result.base, result.fm_base, result.fs_ch,
                selected.effective_bw_hz,
                selected.decimation, result.line_hint,
                result.input_offset_samples,
            )

        selected = lifecycle.selection
        if selected is None:  # defensive; needs_evaluation normally catches it
            raise RuntimeError("adaptive IF selection missing")
        stage_timings: dict[str, float] = {}
        base, fm_base, fs_ch = adaptive_if.demodulate_cached(
            iq, fs, selected, base_mix_hz=base_mix, deviation_hz=deviation,
            timings_ms=stage_timings,
        )
        for stage, elapsed_ms in stage_timings.items():
            self._observe_timing(stage, elapsed_ms)
        trusted_hint = bool(
            selected.confirmed and selected.stable and selected.analog
            and cvbs.ANALOG_LINE_LO_HZ
            < float(selected.line_rate_hz)
            < cvbs.ANALOG_LINE_HI_HZ
        )
        line_hint = (
            cvbs.LineRateHint(
                line_hz=float(selected.line_rate_hz),
                attempted=True,
                confirmed=True,
            )
            if trusted_hint else None
        )
        return (
            base, fm_base, fs_ch, selected.effective_bw_hz, selected.decimation,
            line_hint, 0,
        )

    def _stream_lock_iq(
        self,
        iq: np.ndarray,
        abs_start_iq: int,
        fs: float,
        off: float,
        vcfg: dict,
        deviation: float,
        *,
        gap: bool = False,
    ) -> tuple[
        cvbs.Frame | None, np.ndarray, np.ndarray, float, float, int,
        cvbs.LineRateHint,
    ]:
        """Demodulate one unseen IQ range and assemble the newest full field."""
        adaptive_enabled = self._adaptive_if_enabled(vcfg)
        acquisition = adaptive_enabled and self._adaptive_if.needs_evaluation(fs)
        initial_hint: cvbs.LineRateHint | None = None
        if acquisition:
            # Selection is intentionally a one-generation acquisition cost.
            # Its preview arrays do not enter the streaming FIFO.
            preview = np.array(iq, copy=True)
            np.subtract(preview, np.mean(preview), out=preview)
            started = time.perf_counter()
            _, _, _, _, _, initial_hint, _ = self._adaptive_demod(
                preview, fs, off, vcfg, deviation,
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            self._stream_acquisition_ms += elapsed
            self._observe_timing("adaptive_if", elapsed)

        if adaptive_enabled:
            selected = self._adaptive_if.selection
            if selected is None:
                raise RuntimeError("adaptive IF selection missing")
            dec = int(selected.decimation)
            ch_bw = float(selected.effective_bw_hz)
            selected_offset = float(selected.offset_hz)
            # Analog comb is enough to seed PAL/NTSC.  Waiting for
            # adaptive "confirmed" let a 20 ms NTSC hunt stick, then
            # free-run re-hunted every snow frame (~1.7 s on the Pi).
            trusted_hz = streaming.analog_line_hint_hz(selected.line_rate_hz)
        else:
            headroom = float(vcfg.get("afc_digital_headroom_hz", 1.5e6))
            ch_bw = scan_view.lock_channel_bw(
                fs, self._lock_bw(float(self.state.lock_target or 0.0),
                                  float(vcfg.get("channel_bw_hz", 20e6))),
                off_hz=off, headroom_hz=headroom,
            )
            dec = scan_view.lock_decimation(fs, ch_bw)
            selected_offset = 0.0
            trusted_hz = None

        mix = -float(off) + float(self._afc) + selected_offset
        stream = self._stream_demod
        reset_pipeline = bool(
            stream is None
            or abs(stream.source_rate_hz - fs) > 1.0
            or stream.decimation != dec
            or abs(stream.deviation_hz - deviation) > 1.0
        )
        if reset_pipeline:
            stream = streaming.StreamingDemodulator(
                fs, decimation=dec, mix_hz=mix, deviation_hz=deviation,
            )
            stream.reset(abs_start_iq)
            self._stream_demod = stream
            self._field_assembler = streaming.StreamingFieldAssembler(
                stream.output_rate_hz,
                width=int(vcfg.get("width", 640)),
                line_hint_hz=trusted_hz,
                auto_levels=bool(vcfg.get("auto_levels", True)),
                sharpen=float(vcfg.get("sharpen", 0.0)),
            )
        else:
            stream.set_mix_hz(mix)

        demodulated = stream.process(iq, abs_start_iq, gap=gap)
        self._stream_new_samples += int(iq.size)
        for stage, elapsed_ms in stream.timings_ms.items():
            self._observe_timing(f"stream_{stage}", elapsed_ms)
        assembler = self._field_assembler
        if assembler is None:
            raise RuntimeError("streaming field assembler missing")
        assemble_t0 = time.perf_counter()
        frames = assembler.feed(demodulated, line_hint_hz=trusted_hz)
        self._observe_timing(
            "field_assembler", (time.perf_counter() - assemble_t0) * 1000.0,
        )
        if len(frames) > 1:
            dropped = len(frames) - 1
            assembler.metrics["fields_dropped"] += dropped
            self._stream_fields_dropped += dropped
        frame = frames[-1] if frames else None
        if frame is not None:
            frame.raw_lines = int(frame.lines)
            frame.luma = cvbs._h_crop(
                frame.luma,
                float(vcfg.get("crop_left_frac", cvbs.CROP_LEFT_FRAC)),
                int(vcfg.get("crop_bottom_lines", cvbs.CROP_BOTTOM_LINES)),
            )
            frame.lines = cvbs.published_lines(frame)
        if assembler.period is not None:
            if self._lock_state is None:
                self._lock_state = cvbs.DecodeState()
            self._lock_state.period = float(assembler.period)
            self._lock_state.standard = str(assembler.standard)
            self._lock_state.lost = 0
        hint = initial_hint or cvbs.LineRateHint(
            line_hz=trusted_hz or (
                assembler.sample_rate_hz / assembler.period
                if assembler.period is not None else None
            ),
            attempted=assembler.period is not None,
            confirmed=trusted_hz is not None,
        )
        return (
            frame, demodulated.samples, stream.last_fm,
            stream.output_rate_hz, ch_bw, dec, hint,
        )

    def _do_lock(self):
        # GIL: decode тримає його довго; sleep(0) віддає Starlette/WS.
        lock_t0 = time.perf_counter()
        time.sleep(0)
        self._drain_commands()
        if self._end_auto_peek_if_due():
            return
        if self.state.mode != "LOCK" or not self.state.lock_target:
            return
        fs = self._rate_plan.hardware_sample_rate_hz
        if self._motor_sdr_guarded():
            self._lock_during_motor_pwm(fs)
            return
        vcfg = self.cfg["video"]
        f = self.state.lock_target
 
        off = float(vcfg.get("lo_offset_hz", 0.0))
        bw = float(vcfg.get("channel_bw_hz", 20e6))
        if off and (abs(off) + bw / 2) > fs * 0.45:
            off = 0.0
 
        capture_s = float(vcfg.get("capture_ms", 120)) / 1000
        n_full = int(fs * capture_s)
        holding = self._lock_holding()
 
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
            holding = self._lock_holding()
        ring = self._ring
        if ring is None:
            return
        cursor = 0 if self._lock_cursor is None else int(self._lock_cursor)
        available = max(0, int(ring.filled) - cursor)
        acquisition = (
            self._adaptive_if_enabled(vcfg)
            and self._adaptive_if.needs_evaluation(fs)
        )
        minimum = min(n_full, int(ring.capacity)) if acquisition else 2048
        if available < minimum:
            time.sleep(0.004)
            self._end_auto_peek_if_due()
            return
        n_chunk = max(2048, int(fs * STREAM_LOCK_CHUNK_S))
        if acquisition:
            n = min(available, int(ring.capacity), max(minimum, n_full))
        else:
            n = min(available, int(ring.capacity), n_chunk)
        if self._lock_iq_buf is None or self._lock_iq_buf.size < n:
            self._lock_iq_buf = np.empty(n, dtype=np.complex64)
        snapshot_t0 = time.perf_counter()
        iq, abs_start_iq, next_cursor, gap = ring.read_since_into(
            self._lock_iq_buf, cursor, max_samples=n,
        )
        self._lock_cursor = next_cursor
        self._observe_timing(
            "ring_snapshot", (time.perf_counter() - snapshot_t0) * 1000.0)
        if gap:
            self._stream_gap_count += 1
            self._lock_state = cvbs.DecodeState()
            self._lock_good = None
            self._lock_good_at = 0.0
            self._acc = None
            self._acc_parity = None
        if len(iq) == 0:
            time.sleep(0.004)
            self._end_auto_peek_if_due()
            return
 
        self._lock_n += 1
 
        # Live catalog: spectrum_every_4 → 4, else YAML spectrum_every (1).
        # Sweep occupancy FFT does not read this.
        every = scan_view.lock_spectrum_every(vcfg)
        if self._rec is not None:
            every = max(every, int(vcfg.get("rec_spectrum_every", 6)))
        # Sticky analog: FFT every IQ steals BLAS from the 7 ms TBC path.
        if holding:
            every = max(every, 4)
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

        dev_cfg = float(vcfg.get("deviation_hz") or 0)
        deviation = dev_cfg if dev_cfg > 1e5 else max(1e5, bw / 4)
        frame, base, fm_base, fs_ch, ch_bw, dec, line_hint = \
            self._stream_lock_iq(
                iq, abs_start_iq, fs, off, vcfg, deviation, gap=gap,
            )
        self._lock_dec = dec
        period_hint = cvbs.analog_period_hint(self._lock_state, fs_ch)
        self._observe_timing("line_hunt", line_hint.elapsed_ms)

        # No field is a normal streaming state while the next boundary is
        # arriving.  JPEG still comes from _choose_lock_display (complete
        # field, last-good, or free-run snow) — never wait for analog_usable.

        afc_t0 = time.perf_counter()
        pic = cvbs.score_picture(frame)
        hold = float(vcfg.get("hunt_hold_score", scan_view.HUNT_HOLD_SCORE))
        analog_ok = cvbs.analog_usable(frame, pic=pic)
        luma_avg = cvbs.luma_mean(frame)
        visible = not cvbs.raster_is_black(frame)
        pic_locked = bool(pic.locked) and visible
        framed = cvbs.field_is_framed(frame)
        hold_afc = scan_view.freeze_lock_afc(
            analog_ok=analog_ok and visible,
            pic_locked=pic_locked,
            pic_score=float(pic.value),
            row_corr=float(pic.row_corr),
            luma_mean=luma_avg,
            min_score=hold) and framed
        freeze_hunt = hold_afc or (
            framed and scan_view.freeze_afc_hunt(
                pic_locked=pic_locked,
                pic_score=float(pic.value),
                min_score=hold)
        )
        # Tell an in-flight hunt to stop scoring mix trials. Do not bump
        # _lock_gen — that restarts the IQ reader and dumps sticky t0.
        self._hunt_hold = bool((analog_ok and framed) or hold_afc or freeze_hunt)

        self._maybe_auto_mgc(frame, iq, pic=pic)
        self._note_motor_guard_rx_success()
        self._take_hunt_result(fs, off, ch_bw, freeze=freeze_hunt or (analog_ok and framed))
        if (vcfg.get("afc", True) and self._video_ok_for_afc(frame, pic=pic)
                and not hold_afc):
            self._apply_afc(fm_base, fs, off, ch_bw, deviation, vcfg,
                            pic_locked=freeze_hunt, freeze_rf=bool(analog_ok and framed))
        # analog_ok pictures must not RF-snap (3429→3428→3431). Snow/peg
        # can still step the LO via _nudge_lock_for_peg / rf_snap_due.
        if not hold_afc and not analog_ok:
            self._maybe_rf_snap(
                pic_locked=pic_locked,
                pic_score=float(pic.value),
                digital_max_hz=self._digital_afc_lim(fs, off, ch_bw, vcfg),
            )
        self._observe_timing(
            "afc", (time.perf_counter() - afc_t0) * 1000.0)

        self._kick_hunt(
            iq, fs, off, ch_bw, vcfg, frame,
            hold_afc=hold_afc, analog_ok=analog_ok,
            pic=pic,
        )

        fallback_t0 = time.perf_counter()
        assembler = self._field_assembler
        show = self._choose_lock_display(frame, assembler, vcfg)
        self._observe_timing(
            "fallback", (time.perf_counter() - fallback_t0) * 1000.0)
        preblend_pic = (
            pic if show is frame else (
                cvbs.score_picture(show) if show is not None else pic
            )
        )
        if show is frame and show is not None:
            if not bool(getattr(show, "free_run", False)):
                self._remember_good_frame(show, pic=preblend_pic)
        snow = bool(getattr(show, "free_run", False)) if show is not None else False
        analog_flowing = show is not None and not snow

        if show is not None:
            blend_t0 = time.perf_counter()
            k = float(vcfg.get("average", 0.0))
            do_blend = k > 0 and not snow and show is frame
            if do_blend:
                cur = show.luma.astype(np.float32)
                blended, self._acc_parity = blend_same_field(
                    self._acc, self._acc_parity, cur, show.field_parity,
                    k, float(vcfg.get("motion_thresh", 24.0)),
                )
                self._acc = blended
                show.luma = np.clip(blended, 0, 255).astype(np.uint8)
            self._observe_timing(
                "blend", (time.perf_counter() - blend_t0) * 1000.0)
 
            show_pic = (
                preblend_pic if not do_blend
                else cvbs.score_picture(show)
            )
            keep_still = cvbs.should_save_still(show, show_pic)
            snapshot_request = None
            if self._snap and keep_still:
                self._snap = False
                snapshot_request = self._snapshot_request()
 
            self._last_video = {
                "freq_hz": f,
                "standard": show.standard,
                "line_rate": round(show.line_rate, 1),
                "lines": cvbs.published_lines(show),
                "locked": show.locked,
                "free_run": snow,
                "pic_score": round(show_pic.value, 3),
                "row_corr": round(show_pic.row_corr, 3),
                "afc_hz": round(self._afc, 0),
                "freq_err_hz": round(self._last_err, 0),
            }
            now = time.perf_counter()
            self._frame_ts, self._fps_ema = self._tick_fps(
                now, self._frame_ts, self._fps_ema)
            self._frame_seq += 1
            self._frame_mono = time.monotonic()
            self._last_video["frame_seq"] = self._frame_seq
            self._last_video["frame_mono_ms"] = int(self._frame_mono * 1000)
            submit_t0 = time.perf_counter()
            self.video_publisher.submit(
                show.luma,
                self._last_video,
                recorder=self._rec if keep_still else None,
                snapshot=snapshot_request,
                fmt=str(vcfg.get("stream_fmt", "jpeg")),
                quality=int(vcfg.get("stream_quality", 75)),
                method=int(vcfg.get("stream_method", 0)),
            )
            self._observe_timing(
                "submit", (time.perf_counter() - submit_t0) * 1000.0)
            if keep_still:
                self._touch_lock_picture(f, show, show_pic)

        if self._adaptive_if_enabled(vcfg):
            # Free-run snow still has an analog comb.  Treating missing
            # assembler.period as IF loss re-ran select() every ~8 frames
            # (~600 ms) and rebuilt the assembler, so LOCK sat at <1 fps.
            assembler = self._field_assembler
            frame_hz = getattr(frame, "line_rate", None) if frame is not None else None
            show_hz = getattr(show, "line_rate", None) if show is not None else None
            keep = adaptive_if.keep_cached_selection(
                line_rate_hz=frame_hz if frame_hz else show_hz,
                assembler_period=(
                    getattr(assembler, "period", None) if assembler is not None else None
                ),
                assembler_sample_rate_hz=(
                    getattr(assembler, "sample_rate_hz", None)
                    if assembler is not None else None
                ),
            )
            if keep:
                self._adaptive_if.observe(True)
            elif show is not None:
                adaptive_usable = bool(
                    not cvbs.raster_is_black(show)
                    and 14_000.0 < float(show.line_rate or 0.0) < 17_500.0
                    and float(show_pic.row_corr) >= 0.06
                )
                if self._adaptive_if.observe(adaptive_usable):
                    # Genuine sustained loss: next IQ gets one bounded rescore.
                    # The reader/ring continue uninterrupted.
                    self._lock_state = cvbs.DecodeState()
                    self._lock_good = None
                    self._lock_good_at = 0.0

        if show is not None:
            self._maybe_prune_empty_lock(
                show, show_pic, analog_flowing=analog_flowing,
            )
        else:
            self._maybe_prune_empty_lock(
                frame, pic, analog_flowing=False,
            )

        # шпаруватість: даємо процесору видихнути між знімками.
        # 0 — без штучної стелі fps (раніше 10 мс різали все, що вище ~15 к/с).
        idle = float(vcfg.get("idle_ms", 0))
        if idle > 0:
            time.sleep(idle / 1000)

        lock_ms = (time.perf_counter() - lock_t0) * 1000
        prev = self._timings.get("lock_total")
        self._timings["lock_total"] = (
            lock_ms if prev is None else prev * 0.8 + lock_ms * 0.2
        )
        iter_now = time.perf_counter()
        self._iteration_ts, self._iteration_fps_ema = self._tick_fps(
            iter_now, self._iteration_ts, self._iteration_fps_ema)
        now_mono = time.monotonic()
        if lock_ms >= 500 and now_mono - self._lock_warn_at >= 30:
            self._lock_warn_at = now_mono
            ring_age = float(getattr(self._ring, "last_write_age_s", 0.0))
            log.warning(
                "LOCK iteration slow total_ms=%.1f ring_age_s=%.3f "
                "samples=%d timings_ms=%s",
                lock_ms, ring_age, len(iq),
                {k: round(v, 1) for k, v in self._timings.items()},
            )
 
        self._end_auto_peek_if_due()

    def _end_auto_peek_if_due(self) -> bool:
        """Leave auto-LOCK even if the IQ ring never fills (silent decode)."""
        if not (self.state.auto and time.time() >= self.state.auto_until):
            return False
        self.state.auto = False
        self.state.lock_target = None
        self.state.mode = "SWEEP"
        self._acc = None
        self._acc_parity = None
        self._afc = 0.0
        self._lock_gen += 1
        self._adaptive_if.reset("auto peek ended")
        self._lock_score_peak = 0.0
        self._lock_good = None
        self._stop_reader()
        self._emit_state()
        return True

    def _patch_pub_scan(self, spec: dict, grid: dict) -> None:
        """Keep HTTP/heartbeat grid on the last dwell without a full state emit.

        snapshot() is a cached copy; without this, coverage only jumps when
        Scan/Lock calls _emit_state().
        """
        with self._pub_lock:
            if self._pub_snap is None:
                return
            snap = dict(self._pub_snap)
            snap["spectrum"] = spec
            snap["grid"] = grid
            snap["mode"] = self.state.mode
            snap["tuned_hz"] = self.state.tuned_hz
            snap["lock_target"] = self.state.lock_target
            snap["sweep_pos_hz"] = self.state.sweep_pos_hz
            snap["sweeps_done"] = self.state.sweeps_done
            self._pub_snap = snap
 
    def refresh_lock(self) -> None:
        """Serialize production refresh with the LOCK worker."""
        if self._thread is not None and self._thread.is_alive():
            self.command("refresh_lock")
            return
        self._refresh_lock_now()

    def _refresh_lock_now(self) -> None:
        """Re-arm LOCK tuning/decoder state on the same target.

        The shared hardware sample rate is deliberately untouched.  LO
        changes enqueue this path; decoder-only knobs are read directly by
        the next iteration. Does not stop a recording or bounce through sweep.
        """
        if self.state.mode != "LOCK" or not self.state.lock_target:
            return
        self._acc = None
        self._acc_parity = None
        self._lock_state = None
        self._lock_good = None
        self._lock_good_at = 0.0
        self._lock_dec = None
        self._lock_tuned = None
        self._last_video = None
        self._lock_n = 0
        self._lock_gen += 1
        self._adaptive_if.reset("lock tuning refresh")
        self._reset_lock_metrics()
        self._hunt_hold = False
        self._reset_auto_mgc()
        self._afc = 0.0
        self._last_err = 0.0
        self._afc_pegged = False
        self._afc_nudge = False
        self._afc_peg_n = 0
        self._rf_snap_at = 0.0
        self._hunt_out = None
        self._hunt_note = None

    def _lock_metrics_snapshot(self) -> dict:
        now = time.monotonic()
        frame_age = (
            None if self._frame_mono <= 0.0
            else max(0.0, (now - self._frame_mono) * 1000.0)
        )
        ws_age = (
            None if self._ws_frame_mono <= 0.0
            else max(0.0, (now - self._ws_frame_mono) * 1000.0)
        )
        publisher = self.video_publisher.metrics()
        if publisher["emitted_fps"] == 0.0 and self._ws_fps_ema > 0.0:
            publisher["emitted_fps"] = round(self._ws_fps_ema, 2)
        if publisher["emitted_frame_seq"] == 0 and self._ws_frame_seq > 0:
            publisher["emitted_frame_seq"] = int(self._ws_frame_seq)
            publisher["emitted_frame_age_ms"] = (
                None if ws_age is None else round(ws_age, 1)
            )
        assembler = self._field_assembler
        stream_diag = (
            assembler.diagnostics() if assembler is not None else {
                "fifo_samples": 0,
                "fifo_age_ms": 0.0,
                "fields_detected": 0,
                "fields_complete": 0,
                "fields_incomplete": 0,
                "fields_dropped": 0,
                "v_start_jitter_p95_lines": 0.0,
                "h_period_samples": None,
                "h_pll_correction_samples": 0.0,
                "parity": None,
                "field_index": 0,
            }
        )
        return {
            "fps": round(self._fps_ema, 2),
            "processed_fps": round(self._fps_ema, 2),
            "iteration_fps": round(self._iteration_fps_ema, 2),
            "frame_seq": int(self._frame_seq),
            "frame_mono_ms": (
                None if self._frame_mono <= 0.0 else int(self._frame_mono * 1000)
            ),
            "frame_age_ms": None if frame_age is None else round(frame_age, 1),
            "processed_frame_age_ms": (
                None if frame_age is None else round(frame_age, 1)
            ),
            "input_cursor": self._lock_cursor,
            "input_gap_count": int(self._stream_gap_count),
            "new_samples_consumed": int(self._stream_new_samples),
            "fifo_samples": int(stream_diag["fifo_samples"]),
            "fifo_age_ms": float(stream_diag["fifo_age_ms"]),
            "fields_detected": int(stream_diag["fields_detected"]),
            "fields_complete": int(stream_diag["fields_complete"]),
            "fields_incomplete": int(stream_diag["fields_incomplete"]),
            "fields_dropped": int(stream_diag["fields_dropped"]),
            "v_start_jitter_p95_lines": float(
                stream_diag["v_start_jitter_p95_lines"]
            ),
            "h_period_samples": stream_diag["h_period_samples"],
            "h_pll_correction_samples": float(
                stream_diag["h_pll_correction_samples"]
            ),
            "field_parity": stream_diag["parity"],
            "field_index": int(stream_diag["field_index"]),
            "adaptive_acquisition_ms": round(
                float(self._stream_acquisition_ms), 1,
            ),
            "streaming": stream_diag,
            **publisher,
            "timings_ms": {
                **{k: round(v, 1) for k, v in self._timings.items()},
                "encode": publisher["video_encode_ms"],
            },
            "event_queue_depth": (
                int(self.events.qsize()) if hasattr(self.events, "qsize") else None
            ),
            "reader_timeouts": int(getattr(self.src, "timeouts", 0)),
            "reader_restarts": int(getattr(self.src, "stream_restarts", 0)),
        }

    def _build_snapshot(self) -> dict:
        metrics = self._lock_metrics_snapshot()
        mgc_hold_ms, mgc_hold_reason = self._auto_mgc_hold_snapshot()
        rate = self._rate_plan.diagnostics()
        return {
            "mode": self.state.mode,
            "tuned_hz": self.state.tuned_hz,
            "sweeps_done": self.state.sweeps_done,
            "lock_target": self.state.lock_target,
            "auto": self.state.auto,
            "sweep_auto_lock": self._sweep_auto_lock,
            "sweep_hold": not self._sweep_auto_lock,
            "sweep_generation": self._sweep_generation,
            "source": self.src.name,
            "hardware_sample_rate": rate["hardware_sample_rate_hz"],
            "requested_scan_sample_rate": rate["requested_scan_rate_hz"],
            "requested_video_sample_rate": rate["requested_video_rate_hz"],
            "effective_sweep_step_hz": rate["effective_sweep_step_hz"],
            "estimated_sc16_mb_s": rate["estimated_sc16_mb_s"],
            "hardware_rate_reason": rate["reason"],
            "hardware_rate_plan": rate,
            "engine_alive": self.worker_alive(),
            "engine_error": self._worker_failure,
            "sdr_state": self._sdr_state,
            "sdr_state_age_ms": self._sdr_state_age_ms(),
            "sdr_operation": self._sdr_operation,
            "sdr_last_operation": self._sdr_last_operation,
            "sdr_consecutive_errors": self._sdr_consecutive_errors,
            "sdr_timeouts": int(getattr(self.src, "timeouts", 0)),
            "sdr_restarts": int(getattr(self.src, "stream_restarts", 0)),
            "sdr_stream_state": getattr(self.src, "_stream_state", None),
            "sdr_stream_transitions": int(
                getattr(self.src, "_stream_state_changes", 0)),
            "sdr_disable_failures": int(
                getattr(self.src, "_stream_disable_failures", 0)),
            "sdr_recovery_attempts": self._sdr_recovery_attempts,
            "sdr_open_attempts": self._sdr_open_attempts,
            "sdr_open_failures": self._sdr_open_failures,
            "sdr_next_retry_ms": self._sdr_next_retry_ms,
            "sdr_last_error": self._sdr_last_error,
            "sdr_last_error_code": self._sdr_last_error_code,
            "sdr_last_error_time": self._sdr_last_error_time,
            "sdr_last_error_operation": self._sdr_last_error_operation,
            "sdr_last_recovery_action": self._sdr_last_recovery_action,
            "recording": self._rec is not None,
            "bias_tee": bool(getattr(self.src, "bias_tee", False)),
            "gain_db": float((self.cfg.get("sdr") or {}).get("gain_db", 0)),
            "auto_gain": bool((self.cfg.get("sdr") or {}).get("auto_gain", True)),
            "auto_mgc_hold_ms": mgc_hold_ms,
            "auto_mgc_hold_reason": mgc_hold_reason,
            "motor_sdr_guard": self._motor_guard_snapshot(),
            "rec_seconds": (round(time.time() - self._rec.started_at, 1)
            if self._rec else 0),
            **metrics,
            "overflows": int(getattr(self.src, "overflows", 0)),
            "clip_frac": round(float(getattr(self.src, "clip_frac", 0.0)), 4),
            "adc_rms": round(float(getattr(self.src, "adc_rms", 0.0) or 0.0), 4),
            "afc_hz": round(self._afc, 0),
            "freq_err_hz": round(self._last_err, 0),
            "afc_pegged": bool(self._afc_pegged),
            "hunt_span_hz": float(self._hunt_span),
            "lock_tuned_hz": self._lock_tuned,
            "adaptive_if": self._adaptive_if_snapshot(),
            "video": self._last_video,
            "last_frame_ref": self._last_frame_ref,
            "sweep_pos_hz": self.state.sweep_pos_hz,
            "spectrum": self._spectrum_view(),
            "grid": self._grid_view(),
            "inspect_dbg": list(self._insp_dbg),
            "sweep_diagnostics": list(self._sweep_dbg),
            "detections": self._published_detections(),
            "rotator": self.rotator.status(),
        }

    def publish_snapshot(self) -> dict:
        data = self._build_snapshot()
        body = json.dumps(data, default=str)
        with self._pub_lock:
            self._pub_snap = data
            self._pub_json = body
            self._ws_json = '{"type":"state","data":' + body + "}"
        return data

    def snapshot(self) -> dict:
        """Last published copy, with live rotator/mode/video so HTTP does not freeze.

        LOCK decode writes _last_video every frame but rarely calls
        publish_snapshot (WS 'frame' events carry the picture). GET /api/state
        used to keep the cached video blob forever — knob_sweep then saw the
        same pic_score/line_rate on every PUT while fps (live-patched) moved.
        """
        with self._pub_lock:
            cached = dict(self._pub_snap) if self._pub_snap is not None else None
        if cached is None:
            cached = dict(self.publish_snapshot())
        cached["mode"] = self.state.mode
        cached["lock_target"] = self.state.lock_target
        cached["tuned_hz"] = self.state.tuned_hz
        cached["sweep_pos_hz"] = self.state.sweep_pos_hz
        cached["sweeps_done"] = self.state.sweeps_done
        cached["sweep_auto_lock"] = self._sweep_auto_lock
        cached["sweep_hold"] = not self._sweep_auto_lock
        cached["sweep_generation"] = self._sweep_generation
        rate = self._rate_plan.diagnostics()
        cached["hardware_sample_rate"] = rate["hardware_sample_rate_hz"]
        cached["requested_scan_sample_rate"] = rate["requested_scan_rate_hz"]
        cached["requested_video_sample_rate"] = rate["requested_video_rate_hz"]
        cached["effective_sweep_step_hz"] = rate["effective_sweep_step_hz"]
        cached["estimated_sc16_mb_s"] = rate["estimated_sc16_mb_s"]
        cached["hardware_rate_reason"] = rate["reason"]
        cached["hardware_rate_plan"] = rate
        cached["engine_alive"] = self.worker_alive()
        cached["engine_error"] = self._worker_failure
        cached["sdr_state"] = self._sdr_state
        cached["sdr_state_age_ms"] = self._sdr_state_age_ms()
        cached["sdr_operation"] = self._sdr_operation
        cached["sdr_last_operation"] = self._sdr_last_operation
        cached["sdr_consecutive_errors"] = self._sdr_consecutive_errors
        cached["sdr_timeouts"] = int(getattr(self.src, "timeouts", 0))
        cached["sdr_restarts"] = int(getattr(self.src, "stream_restarts", 0))
        cached["sdr_stream_state"] = getattr(self.src, "_stream_state", None)
        cached["sdr_stream_transitions"] = int(
            getattr(self.src, "_stream_state_changes", 0))
        cached["sdr_disable_failures"] = int(
            getattr(self.src, "_stream_disable_failures", 0))
        cached["sdr_recovery_attempts"] = self._sdr_recovery_attempts
        cached["sdr_open_attempts"] = self._sdr_open_attempts
        cached["sdr_open_failures"] = self._sdr_open_failures
        cached["sdr_next_retry_ms"] = self._sdr_next_retry_ms
        cached["sdr_last_error"] = self._sdr_last_error
        cached["sdr_last_error_code"] = self._sdr_last_error_code
        cached["sdr_last_error_time"] = self._sdr_last_error_time
        cached["sdr_last_error_operation"] = self._sdr_last_error_operation
        cached["sdr_last_recovery_action"] = self._sdr_last_recovery_action
        mgc_hold_ms, mgc_hold_reason = self._auto_mgc_hold_snapshot()
        cached["auto_mgc_hold_ms"] = mgc_hold_ms
        cached["auto_mgc_hold_reason"] = mgc_hold_reason
        cached["motor_sdr_guard"] = self._motor_guard_snapshot()
        cached.update(self._lock_metrics_snapshot())
        video = self._last_video
        cached["video"] = dict(video) if isinstance(video, dict) else video
        cached["last_frame_ref"] = self._last_frame_ref
        cached["afc_hz"] = round(self._afc, 0)
        cached["freq_err_hz"] = round(self._last_err, 0)
        cached["afc_pegged"] = bool(self._afc_pegged)
        cached["lock_tuned_hz"] = self._lock_tuned
        cached["adaptive_if"] = self._adaptive_if_snapshot()
        cached["clip_frac"] = round(float(getattr(self.src, "clip_frac", 0.0)), 4)
        cached["overflows"] = int(getattr(self.src, "overflows", 0))
        cached["adc_rms"] = round(float(getattr(self.src, "adc_rms", 0.0) or 0.0), 4)
        rot = getattr(self, "rotator", None)
        if rot is not None:
            cached["rotator"] = rot.status()
        return cached

    def snapshot_json(self) -> str:
        return json.dumps(self.snapshot(), default=str)

    def ws_state_json(self) -> str:
        return '{"type":"state","data":' + self.snapshot_json() + "}"

    def note_rotator(self, st: dict) -> None:
        if bool(st.get("moving") or st.get("busy")):
            with self._sdr_guard_lock:
                active = self._motor_guard_active
            if not active:
                self.begin_motor_guard()
        self._update_motor_guard_status(st, time.monotonic())
        with self._pub_lock:
            if self._pub_snap is None:
                return
            snap = dict(self._pub_snap)
            snap["rotator"] = st
        body = json.dumps(snap, default=str)
        with self._pub_lock:
            self._pub_snap = snap
            self._pub_json = body
            self._ws_json = '{"type":"state","data":' + body + "}"

    def _emit_state(self) -> None:
        self._emit("state", self.publish_snapshot())

    def reset_sweep_plan(self) -> None:
        """Drop queued cluster extras so the next dwell uses the new step."""
        self._dense_q.clear()
        self._dense_seen.clear()

    def _published_detections(self) -> list[dict]:
        need = int(self.cfg["scan"].get("confirm_hits", 1))
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
            "lines": cvbs.published_lines(frame) or int(pic.lines),
        }

    def _maybe_prune_empty_lock(
        self, frame, pic, analog_flowing: bool | None = None,
    ) -> None:
        if self.state.mode != "LOCK" or self.state.auto:
            return
        mhz = self._operator_lock_hit_mhz
        if mhz is None:
            return
        sample = self._lock_pic_sample(frame, pic)
        if analog_flowing is None:
            analog_flowing = (
                frame is not None
                and not bool(getattr(frame, "free_run", False))
                and self._frame_ts is not None
            )
        flowing = bool(analog_flowing)
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
        self._emit_state()

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
        self._emit_state()
 


