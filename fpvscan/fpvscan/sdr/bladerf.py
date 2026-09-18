"""Драйвер bladeRF 2.0 micro напряму через libbladeRF.

Ніяких прошарків. ctypes до тієї самої бібліотеки, якою користується
bladeRF-cli.

  Ubuntu / Pi 5:  apt install libbladerf2 bladerf bladerf-fpga-hostedxa4
                  -> /usr/lib/aarch64-linux-gnu/libbladeRF.so.2
  Windows:        інсталятор Nuand bladeRF
                  -> C:\\Program Files\\bladeRF\\x64\\bladeRF.dll

Дає те, чого прошарки не дають: quick tune (перебудова за десятки
мікросекунд замість мілісекунд), доступ до міток часу потоку та до
реальних діапазонів підсилення конкретної плати.
"""
from __future__ import annotations

import ctypes as C
import logging
import os
import sys
import threading
import time

import numpy as np

from fpvscan.app.core.logging import sanitize_log_text

from .base import SdrSource


log = logging.getLogger(__name__)

# ---------------------------------------------------------------- константи

def CHANNEL_RX(n: int) -> int: return (n << 1) | 0
def CHANNEL_TX(n: int) -> int: return (n << 1) | 1

FORMAT_SC16_Q11 = 0
FORMAT_SC16_Q11_META = 1

RX_X1 = 0
RX_X2 = 2

GAIN_DEFAULT = 0
GAIN_MGC = 1            # ручне керування — те, що потрібно для декодування
GAIN_FASTATTACK_AGC = 2
GAIN_SLOWATTACK_AGC = 3
GAIN_HYBRID_AGC = 4

DIRECTION_RX = 0
ERR_TIMEOUT = -6
ERR_IO = -5
ERR_NODEV = -7
ERR_UNEXPECTED = -1
# libbladeRF may collapse a lower-level LIBUSB_ERROR_NO_DEVICE into its
# "unexpected" result after an internal restore/deinitialize sequence.
DEVICE_LOST_CODES = frozenset((ERR_UNEXPECTED, ERR_IO, ERR_NODEV, -17, -18))

RETUNE_NOW = 0
META_FLAG_RX_NOW = 1 << 31
META_STATUS_OVERRUN = 1 << 0

SC16_SCALE = 2048.0     # Q11: 12-бітний АЦП у форматі int16

# Розмір struct bladerf_quick_tune відрізняється між поколіннями плат
# (для xA4 це профіль швидкого захоплення RFIC). Виділяємо з запасом —
# нам вміст непотрібен, ми тільки передаємо його назад у бібліотеку.
QUICK_TUNE_BYTES = 128


class _Metadata(C.Structure):
    _fields_ = [("timestamp", C.c_uint64),
                ("flags", C.c_uint32),
                ("status", C.c_uint32),
                ("actual_count", C.c_uint),
                ("reserved", C.c_uint8 * 32)]


class _Range(C.Structure):
    _fields_ = [("min", C.c_int64), ("max", C.c_int64),
                ("step", C.c_int64), ("scale", C.c_float)]


class BladeRFError(RuntimeError):
    def __init__(self, message: str, *, code: int | None = None):
        super().__init__(sanitize_log_text(message))
        self.code = code


# ---------------------------------------------------------------- бібліотека

_LIB = None
_LIB_PATH = None

# Залежності, які лежать поруч із bladeRF.dll і без яких вона не
# завантажиться. Саме через них Windows видає «не знайдено модуль»
# на файл, який насправді існує.
WIN_DEPS = ("libusb-1.0.dll", "pthreadVC2.dll", "msvcr120.dll", "msvcp120.dll")


def _win_reg_dirs() -> list[str]:
    out = []
    try:
        import winreg
    except ImportError:
        return out
    keys = [(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Nuand LLC\bladeRF"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\bladeRF"),
            (winreg.HKEY_LOCAL_MACHINE,
             r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\bladeRF")]
    for root, path in keys:
        for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
            try:
                with winreg.OpenKey(root, path, 0,
                                    winreg.KEY_READ | view) as k:
                    for name in ("Path", "InstallLocation", "InstallDir"):
                        try:
                            v = winreg.QueryValueEx(k, name)[0]
                            if v:
                                out += [v, os.path.join(v, "x64"),
                                        os.path.join(v, "bin")]
                        except OSError:
                            pass
            except OSError:
                pass
    return out


def _search_dirs() -> list[str]:
    """Каталоги, де може лежати бібліотека, у порядку правдоподібності."""
    dirs = []
    env = os.environ.get("BLADERF_LIB_DIR")
    if env:
        dirs.append(env)

    if sys.platform.startswith("win"):
        bases = [os.environ.get("ProgramW6432"),
                 os.environ.get("ProgramFiles"),
                 os.environ.get("ProgramFiles(x86)"),
                 r"C:\Program Files", r"C:\Program Files (x86)"]
        subs = ["bladeRF", os.path.join("Nuand", "bladeRF"),
                os.path.join("bladeRF", "bin")]
        for b in filter(None, bases):
            for sub in subs:
                for arch in ("x64", "x86", ""):
                    dirs.append(os.path.join(b, sub, arch))
        dirs += _win_reg_dirs()
    else:
        dirs += ["/usr/lib/aarch64-linux-gnu", "/usr/lib/x86_64-linux-gnu",
                 "/usr/lib", "/usr/local/lib", "/lib/aarch64-linux-gnu"]

    dirs += [d for d in os.environ.get("PATH", "").split(os.pathsep) if d]

    seen, out = set(), []
    for d in dirs:
        d = os.path.normpath(d)
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out


def _lib_names() -> list[str]:
    if sys.platform.startswith("win"):
        return ["bladeRF.dll"]
    return ["libbladeRF.so.2", "libbladeRF.so"]


def find_lib_files() -> list[str]:
    """Усі знайдені на диску кандидати. Порожньо = бібліотека не стоїть."""
    found = []
    for d in _search_dirs():
        for n in _lib_names():
            p = os.path.join(d, n)
            if os.path.isfile(p) and p not in found:
                found.append(p)
    return found


def missing_deps(lib_path: str) -> list[str]:
    """Залежності, яких бракує поруч із бібліотекою (тільки Windows)."""
    if not sys.platform.startswith("win"):
        return []
    d = os.path.dirname(lib_path)
    return [x for x in WIN_DEPS if not os.path.isfile(os.path.join(d, x))]


def _try_load(path: str):
    """Завантаження з урахуванням того, що з Python 3.8 ctypes на Windows
    не шукає залежності DLL по PATH — каталог треба назвати явно."""
    errs = []
    d = os.path.dirname(path)
    if sys.platform.startswith("win") and hasattr(os, "add_dll_directory"):
        try:
            with os.add_dll_directory(d):
                return C.CDLL(path), None
        except OSError as e:
            errs.append(str(e))
    try:
        return C.CDLL(path), None
    except OSError as e:
        errs.append(str(e))
    if sys.platform.startswith("win"):
        try:                       # winmode=0 повертає стару поведінку пошуку
            return C.CDLL(path, winmode=0), None
        except OSError as e:
            errs.append(str(e))
    return None, "; ".join(errs)


def load_lib(path: str | None = None):
    global _LIB, _LIB_PATH
    if _LIB is not None:
        return _LIB

    candidates = [path] if path else find_lib_files()
    if not candidates:
        raise BladeRFError(_not_installed_msg())

    errs = []
    for cand in candidates:
        lib, err = _try_load(cand)
        if lib is not None:
            _LIB, _LIB_PATH = lib, cand
            _declare(lib)
            return lib
        deps = missing_deps(cand)
        errs.append(f"  {cand}\n    {err}" +
                    (f"\n    поруч бракує: {', '.join(deps)}" if deps else ""))

    bits = 64 if C.sizeof(C.c_void_p) == 8 else 32
    raise BladeRFError(
        "Бібліотека знайдена на диску, але не завантажується:\n"
        + "\n".join(errs) +
        f"\n\nPython {bits}-бітний. Найчастіші причини:\n"
        "  1) розрядність не збігається — постав 64-бітний Python під x64-збірку;\n"
        "  2) поруч немає libusb-1.0.dll (ставиться разом з bladeRF, але\n"
        "     інколи не потрапляє в каталог) — скопіюй її туди;\n"
        "  3) не встановлений Visual C++ Redistributable.\n\n"
        "Точний шлях можна задати напряму: --lib \"C:\\...\\bladeRF.dll\"\n"
        "або змінною оточення BLADERF_LIB_DIR.")


def _not_installed_msg() -> str:
    where = "\n".join(f"  {d}" for d in _search_dirs()[:8])
    if sys.platform.startswith("win"):
        how = ("Постав bladeRF для Windows від Nuand — інсталятор кладе\n"
               "bladeRF.dll і bladeRF-cli.exe. Перевірити, що стало:\n"
               "  bladeRF-cli -e info\n"
               "Якщо ця команда працює, а Python бібліотеки не бачить —\n"
               "вкажи шлях: --lib \"C:\\Program Files\\bladeRF\\x64\\bladeRF.dll\"")
    else:
        how = "sudo apt install libbladerf2 bladerf bladerf-fpga-hostedxa4"
    return (f"{_lib_names()[0]} не знайдено на диску.\n\nШукав у:\n{where}\n\n{how}")


def lib_path() -> str | None:
    return _LIB_PATH


def _declare(lib):
    p = C.c_void_p
    sig = {
        "bladerf_open": ([C.POINTER(p), C.c_char_p], C.c_int),
        "bladerf_close": ([p], None),
        "bladerf_get_board_name": ([p], C.c_char_p),
        "bladerf_get_serial": ([p, C.c_char_p], C.c_int),
        "bladerf_is_fpga_configured": ([p], C.c_int),
        "bladerf_get_fpga_size": ([p, C.POINTER(C.c_int)], C.c_int),
        "bladerf_enable_module": ([p, C.c_int, C.c_bool], C.c_int),
        "bladerf_set_bias_tee": ([p, C.c_int, C.c_bool], C.c_int),
        "bladerf_get_bias_tee": ([p, C.c_int, C.POINTER(C.c_bool)], C.c_int),
        "bladerf_set_frequency": ([p, C.c_int, C.c_uint64], C.c_int),
        "bladerf_get_frequency": ([p, C.c_int, C.POINTER(C.c_uint64)], C.c_int),
        "bladerf_set_sample_rate": ([p, C.c_int, C.c_uint,
                                    C.POINTER(C.c_uint)], C.c_int),
        "bladerf_set_bandwidth": ([p, C.c_int, C.c_uint,
                                C.POINTER(C.c_uint)], C.c_int),
        "bladerf_set_gain": ([p, C.c_int, C.c_int], C.c_int),
        "bladerf_get_gain": ([p, C.c_int, C.POINTER(C.c_int)], C.c_int),
        "bladerf_set_gain_mode": ([p, C.c_int, C.c_int], C.c_int),
        "bladerf_get_gain_range": ([p, C.c_int,
                                    C.POINTER(C.POINTER(_Range))], C.c_int),
        "bladerf_sync_config": ([p, C.c_int, C.c_int, C.c_uint, C.c_uint,
                                C.c_uint, C.c_uint], C.c_int),
        "bladerf_sync_rx": ([p, C.c_void_p, C.c_uint,
                            C.POINTER(_Metadata), C.c_uint], C.c_int),
        "bladerf_get_timestamp": ([p, C.c_int, C.POINTER(C.c_uint64)], C.c_int),
        "bladerf_get_quick_tune": ([p, C.c_int, C.c_void_p], C.c_int),
        "bladerf_schedule_retune": ([p, C.c_int, C.c_uint64, C.c_uint64,
                                    C.c_void_p], C.c_int),
        "bladerf_strerror": ([C.c_int], C.c_char_p),
    }
    for name, (argtypes, restype) in sig.items():
        try:
            fn = getattr(lib, name)
        except AttributeError:
            continue
        fn.argtypes = argtypes
        fn.restype = restype


# ---------------------------------------------------------------- джерело

class BladeRF(SdrSource):
    name = "bladerf"

    def __init__(self, device: str = "", channel: int = 0,
                lib_path: str | None = None,
                gain_db: float = 30.0, agc: bool = False,
                num_buffers: int = 32, buffer_size: int = 32768,
                num_transfers: int = 16, timeout_ms: int = 750,
                bandwidth_ratio: float = 0.9,
                settle_us: float = 400.0,
                use_meta: bool = False,
                bias_tee: bool = False):
        self.lib = load_lib(lib_path)
        self.device = device
        self.ch = CHANNEL_RX(channel)
        self.gain_db = gain_db
        self.agc = agc
        self.num_buffers = num_buffers
        self.buffer_size = buffer_size          # має бути кратним 1024
        self.num_transfers = num_transfers
        self.timeout_ms = timeout_ms
        self.bandwidth_ratio = bandwidth_ratio
        self.settle_us = settle_us
        self.use_meta = use_meta                # потрібен для quick tune
        self.bias_tee = bias_tee                # потрібен для bias tee

        self._dev = C.c_void_p()
        self._fc = 0.0
        self._fs = 0.0
        self._raw = None                        # буфер int16, перевикористовується
        self._streaming = False
        self.timeouts = 0
        self.stream_restarts = 0
        self.io_errors = 0
        self.overflows = 0
        self.clip_frac = 0.0
        self.adc_rms = 0.0
        self._quick: dict[int, C.Array] = {}     # частота(Гц) -> профіль
        self._read_cancel_event = None
        self._consecutive_rx_errors = 0
        self._io_lock = threading.RLock()
        self._sync_rx_active = False
        self._stream_state = "INIT"
        self._stream_state_changes = 0
        self._stream_disable_failures = 0
        self._last_stream_log = 0.0
        self._device_lost_error: BladeRFError | None = None
        self._lost_handle_cleanup_done = False
        self._applied_gain_mode: int | None = None
        self._applied_gain_db: int | None = None
        self._rx_fragile = False

    # ---------- життєвий цикл ----------

    def _ensure_lifecycle(self) -> None:
        """Initialize lifecycle fields for normal and __new__-based test sources."""
        if not hasattr(self, "_io_lock"):
            self._io_lock = threading.RLock()
        if not hasattr(self, "_sync_rx_active"):
            self._sync_rx_active = False
        if not hasattr(self, "_streaming"):
            self._streaming = False
        if not hasattr(self, "_stream_state"):
            self._stream_state = "INIT"
        if not hasattr(self, "_stream_state_changes"):
            self._stream_state_changes = 0
        if not hasattr(self, "_stream_disable_failures"):
            self._stream_disable_failures = 0
        if not hasattr(self, "_last_stream_log"):
            self._last_stream_log = 0.0
        if not hasattr(self, "_device_lost_error"):
            self._device_lost_error = None
        if not hasattr(self, "_lost_handle_cleanup_done"):
            self._lost_handle_cleanup_done = False
        if not hasattr(self, "_applied_gain_mode"):
            self._applied_gain_mode = None
        if not hasattr(self, "_applied_gain_db"):
            self._applied_gain_db = None
        if not hasattr(self, "_rx_fragile"):
            self._rx_fragile = False
        if not hasattr(self, "_gain_limits"):
            self._gain_limits = (-15, 60)

    def _error_for_rc_locked(self, rc: int, what: str) -> BladeRFError:
        try:
            msg = self.lib.bladerf_strerror(rc)
            text = msg.decode(errors="replace") if msg else "?"
        except Exception:
            text = "?"
        return BladeRFError(f"{what}: {text} ({rc})", code=rc)

    def _mark_device_lost_locked(self, error: BladeRFError) -> None:
        """Invalidate then release a lost host handle exactly once."""
        dev = self._dev
        self._device_lost_error = error
        self._dev = C.c_void_p()
        self._applied_gain_mode = None
        self._applied_gain_db = None
        self._streaming = False
        self._sync_rx_active = False
        self._raw = None
        quick = getattr(self, "_quick", None)
        if quick is not None:
            quick.clear()
        self._set_stream_state("WAITING_DEVICE", force_log=True)
        if dev and not self._lost_handle_cleanup_done:
            # bladerf_close is the only libbladeRF host/libusb cleanup API.
            # Mark first so later recovery close calls can never repeat it.
            self._lost_handle_cleanup_done = True
            try:
                self.lib.bladerf_close(dev)
            except Exception as exc:
                log.warning(
                    "BladeRF lost-handle cleanup failed once: %s",
                    sanitize_log_text(exc),
                )

    def _check_rc_locked(self, rc: int, what: str) -> int:
        if rc >= 0:
            return rc
        error = self._error_for_rc_locked(rc, what)
        if rc in DEVICE_LOST_CODES:
            self._mark_device_lost_locked(error)
        raise error

    def _set_stream_state(self, state: str, *, force_log: bool = False) -> None:
        self._ensure_lifecycle()
        if state == self._stream_state:
            return
        self._stream_state = state
        self._stream_state_changes += 1
        now = time.monotonic()
        if force_log or now - self._last_stream_log >= 2.0:
            log.info(
                "BladeRF %s: transitions=%d restarts=%d disable_failures=%d",
                state,
                self._stream_state_changes,
                int(getattr(self, "stream_restarts", 0)),
                self._stream_disable_failures,
            )
            self._last_stream_log = now

    def open(self):
        self._ensure_lifecycle()
        with self._io_lock:
            dev = C.c_void_p()
            rc = self.lib.bladerf_open(
                C.byref(dev), self.device.encode() or None)
            self._dev = dev
            self._lost_handle_cleanup_done = False
            self._applied_gain_mode = None
            self._applied_gain_db = None
            self._check_rc_locked(rc, "bladerf_open")
            self._device_lost_error = None
            self._streaming = False
            fpga_rc = self.lib.bladerf_is_fpga_configured(self._dev)
            self._check_rc_locked(fpga_rc, "bladerf_is_fpga_configured")
            if fpga_rc == 0:
                raise BladeRFError(
                    "FPGA не завантажена. Ubuntu: apt install bladerf-fpga-hostedxa4; "
                    "або bladeRF-cli -l hostedxA4.rbf")
            self.set_gain(self.gain_db)
            if self.bias_tee:
                self.set_bias_tee(True)
            if self._fs:
                self.set_sample_rate(self._fs)

    def _disable_stream_locked(self, retries: int = 3) -> None:
        """Disable RX after all receives have returned; never ignore failure."""
        if self._sync_rx_active:
            raise RuntimeError("cannot disable BladeRF while sync_rx is active")
        if not self._streaming:
            return
        self._set_stream_state("STREAM_STOPPING")
        last_rc = 0
        for attempt in range(max(1, retries)):
            last_rc = self.lib.bladerf_enable_module(
                self._dev, self.ch, False)
            if last_rc >= 0:
                self._streaming = False
                return
            self._stream_disable_failures += 1
            if last_rc in DEVICE_LOST_CODES:
                self._check_rc_locked(
                    last_rc, "bladerf_enable_module(RX disable)")
            if attempt + 1 < retries:
                time.sleep(0.02 * (attempt + 1))
        self._check_rc_locked(last_rc, "bladerf_enable_module(RX disable)")

    def _start_stream_locked(self) -> None:
        if self._sync_rx_active:
            raise RuntimeError("cannot configure BladeRF while sync_rx is active")
        fmt = FORMAT_SC16_Q11_META if self.use_meta else FORMAT_SC16_Q11
        self._check_rc_locked(self.lib.bladerf_sync_config(
            self._dev, RX_X1, fmt, self.num_buffers, self.buffer_size,
            self.num_transfers, self.timeout_ms), "bladerf_sync_config")
        self._check_rc_locked(
            self.lib.bladerf_enable_module(self._dev, self.ch, True),
            "bladerf_enable_module(RX enable)")
        self._streaming = True
        self._set_stream_state("OK")

    def _config_stream(self):
        """Restart the synchronous stream in one serialized owner path.

        Викликати sync_config на активному потоці не можна: буфери
        перевиділяються, поки в них ще летять передачі USB, і плата
        зависає на керуючому інтерфейсі NIOS II. Тому спершу завжди
        гасимо модуль.
        """
        self._ensure_lifecycle()
        with self._io_lock:
            self._set_stream_state("STREAM_RECOVERING")
            self._disable_stream_locked()
            self._start_stream_locked()

    def close(self):
        self._ensure_lifecycle()
        with self._io_lock:
            if self._dev:
                self._disable_stream_locked()
                dev = self._dev
                self.lib.bladerf_close(dev)
                self._streaming = False
                self._dev = C.c_void_p()
                self._device_lost_error = None
                self._applied_gain_mode = None
                self._applied_gain_db = None
            self._set_stream_state("WAITING_DEVICE")

    def _need_dev(self):
        """Виклик у libbladeRF з нульовим вказівником — це негайний
        access violation, а не акуратна помилка. Тому перевіряємо."""
        if not self._dev:
            cause = getattr(self, "_device_lost_error", None)
            if cause is not None:
                raise BladeRFError(str(cause), code=cause.code)
            raise BladeRFError(
                "пристрій не відкритий (відвалився від USB і не піднявся)")

    def _recover(self, retries: int = 6):
        """Device reopen is owned exclusively by Engine's bounded wait loop."""
        del retries
        self._ensure_lifecycle()
        with self._io_lock:
            cause = self._device_lost_error
            if cause is not None:
                raise BladeRFError(str(cause), code=cause.code)
            raise BladeRFError(
                "перевідкриття BladeRF дозволене лише через WAITING_DEVICE")

    # ---------- інформація про плату ----------

    def info(self) -> dict:
        self._ensure_lifecycle()
        with self._io_lock:
            self._need_dev()
            buf = C.create_string_buffer(64)
            self._check_rc_locked(
                self.lib.bladerf_get_serial(self._dev, buf),
                "bladerf_get_serial")
            size = C.c_int(0)
            rc = self.lib.bladerf_get_fpga_size(self._dev, C.byref(size))
            if rc in DEVICE_LOST_CODES:
                self._check_rc_locked(rc, "bladerf_get_fpga_size")
            name = self.lib.bladerf_get_board_name(self._dev)
            return {
                "board": name.decode() if name else "?",
                "serial": buf.value.decode(errors="ignore"),
                "fpga_kle": size.value if rc >= 0 else 0,
                "gain_range": self.gain_range(),
            }

    def gain_range(self) -> tuple[int, int]:
        self._ensure_lifecycle()
        with self._io_lock:
            self._need_dev()
            rng = C.POINTER(_Range)()
            rc = self.lib.bladerf_get_gain_range(
                self._dev, self.ch, C.byref(rng))
            if rc in DEVICE_LOST_CODES:
                self._check_rc_locked(rc, "gain_range")
            if rc >= 0 and rng:
                limits = (int(rng.contents.min), int(rng.contents.max))
                self._gain_limits = limits
                return limits
            return tuple(self._gain_limits)

    # ---------- налаштування ----------

    def set_center_freq(self, hz: float):
        self._ensure_lifecycle()
        with self._io_lock:
            self._need_dev()
            rc = self.lib.bladerf_set_frequency(self._dev, self.ch, int(hz))
            # Device reopen belongs to Engine's single-owner recovery loop.
            self._check_rc_locked(rc, f"set_frequency({hz/1e6:.1f} МГц)")
            self._fc = float(hz)

    def set_sample_rate(self, hz: float):
        self._ensure_lifecycle()
        with self._io_lock:
            self._need_dev()
            if self._streaming and abs(self._fs - float(hz)) < 1.0:
                return
            self._disable_stream_locked()
            self._set_stream_state("RECONFIGURING")
            actual = C.c_uint(0)
            self._check_rc_locked(self.lib.bladerf_set_sample_rate(
                self._dev, self.ch, int(hz), C.byref(actual)),
                f"set_sample_rate({hz/1e6:.2f} Мвідл/с)")
            self._fs = float(actual.value)
            bw = C.c_uint(0)
            self._check_rc_locked(self.lib.bladerf_set_bandwidth(
                self._dev, self.ch, int(self._fs * self.bandwidth_ratio),
                C.byref(bw)), "set_bandwidth")
            self._bw = float(bw.value)
            self._start_stream_locked()

    def set_bias_tee(self, on: bool) -> bool:
        """Увімкнути/вимкнути 4.5В на антенному роз'ємі RX (Bias-T) — живлення
        зовнішнього LNA просто по коаксіалу, без окремого кабелю живлення.

        Підтримується платами bladeRF 2.0 (xA4/xA9) і достатньо новою
        бібліотекою/прошивкою. Якщо виклику нема — тихо повертаємо False,
        а не валимо сервер (той самий принцип, що й try/except у _declare()).
        """
        self._ensure_lifecycle()
        with self._io_lock:
            self._need_dev()
            fn = getattr(self.lib, "bladerf_set_bias_tee", None)
            if fn is None:
                if on:
                    print("[bladerf] bias-tee не підтримується цією збіркою "
                    "libbladeRF/прошивкою — потрібен bladerf_set_bias_tee",
                    flush=True)
                self.bias_tee = False
                return False
            was_streaming = self._streaming
            self._disable_stream_locked()
            self._set_stream_state("RECONFIGURING")
            rc = fn(self._dev, self.ch, C.c_bool(bool(on)))
            if rc < 0:
                if rc in DEVICE_LOST_CODES:
                    self._check_rc_locked(rc, "set_bias_tee")
                msg = self.lib.bladerf_strerror(rc)
                print(f"[bladerf] bias-tee: {msg.decode() if msg else '?'} ({rc}) "
                "— плата/прошивка не підтримує", flush=True)
                self.bias_tee = False
                if was_streaming:
                    self._start_stream_locked()
                return False
            self.bias_tee = bool(on)
            if was_streaming:
                self._start_stream_locked()
            return True

    def get_bias_tee(self) -> bool | None:
        """Реальний стан з плати. None — невідомо/не підтримується."""
        self._ensure_lifecycle()
        with self._io_lock:
            fn = getattr(self.lib, "bladerf_get_bias_tee", None)
            if fn is None:
                return None
            self._need_dev()
            val = C.c_bool(False)
            rc = fn(self._dev, self.ch, C.byref(val))
            if rc in DEVICE_LOST_CODES:
                self._check_rc_locked(rc, "get_bias_tee")
            return bool(val.value) if rc >= 0 else None

    def set_gain(self, db: float):
        self._ensure_lifecycle()
        with self._io_lock:
            self._need_dev()
            mode = GAIN_SLOWATTACK_AGC if self.agc else GAIN_MGC
            if mode != self._applied_gain_mode:
                self._check_rc_locked(
                    self.lib.bladerf_set_gain_mode(self._dev, self.ch, mode),
                    "set_gain_mode")
                self._applied_gain_mode = mode
                if self.agc:
                    self._applied_gain_db = None
            if not self.agc:
                lo, hi = self._gain_limits
                db = max(lo, min(hi, db))
                applied = int(db)
                if applied != self._applied_gain_db:
                    self._check_rc_locked(
                        self.lib.bladerf_set_gain(
                            self._dev, self.ch, applied),
                        "set_gain")
                    self._applied_gain_db = applied
            self.gain_db = db

    @property
    def center_freq(self): return self._fc

    @property
    def sample_rate(self): return self._fs

    @property
    def bandwidth(self): return getattr(self, "_bw", 0.0)

    # ---------- читання ----------

    def set_read_cancel_event(self, event) -> None:
        """Let the owner cancel the retry after a blocking RX timeout."""
        self._read_cancel_event = event

    def set_rx_fragile(self, fragile: bool) -> None:
        """During motor PWM, retry a timeout in-place — never rebuild USB RX."""
        self._ensure_lifecycle()
        self._rx_fragile = bool(fragile)

    def _raise_if_read_cancelled(self, where: str) -> None:
        event = self._read_cancel_event
        if event is not None and event.is_set():
            raise InterruptedError(f"BladeRF read cancelled {where}")

    def _sync_rx(self, n: int, meta: _Metadata) -> int:
        if self._sync_rx_active:
            raise RuntimeError("overlapping BladeRF sync_rx calls")
        self._sync_rx_active = True
        try:
            return self.lib.bladerf_sync_rx(
                self._dev, self._raw.ctypes.data_as(C.c_void_p), n,
                C.byref(meta) if self.use_meta else None, self.timeout_ms)
        finally:
            self._sync_rx_active = False

    def read(self, n: int) -> np.ndarray:
        self._ensure_lifecycle()
        with self._io_lock:
            self._need_dev()
            self._raise_if_read_cancelled("before receive")
            n = int(n)
            need = 2 * n
            if self._raw is None or self._raw.size < need:
                self._raw = np.empty(need, dtype=np.int16)
            meta = _Metadata()
            if self.use_meta:
                meta.flags = META_FLAG_RX_NOW
            rc = self._sync_rx(n, meta)
            if rc in DEVICE_LOST_CODES:
                self.io_errors += 1
                self._consecutive_rx_errors = getattr(
                    self, "_consecutive_rx_errors", 0) + 1
                self._check_rc_locked(rc, "sync_rx")
            if rc == ERR_TIMEOUT:
                # A timeout gets one bounded retry. Rebuilding the USB
                # stream (disable + sync_config) during motor PWM drops
                # the BladeRF; keep that restart off the fragile path.
                self.timeouts += 1
                self._consecutive_rx_errors = getattr(
                    self, "_consecutive_rx_errors", 0) + 1
                self._raise_if_read_cancelled("after timeout")
                if not bool(getattr(self, "_rx_fragile", False)):
                    self._config_stream()
                    self.stream_restarts += 1
                    self._raise_if_read_cancelled("before retry")
                rc = self._sync_rx(n, meta)
                if rc == ERR_TIMEOUT:
                    self.timeouts += 1
                    self._consecutive_rx_errors += 1
                elif rc == ERR_IO:
                    self.io_errors += 1
                    self._consecutive_rx_errors += 1
            self._check_rc_locked(rc, "sync_rx")
            self._consecutive_rx_errors = 0
            self._set_stream_state("OK")
            if self.use_meta and (meta.status & META_STATUS_OVERRUN):
                self.overflows += 1

            raw = self._raw[:need]
            # Контроль насичення АЦП. Обрізаний сигнал у спектрі виглядає
            # нормально, а на виході дискримінатора дає сміття замість
            # відео — тому міряємо це на кожному читанні.
            abs_raw = np.abs(raw)
            self.clip_frac = float(np.mean(abs_raw > 2000))
            scaled = raw.astype(np.float32) * (1.0 / SC16_SCALE)
            iq = scaled.view(np.complex64)
            self.adc_rms = float(np.sqrt(np.mean(
                iq.real * iq.real + iq.imag * iq.imag)))
            return iq

    def retune_and_read(self, hz: float, n: int) -> np.ndarray:
        self._raise_if_read_cancelled("before retune")
        self.set_center_freq(hz)
        skip = max(2048, int(self._fs * self.settle_us * 1e-6))
        self._raise_if_read_cancelled("before discard")
        self.read(skip)
        self._raise_if_read_cancelled("after discard")
        return self.read(n)

    # ---------- quick tune ----------

    def prime_quick_tune(self, freqs) -> int:
        """Заздалегідь зняти профілі швидкої перебудови для списку частот.

        Робиться один раз при старті: на кожній частоті плата
        налаштовується звичайним шляхом, а стан синтезатора зберігається.
        Далі перебудова на будь-яку з цих частот — це перезавантаження
        готового профілю замість повного захоплення ФАПЧ.
        """
        self._quick.clear()
        ok = 0
        for f in freqs:
            key = int(f)
            try:
                self.set_center_freq(key)
                buf = (C.c_uint8 * QUICK_TUNE_BYTES)()
                with self._io_lock:
                    self._need_dev()
                    self._check_rc_locked(
                        self.lib.bladerf_get_quick_tune(
                            self._dev, self.ch, buf),
                        "get_quick_tune")
                    self._quick[key] = buf
                ok += 1
            except BladeRFError as exc:
                if exc.code in DEVICE_LOST_CODES:
                    raise
                break        # плата або збірка бібліотеки не підтримує
        return ok

    def quick_retune(self, hz: float) -> bool:
        self._ensure_lifecycle()
        with self._io_lock:
            buf = self._quick.get(int(hz))
            if buf is None:
                return False
            rc = self.lib.bladerf_schedule_retune(
                self._dev, self.ch, RETUNE_NOW, int(hz), buf)
            if rc < 0:
                if rc in DEVICE_LOST_CODES:
                    self._check_rc_locked(rc, "schedule_retune")
                return False
            self._fc = float(hz)
            return True

    def retune_and_read_fast(self, hz: float, n: int) -> np.ndarray:
        """Перебудова профілем, якщо він знятий; інакше звичайним шляхом."""
        self._raise_if_read_cancelled("before fast retune")
        if self.quick_retune(hz):
            self.read(max(1024, int(self._fs * 20e-6)))   # ~20 мкс на осідання
            self._raise_if_read_cancelled("after fast discard")
            return self.read(n)
        return self.retune_and_read(hz, n)
