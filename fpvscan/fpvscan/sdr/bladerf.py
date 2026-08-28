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
import os
import sys
import time

import numpy as np

from .base import SdrSource

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
    pass


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


def _ck(rc: int, what: str):
    if rc < 0:
        lib = load_lib()
        msg = lib.bladerf_strerror(rc)
        msg = msg.decode() if msg else "?"
        raise BladeRFError(f"{what}: {msg} ({rc})")
    return rc


# ---------------------------------------------------------------- джерело

class BladeRF(SdrSource):
    name = "bladerf"

    def __init__(self, device: str = "", channel: int = 0,
                lib_path: str | None = None,
                gain_db: float = 30.0, agc: bool = False,
                num_buffers: int = 32, buffer_size: int = 32768,
                num_transfers: int = 16, timeout_ms: int = 3500,
                bandwidth_ratio: float = 0.9,
                settle_us: float = 400.0,
                use_meta: bool = False):
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

        self._dev = C.c_void_p()
        self._fc = 0.0
        self._fs = 0.0
        self._raw = None                        # буфер int16, перевикористовується
        self._streaming = False
        self.timeouts = 0
        self.io_errors = 0
        self.overflows = 0
        self.clip_frac = 0.0
        self._quick: dict[int, C.Array] = {}     # частота(Гц) -> профіль

    # ---------- життєвий цикл ----------

    def open(self):
        _ck(self.lib.bladerf_open(C.byref(self._dev),
                                  self.device.encode() or None), "bladerf_open")
        if self.lib.bladerf_is_fpga_configured(self._dev) <= 0:
            raise BladeRFError(
                "FPGA не завантажена. Ubuntu: apt install bladerf-fpga-hostedxa4; "
                "або bladeRF-cli -l hostedxA4.rbf")
        self.set_gain(self.gain_db)
        if self._fs:
            self.set_sample_rate(self._fs)

    def _config_stream(self):
        """Переналаштування потоку.

        Викликати sync_config на активному потоці не можна: буфери
        перевиділяються, поки в них ще летять передачі USB, і плата
        зависає на керуючому інтерфейсі NIOS II. Тому спершу завжди
        гасимо модуль.
        """
        if self._streaming:
            try:
                self.lib.bladerf_enable_module(self._dev, self.ch, False)
            except Exception:
                pass
            self._streaming = False
            time.sleep(0.05)
        fmt = FORMAT_SC16_Q11_META if self.use_meta else FORMAT_SC16_Q11
        _ck(self.lib.bladerf_sync_config(
            self._dev, RX_X1, fmt, self.num_buffers, self.buffer_size,
            self.num_transfers, self.timeout_ms), "bladerf_sync_config")
        _ck(self.lib.bladerf_enable_module(self._dev, self.ch, True),
            "bladerf_enable_module")
        self._streaming = True

    def close(self):
        if self._dev:
            try:
                self.lib.bladerf_enable_module(self._dev, self.ch, False)
            except Exception:
                pass
            self.lib.bladerf_close(self._dev)
            self._streaming = False
            self._dev = C.c_void_p()

    def _need_dev(self):
        """Виклик у libbladeRF з нульовим вказівником — це негайний
        access violation, а не акуратна помилка. Тому перевіряємо."""
        if not self._dev:
            raise BladeRFError(
                "пристрій не відкритий (відвалився від USB і не піднявся)")

    def _recover(self, retries: int = 6):
        """Перевідкриття пристрою після зриву зв'язку.

        Плата може не просто «залипнути», а зникнути з шини і
        переперелічитись — тоді відкриття вдається не одразу. Тому
        кілька спроб зі зростаючою паузою. Якщо не вийшло, чесно
        падаємо, лишивши вказівник нульовим: далі його ловить
        _need_dev, а не процесор.
        """
        self.io_errors += 1
        fc, fs, g = self._fc, self._fs, self.gain_db
        try:
            self.close()
        except Exception:
            pass

        last = "?"
        for i in range(retries):
            time.sleep(0.4 * (i + 1))        # 0.4, 0.8, 1.2 ... с
            dev = C.c_void_p()
            rc = self.lib.bladerf_open(C.byref(dev),
                                       self.device.encode() or None)
            if rc >= 0 and dev:
                self._dev = dev
                self._streaming = False
                try:
                    self.set_gain(g)
                    self._fs = 0.0
                    self.set_sample_rate(fs)
                    if fc:
                        _ck(self.lib.bladerf_set_frequency(
                            self._dev, self.ch, int(fc)), "set_frequency")
                        self._fc = fc
                    return
                except BladeRFError as e:
                    last = str(e)
                    try:
                        self.close()
                    except Exception:
                        pass
                    continue
            msg = self.lib.bladerf_strerror(rc)
            last = msg.decode() if msg else str(rc)

        self._dev = C.c_void_p()
        raise BladeRFError(
            f"плату не вдалось підняти за {retries} спроб ({last}).\n"
            "Вона зникла з шини USB. Це живлення або порт, не програма:\n"
            "  - увімкни в порт USB 3.0 на материнській платі, без хабів\n"
            "  - інший кабель, короткий і якісний\n"
            "  - вимкни енергозбереження порту в Диспетчері пристроїв\n"
            "  - або живи від активного хаба USB 3.0 з власним БЖ")

    # ---------- інформація про плату ----------

    def info(self) -> dict:
        name = self.lib.bladerf_get_board_name(self._dev)
        buf = C.create_string_buffer(64)
        self.lib.bladerf_get_serial(self._dev, buf)
        size = C.c_int(0)
        try:
            self.lib.bladerf_get_fpga_size(self._dev, C.byref(size))
        except Exception:
            pass
        return {
            "board": name.decode() if name else "?",
            "serial": buf.value.decode(errors="ignore"),
            "fpga_kle": size.value,
            "gain_range": self.gain_range(),
        }

    def gain_range(self) -> tuple[int, int]:
        rng = C.POINTER(_Range)()
        try:
            _ck(self.lib.bladerf_get_gain_range(self._dev, self.ch,
                                                C.byref(rng)), "gain_range")
            return int(rng.contents.min), int(rng.contents.max)
        except Exception:
            return (-15, 60)

    # ---------- налаштування ----------

    def set_center_freq(self, hz: float):
        self._need_dev()
        rc = self.lib.bladerf_set_frequency(self._dev, self.ch, int(hz))
        if rc == ERR_IO:
            # Свіп по всьому діапазону змушує RFIC щокроку перемикати
            # смуги, і на частині контролерів USB керуючий канал це
            # інколи не витримує. Повне перевідкриття пристрою дешевше,
            # ніж падіння сервера.
            self._recover()
            rc = self.lib.bladerf_set_frequency(self._dev, self.ch, int(hz))
        _ck(rc, f"set_frequency({hz/1e6:.1f} МГц)")
        self._fc = float(hz)

    def set_sample_rate(self, hz: float):
        self._need_dev()
        if self._streaming and abs(self._fs - float(hz)) < 1.0:
            return
        actual = C.c_uint(0)
        _ck(self.lib.bladerf_set_sample_rate(self._dev, self.ch, int(hz),                                     C.byref(actual)),
            f"set_sample_rate({hz/1e6:.2f} Мвідл/с)")
        self._fs = float(actual.value)
        bw = C.c_uint(0)
        _ck(self.lib.bladerf_set_bandwidth(
            self._dev, self.ch, int(self._fs * self.bandwidth_ratio),
            C.byref(bw)), "set_bandwidth")
        self._bw = float(bw.value)
        self._config_stream()

    def set_gain(self, db: float):
        self._need_dev()
        mode = GAIN_SLOWATTACK_AGC if self.agc else GAIN_MGC
        _ck(self.lib.bladerf_set_gain_mode(self._dev, self.ch, mode),
            "set_gain_mode")
        if not self.agc:
            lo, hi = self.gain_range()
            db = max(lo, min(hi, db))
            _ck(self.lib.bladerf_set_gain(self._dev, self.ch, int(db)),
                "set_gain")
        self.gain_db = db

    @property
    def center_freq(self): return self._fc

    @property
    def sample_rate(self): return self._fs

    @property
    def bandwidth(self): return getattr(self, "_bw", 0.0)

    # ---------- читання ----------

    def read(self, n: int) -> np.ndarray:
        self._need_dev()
        n = int(n)
        need = 2 * n
        if self._raw is None or self._raw.size < need:
            self._raw = np.empty(need, dtype=np.int16)
        meta = _Metadata()
        if self.use_meta:
            meta.flags = META_FLAG_RX_NOW
        rc = self.lib.bladerf_sync_rx(
            self._dev, self._raw.ctypes.data_as(C.c_void_p), n,
            C.byref(meta) if self.use_meta else None, self.timeout_ms)
        if rc in (ERR_TIMEOUT, ERR_IO):
            # Одиничне зависання потоку лікується перезапуском модуля,
            # зрив зв'язку — перевідкриттям пристрою. Без цього
            # посипляться сотні однакових помилок libusb.
            if rc == ERR_IO:
                self._recover()
            else:
                self.timeouts += 1
                self._config_stream()
            rc = self.lib.bladerf_sync_rx(
                self._dev, self._raw.ctypes.data_as(C.c_void_p), n,
                C.byref(meta) if self.use_meta else None, self.timeout_ms)
        _ck(rc, "sync_rx")
        if self.use_meta and (meta.status & META_STATUS_OVERRUN):
            self.overflows += 1

        raw = self._raw[:need]
        # Контроль насичення АЦП. Обрізаний сигнал у спектрі виглядає
        # нормально, а на виході дискримінатора дає сміття замість
        # відео — тому міряємо це на кожному читанні.
        self.clip_frac = float(np.mean(np.abs(raw) > 2000))
        return (raw.astype(np.float32) / SC16_SCALE).view(np.complex64)

    def retune_and_read(self, hz: float, n: int) -> np.ndarray:
        self.set_center_freq(hz)
        skip = max(2048, int(self._fs * self.settle_us * 1e-6))
        self.read(skip)
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
                _ck(self.lib.bladerf_get_quick_tune(self._dev, self.ch, buf),
                    "get_quick_tune")
                self._quick[key] = buf
                ok += 1
            except BladeRFError:
                break        # плата або збірка бібліотеки не підтримує
        return ok

    def quick_retune(self, hz: float) -> bool:
        buf = self._quick.get(int(hz))
        if buf is None:
            return False
        rc = self.lib.bladerf_schedule_retune(
            self._dev, self.ch, RETUNE_NOW, int(hz), buf)
        if rc < 0:
            return False
        self._fc = float(hz)
        return True

    def retune_and_read_fast(self, hz: float, n: int) -> np.ndarray:
        """Перебудова профілем, якщо він знятий; інакше звичайним шляхом."""
        if self.quick_retune(hz):
            self.read(max(1024, int(self._fs * 20e-6)))   # ~20 мкс на осідання
            return self.read(n)
        return self.retune_and_read(hz, n)
