"""Антенний ротатор: Linux sysfs PWM (Type 2), як у SZ-конфігу серво.

Азимут — від лівого упору: 0° зліва, 90° посередині шкали, 180° справа
(ціле 0…180; крок ±1° або ±90°). Reverse лише інвертує PWM, не підписи.

На Windows / без /sys/class/pwm лише тримає ціль у пам'яті і прапорець
«обертання». GPIO 12 на Pi 5 — pwmchip0 / канал 0 після
dtoverlay=pwm,pin=12,func=4.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any


SYSFS_PWM = Path("/sys/class/pwm")
SLEW_DPS = 90.0  # speed=1 → 90° за ~1 с (індикатор, не замкнутий контур)
PWM_FRAME_S = 0.02  # 50 Hz servo frame; one duty write per frame while ramping


def _f(cfg: dict, *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in cfg and cfg[k] is not None:
            try:
                return float(cfg[k])
            except (TypeError, ValueError):
                continue
    return float(default)


def _i(cfg: dict, *keys: str, default: int = 0) -> int:
    return int(round(_f(cfg, *keys, default=float(default))))


def _b(cfg: dict, *keys: str, default: bool = False) -> bool:
    for k in keys:
        if k in cfg and cfg[k] is not None:
            v = cfg[k]
            if isinstance(v, str):
                return v.strip().lower() not in ("0", "false", "no", "off")
            return bool(v)
    return bool(default)


def display_span_deg(offset: float, max_range: float) -> float:
    """Шкала 0…2×offset, щоб offset (90°) був рівно посередині."""
    off = abs(float(offset))
    if off < 1.0:
        return max(float(max_range), 180.0)
    return 2.0 * off


def travel_from_azimuth(azimuth: float, *, max_range: float,
                        offset: float, reverse: bool = False) -> float:
    """Лівий упор → хід серво 0…max_range. reverse на хід не впливає."""
    del reverse
    az = max(0.0, min(display_span_deg(offset, max_range), float(azimuth)))
    return max(0.0, min(float(max_range), az))


def azimuth_from_travel(travel: float, *, max_range: float,
                        offset: float, reverse: bool = False) -> float:
    del reverse
    t = max(0.0, min(float(max_range), float(travel)))
    return max(0.0, min(display_span_deg(offset, max_range), t))


def pulse_us_from_travel(travel: float, *, max_range: float,
                         min_us: float, max_us: float,
                         reverse: bool = False) -> float:
    span = max(float(max_range), 1.0)
    frac = max(0.0, min(1.0, float(travel) / span))
    if reverse:
        frac = 1.0 - frac
    return float(min_us) + frac * (float(max_us) - float(min_us))


def quantize_command(*, current: float, span: float, grid_deg: float,
                     azimuth: float | None = None,
                     step: float | None = None) -> float:
    """Ціле 0…span. Сітка лише для |step| ≥ grid (±90). azimuth і ±1 — як є."""
    lo, hi = 0.0, float(span)
    if azimuth is not None:
        return max(lo, min(hi, round(float(azimuth))))
    if step is None:
        raise ValueError("потрібен azimuth або step")
    delta = float(step)
    nxt = float(current) + delta
    grid = abs(float(grid_deg)) or 90.0
    if abs(delta) + 0.5 >= grid:
        nxt = round(nxt / grid) * grid
    else:
        nxt = round(nxt)
    return max(lo, min(hi, nxt))


class AntennaRotator:
    """Type 2: sysfs PWM servo. Без sysfs — програмний no-op."""

    def __init__(self, cfg: dict | None = None, *,
                 sysfs_root: Path | str | None = None):
        cfg = dict(cfg or {})
        self.enable = _b(cfg, "enable", "Enable", default=False)
        typ = cfg.get("type", cfg.get("Type", "pwm"))
        self.type = "pwm" if str(typ).lower() in ("pwm", "2", "2.0") else str(typ)
        self.max_range = _f(cfg, "max_range", "MaxRange", default=220.0)
        self.gpio_pin = _i(cfg, "gpio_pin", "GpioPin", default=12)
        self.pwm_chip = _i(cfg, "pwm_chip", "PwmChip", default=0)
        self.pwm_channel = _i(cfg, "pwm_channel", "PwmChannel", default=0)
        self.frequency_hz = max(1.0, _f(cfg, "frequency_hz", "FrequencyHz",
                                        default=50.0))
        self.min_us = _f(cfg, "servo_min_pulse_width_us",
                         "ServoMinPulseWidthUs", default=300.0)
        self.max_us = _f(cfg, "servo_max_pulse_width_us",
                         "ServoMaxPulseWidthUs", default=2500.0)
        self.speed = _f(cfg, "speed", "Speed", default=1.0)
        self.reverse = _b(cfg, "reverse", "Reverse", default=False)
        self.azimuth_offset = _f(cfg, "azimuth_offset", "AzimuthOffset",
                                 default=90.0)
        self.step_deg = _f(cfg, "step_deg", "StepDeg", default=90.0)
        root = Path(sysfs_root) if sysfs_root is not None else SYSFS_PWM
        self._sysfs = root
        self._lock = threading.Lock()
        home = float(self.azimuth_offset) % 360.0
        if home > self.display_span():
            home = self.display_span() / 2.0
        self._target = home
        self._azimuth = home
        self._from_az = home
        self._move_start = 0.0
        self._move_end = 0.0
        self._pulse_us: float | None = None
        self._reason = ""
        self._exported = False
        self._cmd_gen = 0
        self._desired = home
        self._applied_us: float | None = None

    def display_span(self) -> float:
        return display_span_deg(self.azimuth_offset, self.max_range)

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._advance_locked()
            return self._status_locked()

    def set_azimuth(self, azimuth: float) -> dict[str, Any]:
        span = self.display_span()
        az = quantize_command(
            current=0.0, span=span, grid_deg=self.step_deg, azimuth=azimuth)
        with self._lock:
            self._cmd_gen += 1
            gen = self._cmd_gen
            self._desired = az
            st = self._prepare_move_locked(float(self._desired))
            target_us = self._pulse_us
            do_slew = bool(
                self.enable and st.get("available") and target_us is not None)
        if do_slew:
            self._slew_to_pulse(float(target_us), gen)
        with self._lock:
            return self._status_locked()

    def nudge(self, step_deg: float) -> dict[str, Any]:
        """Крок від поточної цілі.

        |step| ≥ сітки (90°) — як кнопки ±90, з прив'язкою до сітки.
        Інакше (±1°) — ціле число градусів без стрибка на 90.
        Два швидкі ±1 тримають один lock, щоб не з'їсти інкремент.
        """
        span = self.display_span()
        with self._lock:
            self._advance_locked()
            nxt = quantize_command(
                current=float(self._target), span=span,
                grid_deg=self.step_deg, step=step_deg)
            self._cmd_gen += 1
            gen = self._cmd_gen
            self._desired = nxt
            st = self._prepare_move_locked(nxt)
            target_us = self._pulse_us
            do_slew = bool(
                self.enable and st.get("available") and target_us is not None)
        if do_slew:
            self._slew_to_pulse(float(target_us), gen)
        with self._lock:
            return self._status_locked()

    def _pulse_for_azimuth(self, azimuth: float) -> float:
        travel = travel_from_azimuth(
            azimuth, max_range=self.max_range,
            offset=self.azimuth_offset, reverse=self.reverse)
        return pulse_us_from_travel(
            travel, max_range=self.max_range,
            min_us=self.min_us, max_us=self.max_us,
            reverse=self.reverse)

    def _pwm_step_us(self) -> float:
        span_us = abs(float(self.max_us) - float(self.min_us))
        span_deg = max(float(self.max_range), 1.0)
        us_per_deg = span_us / span_deg
        return max(
            8.0,
            us_per_deg * SLEW_DPS * max(float(self.speed), 0.05) * PWM_FRAME_S,
        )

    def _slew_to_pulse(self, target_us: float, gen: int) -> None:
        """Ramp duty over servo frames. A single jump stalls the 5V rail."""
        target = float(target_us)
        with self._lock:
            current = self._applied_us
            if current is None:
                current = self._pulse_for_azimuth(self._from_az)
            step = self._pwm_step_us()
        while abs(target - current) > 0.5:
            with self._lock:
                if gen != self._cmd_gen:
                    return
            delta = target - current
            current += max(-step, min(step, delta))
            try:
                with self._lock:
                    if gen != self._cmd_gen:
                        return
                    self._apply_pwm_locked(current)
                    self._applied_us = current
            except OSError as exc:
                with self._lock:
                    self._reason = f"pwm: {exc}"
                return
            time.sleep(PWM_FRAME_S)
        try:
            with self._lock:
                if gen != self._cmd_gen:
                    return
                self._apply_pwm_locked(target)
                self._applied_us = target
                self._reason = ""
        except OSError as exc:
            with self._lock:
                self._reason = f"pwm: {exc}"

    def _prepare_move_locked(self, az: float) -> dict[str, Any]:
        self._advance_locked()
        self._begin_move_locked(az)
        self._pulse_us = self._pulse_for_azimuth(az)
        if not self.enable:
            self._reason = "вимкнено"
            return self._status_locked()
        avail, why = self._probe_locked()
        if not avail:
            self._reason = why
            return self._status_locked()
        self._reason = ""
        return self._status_locked()

    def close(self) -> None:
        """Не глушимо PWM — серво лишається на останньому куті."""
        return

    def _begin_move_locked(self, target: float) -> None:
        now = time.monotonic()
        delta = abs(float(target) - float(self._azimuth))
        self._from_az = float(self._azimuth)
        self._target = float(target)
        if delta < 0.5:
            self._azimuth = float(target)
            self._move_start = now
            self._move_end = now
            return
        spd = max(float(self.speed), 0.05)
        dur = max(0.15, delta / (SLEW_DPS * spd))
        self._move_start = now
        self._move_end = now + dur

    def _advance_locked(self) -> None:
        now = time.monotonic()
        if self._move_end <= self._move_start:
            self._azimuth = self._target
            return
        if now >= self._move_end:
            self._azimuth = self._target
            return
        span = self._move_end - self._move_start
        t = 0.0 if span <= 0 else (now - self._move_start) / span
        t = max(0.0, min(1.0, t))
        self._azimuth = self._from_az + (self._target - self._from_az) * t

    def _moving_locked(self) -> bool:
        return time.monotonic() < self._move_end

    def _status_locked(self) -> dict[str, Any]:
        if not self.enable:
            avail, why = False, "вимкнено"
        else:
            avail, why = self._probe_locked()
        moving = self._moving_locked()
        eta_ms = 0
        if moving:
            eta_ms = int(round(max(0.0, self._move_end - time.monotonic()) * 1000))
        reason = self._reason or (why if self.enable else "вимкнено")
        if moving and not reason:
            reason = "обертання"
        return {
            "enable": self.enable,
            "available": bool(self.enable and avail),
            "type": self.type,
            "azimuth": round(self._azimuth, 1),
            "target_azimuth": round(self._target, 1),
            "center_azimuth": round(self.azimuth_offset, 1),
            "display_max": round(self.display_span(), 1),
            "step_deg": round(self.step_deg, 1),
            "moving": moving,
            "busy": moving,
            "eta_ms": eta_ms,
            "max_range": self.max_range,
            "azimuth_offset": self.azimuth_offset,
            "reverse": self.reverse,
            "gpio_pin": self.gpio_pin,
            "pwm_chip": self.pwm_chip,
            "pwm_channel": self.pwm_channel,
            "frequency_hz": self.frequency_hz,
            "pulse_us": (None if self._pulse_us is None
                         else round(self._pulse_us, 1)),
            "speed": self.speed,
            "reason": reason,
        }

    def _chip_dir(self) -> Path:
        return self._sysfs / f"pwmchip{self.pwm_chip}"

    def _pwm_dir(self) -> Path:
        return self._chip_dir() / f"pwm{self.pwm_channel}"

    def _probe_locked(self) -> tuple[bool, str]:
        pwm = self._pwm_dir()
        if self._exported and pwm.is_dir():
            return True, ""
        chip = self._chip_dir()
        if not chip.is_dir():
            return False, "немає pwmchip0 — overlay pwm,pin=12 і reboot"
        if pwm.is_dir():
            self._exported = True
            return True, ""
        npwm_p = chip / "npwm"
        if npwm_p.exists() and _read_ns(npwm_p) == 0:
            return False, "pwmchip порожній — overlay pwm,pin=12 і reboot"
        export = chip / "export"
        if not export.exists():
            return False, f"немає {pwm} — overlay або права gpio"
        try:
            self._ensure_exported_locked()
            return True, ""
        except OSError as exc:
            return False, f"pwm export: {exc}"

    def _ensure_exported_locked(self) -> Path:
        pwm = self._pwm_dir()
        if pwm.is_dir():
            self._exported = True
            return pwm
        export = self._chip_dir() / "export"
        export.write_text(f"{self.pwm_channel}\n")
        deadline = time.monotonic() + 0.4
        while not pwm.is_dir() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not pwm.is_dir():
            raise OSError(f"export pwm{self.pwm_channel} не створив {pwm}")
        self._exported = True
        return pwm

    def _apply_pwm_locked(self, pulse_us: float) -> None:
        pwm = self._ensure_exported_locked()
        period_ns = int(round(1e9 / self.frequency_hz))
        duty_ns = int(round(max(0.0, pulse_us) * 1000.0))
        duty_ns = min(duty_ns, max(0, period_ns - 1))
        period_p = pwm / "period"
        duty_p = pwm / "duty_cycle"
        enable_p = pwm / "enable"
        cur_duty = _read_ns(duty_p)
        cur_period = _read_ns(period_p)
        if cur_period != period_ns:
            if cur_period <= 0 or period_ns < cur_duty:
                duty_p.write_text("0\n")
            period_p.write_text(f"{period_ns}\n")
        duty_p.write_text(f"{duty_ns}\n")
        if _read_ns(enable_p) != 1:
            enable_p.write_text("1\n")


def _read_ns(path: Path) -> int:
    try:
        return int((path.read_text() or "0").strip() or "0")
    except (OSError, ValueError):
        return 0
