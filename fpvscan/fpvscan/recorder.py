"""Запис декодованого відео.

Чому не MJPEG. У MJPEG кожен кадр стискається окремо, міжкадрова
надлишковість не використовується взагалі. У картинці з FPV-камери
сусідні кадри збігаються на 90+ відсотків, і H.264 з'їдає цю
надлишковість повністю.

Кодування йде через ffmpeg: сирі кадри пишуться йому в stdin, він
віддає mp4. Апаратного кодувальника H.264 у Pi 5 немає, але
програмний x264 на такій роздільності завантажує одне ядро на
одиниці відсотків.

Частота кадрів у нас нерівна - захоплення йде зі шпаруватістю. Тому
пише окрема нитка з рівним тактом: щотакту в потік іде останній
наявний кадр. Так час у файлі збігається з реальним, і відео не
"біжить".
"""
from __future__ import annotations

import os 
import shutil
import sys
import subprocess
import threading
import time
from pathlib import Path

import numpy as np


class FfmpegMissing(RuntimeError):
    pass

def _win_candidates() -> list[str]:
    import glob
    out = []
    la = os.environ.get("LOCALAPPDATA", "")
    if la:
        out.append(os.path.join(la,  "Microsoft", "WinGet",
                                "Links", "ffmpeg.exe"))
        out += glob.glob(os.path.join(
            la, "Microsoft", "WinGet", "Packages", "*FFmpeg*",
            "**", "bin", "ffmpeg.exe"), recursive=True)
    for base in filter(None, [os.environ.get("ProgramFiles"),
                              os.environ.get("ProgramFiles(x86)"),
                              os.environ.get("ProgramData")]):
        out.append(os.path.join(base, "ffmpeg", "bin", "ffmpeg.exe"))
        out.append(os.path.join(base, "chocolatey", "bin", "ffmpeg.exe"))
    return out


def ffmpeg_path(explicit: str | None = None) -> str:
    """Пошук ffmpeg: явний шлях -> PATH -> типові місця встановлення."""
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        raise FfmpegMissing(f"вказаний ffmpeg_path не існує: {explicit}")

    p = shutil.which("ffmpeg")
    if p:
        return p

    if sys.platform.startswith("win"):
        for cand in _win_candidates():
            if os.path.isfile(cand):
                return cand
        raise FfmpegMissing(
            "ffmpeg не знайдено ані в PATH, ані в типових каталогах.\n"
            "Якщо ти щойно поставив його через winget — PATH оновлюється\n"
            "тільки для НОВИХ процесів: закрий термінал і відкрий заново.\n"
            "Перевір командою:  where.exe ffmpeg\n"
            "Якщо шлях є, а програма його не бачить — пропиши явно\n"
            "у config.yaml, video.ffmpeg_path")
    raise FfmpegMissing("ffmpeg не знайдено. Ubuntu/Pi: sudo apt install ffmpeg")


class VideoRecorder:
    """Запис послідовності кадрів у mp4/H.264."""

    def __init__(self, path: Path, width: int, height: int,
                 fps: int = 5, crf: int = 24, preset: str = "veryfast", exe: str | None = None):
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.crf = int(crf)
        self.preset = preset
        self._proc: subprocess.Popen | None = None
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.frames_written = 0
        self.started_at = 0.0
        self.exe = exe

    def start(self):
        exe = ffmpeg_path(self.exe)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            exe, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "gray",
            "-s", f"{self.width}x{self.height}",
            "-framerate", str(self.fps), "-i", "pipe:0",
            "-an",
            "-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf),
            # yuv420p і парні розміри - інакше файл не відкриється
            # половиною програвачів і жодним браузером
            "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-movflags", "+faststart",
            "-g", str(self.fps * 2),
            str(self.path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.PIPE)
        self.started_at = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def push(self, luma: np.ndarray):
        """Віддати кадр. Розмір приводиться до заданого при старті."""
        if luma.shape != (self.height, self.width):
            luma = _resize_nearest(luma, self.height, self.width)
        with self._lock:
            self._latest = np.ascontiguousarray(luma, dtype=np.uint8)

    def _pump(self):
        period = 1.0 / self.fps
        nxt = time.perf_counter()
        while not self._stop.is_set():
            nxt += period
            time.sleep(max(0.0, nxt - time.perf_counter()))
            with self._lock:
                fr = self._latest
            if fr is None or self._proc is None or self._proc.stdin is None:
                continue
            try:
                self._proc.stdin.write(fr.tobytes())
                self.frames_written += 1
            except (BrokenPipeError, ValueError):
                break

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        err = b""
        if self._proc:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            if self._proc.stderr:
                err = self._proc.stderr.read() or b""
            self._proc = None
        size = self.path.stat().st_size if self.path.exists() else 0
        dur = max(1e-6, time.time() - self.started_at)
        return {
            "path": str(self.path),
            "bytes": size,
            "seconds": round(dur, 1),
            "frames": self.frames_written,
            "kbps": round(size * 8 / dur / 1000, 1),
            "error": err.decode(errors="ignore").strip(),
        }

    @property
    def active(self) -> bool:
        return self._proc is not None


def _resize_nearest(a: np.ndarray, h: int, w: int) -> np.ndarray:
    yi = (np.arange(h) * a.shape[0] // h).clip(0, a.shape[0] - 1)
    xi = (np.arange(w) * a.shape[1] // w).clip(0, a.shape[1] - 1)
    return a[yi][:, xi]