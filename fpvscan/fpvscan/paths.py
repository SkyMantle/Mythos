"""Єдине місце, де вирішується, куди складати результати.

Усе, що програма створює — кадри, відео, записи ефіру — лягає в out/
у корені проєкту, а не туди, звідки її запустили. Інакше кадри і
гігабайтні .cf32 розповзаються по робочому каталогу.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"

PHOTOS = OUT / "photos"       # окремі кадри (WebP)
VIDEO = OUT / "video"         # відеозаписи (mp4/H.264)
CAPS = OUT / "caps"           # сирий ефір .cf32 + .json
LOGS = OUT / "logs"

FRAMES = PHOTOS               # старе ім'я, щоб не ламати наявні виклики


def ensure(p: Path) -> Path:
    """Створює каталог для файлу (або сам каталог) і повертає шлях."""
    (p.parent if p.suffix else p).mkdir(parents=True, exist_ok=True)
    return p


def resolve(user_path: str | None, default_dir: Path, default_name: str) -> Path:
    """Шлях від користувача — як є; інакше стандартне місце в out/.

    Абсолютний шлях або шлях із каталогом користувач задав свідомо,
    його не чіпаємо. Голе ім'я файлу кладемо в out/.
    """
    if not user_path:
        return ensure(default_dir / default_name)
    p = Path(user_path)
    if p.is_absolute() or len(p.parts) > 1:
        return ensure(p)
    return ensure(default_dir / p)


def stamped(prefix: str, ext: str, freq_hz: float | None = None) -> str:
    """Ім'я з часом і частотою: rec_20260827-142530_3896M.mp4"""
    t = datetime.now().strftime("%Y%m%d-%H%M%S")
    f = f"_{freq_hz/1e6:.0f}M" if freq_hz else ""
    return f"{t}{f}.{ext.lstrip('.')}" if not prefix else \
        f"{prefix}_{t}{f}.{ext.lstrip('.')}"