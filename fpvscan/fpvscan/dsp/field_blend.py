"""Same-field temporal denoise. Opposite PAL/NTSC fields must not be woven."""
from __future__ import annotations

import numpy as np


def blend_same_field(
    acc: np.ndarray | None,
    acc_parity: int | None,
    cur: np.ndarray,
    parity: int | None,
    strength: float,
    motion_thresh: float,
) -> tuple[np.ndarray, int | None]:
    """EMA only when both buffers are the same field (even or odd).

    Mixing field 0 with field 1 is the combing / «рядковість» the operator
    sees: a half-line vertical offset between interlaced fields.
    """
    if strength <= 0:
        return cur, parity
    reset = (
        acc is None
        or acc.shape != cur.shape
        or (parity is not None and acc_parity is not None and parity != acc_parity)
    )
    if reset:
        return cur.astype(np.float32, copy=True), parity
    a_static = 1.0 / max(1.0, float(strength))
    diff = np.abs(cur - acc)
    w = np.minimum(diff / max(1.0, float(motion_thresh)), 1.0)
    alpha = a_static + w * (1.0 - a_static)
    out = acc * (1.0 - alpha) + cur * alpha
    return out, parity
