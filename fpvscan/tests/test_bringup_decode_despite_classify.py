"""A failed line-rate check must not skip the PNG — weak analog still decodes."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "bringup.py").read_text(encoding="utf-8")


def test_bringup_still_decodes_when_classify_rejects():
    """classify_video is a 15.6 kHz line-rate gate. On a distant VTx
    the line is below prominence and is_video is False, but the
    60 ms grab still has sync tips. Returning there leaves the
    operator with «НЕ схоже на відео» and no PNG, so they debug
    ctypes / YAML instead of SNR. Decode is the same LOCK path
    (occ×1.6, bw/4) and must run anyway.
    """
    reject = SRC.index("if not sc.is_video:")
    chunk = SRC[reject:SRC.index("=== 4. декодування кадру")]
    assert "return" not in chunk
    assert "Кадр нижче все одно спробуємо" in chunk
    assert "cvbs.decode(base, fs_ch, width=640)" in SRC[SRC.index("=== 4. декодування кадру"):]
