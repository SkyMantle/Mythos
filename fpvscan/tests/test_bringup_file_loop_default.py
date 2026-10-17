"""bringup --file must wrap a short .cf32, not EOFError mid-classify."""
from __future__ import annotations

import inspect
from pathlib import Path

from fpvscan.sdr.file import FileSource


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "bringup.py").read_text(encoding="utf-8")


def test_bringup_file_uses_constructor_loop_default():
    """#63 pinned factory `cfg.get("loop", True)` for Engine replay.
    bringup `--file` constructs `FileSource(a.file)` directly and
    reads 60 ms (`n_long = fs*0.06`) after a 262 k-sample level
    grab. A leftover `loop=False` EOFError's a 50 ms record.py
    dump during classify — the operator thinks the capture is
    empty. Constructor default True is the only loop on this path.
    """
    assert "src = FileSource(a.file)" in SRC
    file_branch = SRC.split("if a.file:", 1)[1].split("else:", 1)[0]
    assert "loop=" not in file_branch
    assert inspect.signature(FileSource.__init__).parameters["loop"].default is True
