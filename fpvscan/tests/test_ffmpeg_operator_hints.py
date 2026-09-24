"""ffmpeg lookup hints — recording must fail with a command the operator can run."""
from __future__ import annotations

import inspect

from fpvscan import recorder


def test_linux_missing_hint_names_apt():
    """PATH miss on Pi/Ubuntu is `sudo apt install ffmpeg`, not the
    Windows winget paragraph. A generic «not found» sends people into
    the Nuand installer.
    """
    src = inspect.getsource(recorder.ffmpeg_path)
    assert "sudo apt install ffmpeg" in src
    assert "ffmpeg_path" in src


def test_windows_search_includes_winget_and_chocolatey():
    """winget puts ffmpeg.exe under LOCALAPPDATA\\Microsoft\\WinGet.
    Searching only Program Files\\ffmpeg misses a default Windows
    bench install; recording then dies with FfmpegMissing after LOCK.
    """
    src = inspect.getsource(recorder._win_candidates)
    assert "WinGet" in src
    assert "chocolatey" in src
    assert "ffmpeg.exe" in src
